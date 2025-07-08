from functools import partial
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
import torch, time, os, pwd, argparse
import gemlite
from datasets import load_dataset
import nncf
from nncf.parameters import CompressWeightsMode
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Union, Dict
from tqdm import tqdm


WARMUP_PROMPTS = [
    "Write an essay about large language models.",
    "Tell me a funny joke!",
    "How to make a yummy chocolate cake?",
    "Who is Elon Musk?",
    "Write a Python code snippet that adds two numbers together.",
]

# Borrowed from https://github.com/mobiusml/hqq/blob/master/hqq/utils/generation_hf.py
class HFGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 1000,
        cache_size: Union[int, None] = None,
        do_sample: bool = False,
        temperature: float = 0.6,
        top_k: int = 5,
    ):
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.do_sample = do_sample
        self.temperature = temperature if self.do_sample else None
        self.top_k = top_k if self.do_sample else None
        self.use_cache = True  # False

        if do_sample:
            decode_one_token = self.decode_one_token_sampled
        else:
            decode_one_token = self.decode_one_token_no_sample

        # Setup cache
        self.max_new_tokens = max_new_tokens
        if cache_size is None:
            self.cache_size = self.next_multiple(self.max_new_tokens)
        else:
            self.cache_size = cache_size

        self.max_new_tokens = min(self.max_new_tokens, self.cache_size)

        self.setup_cache()

        if hasattr(self, "decode_one_token") is False:
            self.decode_one_token = decode_one_token

        self.init()  # check this: move this before setup_cache?

        ############################
        #Cuda Graph section
        self.static_input     = torch.zeros((1, 1), device=self.device, dtype=torch.int32)
        self.static_output    = torch.zeros((1, 1), device=self.device, dtype=torch.int32)
        self.cuda_graph       = None
        self.do_capture_graph = False
        ############################

    @torch.no_grad()
    def setup_cache(self):
        self.past_key_values = StaticCache(
            self.model.config, 1, self.cache_size, self.model.device, self.model.dtype
        )

    @torch.no_grad()
    def reset_cache(self):
        self.past_key_values.reset()

    def warmup(self):
        for prompt in WARMUP_PROMPTS:
            self.generate(prompt, print_tokens=False)
        return self

    def next_multiple(self, val):  # next power of 2
        vals = [2**i for i in range(5, 20)]  # [32, 64, ...]
        new_val = vals[[i for i in range(len(vals)) if (vals[i] - val) > 0][0]]
        return new_val

    def init(self):
        # Setup inference mode
        self.tokenizer.add_bos_token = False
        self.tokenizer.add_eos_token = False
        if not self.tokenizer.pad_token:
            self.tokenizer.add_special_tokens({"pad_token": "<<[PAD]>>"})
        self.tokenizer.padding_side = "right"
        self.model.eval()
        self.model.generation_config.cache_implementation = "static"
        self.model.config.use_cache = True

    # Copied from https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb
    def multinomial_sample_one_no_sync(self, probs_sort):
        q = torch.empty_like(probs_sort).exponential_(1)
        return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

    # Copied from https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb
    def logits_to_probs(self, logits, temperature=1.0, top_k=None):
        logits = logits / max(temperature, 1e-5)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            pivot = v.select(-1, -1).unsqueeze(-1)
            logits = torch.where(logits < pivot, -float("Inf"), logits)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    # Copied from https://gist.github.com/ArthurZucker/af34221def212259b43d55a2811d2dbb
    def sample(self, logits, temperature, top_k):
        probs = self.logits_to_probs(logits[:, -1], temperature, top_k)
        idx_next = self.multinomial_sample_one_no_sync(probs)
        return idx_next, probs

    def decode_one_token_no_sample(
        self,
        cur_token,
        input_pos,
        cache_position,
        past_key_values,
        temperature=None,
        top_k=None,
    ):
        out = self.model(
            cur_token,
            # position_ids=input_pos,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=self.use_cache,
        )
        logits, self.past_key_values = out.logits, out.past_key_values
        new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        return new_token

    def decode_one_token_sampled(
        self,
        cur_token,
        input_pos,
        cache_position,
        past_key_values,
        temperature=0.6,
        top_k=5,
    ):
        out = self.model(
            cur_token,
            # position_ids=input_pos,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=True,
            use_cache=self.use_cache,
        )
        logits, self.past_key_values = out.logits, out.past_key_values
        new_token = self.sample(logits, temperature=temperature, top_k=top_k)[0]
        return new_token

    # Setup cache and variables
    def setup(self, inputs, max_new_tokens):
        self.reset_cache()
        self.inputs = inputs
        self.batch_size, self.seq_length = self.inputs["input_ids"].shape
        self.cache_position = torch.arange(self.seq_length, device=self.device)
        self.generated_ids = torch.zeros(
            self.batch_size,
            self.seq_length + max_new_tokens + 1,
            dtype=torch.int,
            device=self.device,
        )
        self.generated_ids[:, self.cache_position] = self.inputs["input_ids"].to(
            torch.int
        )

    # Pre-fill phase
    def prefill(self):
        out = self.model(
            **self.inputs,
            cache_position=self.cache_position,
            past_key_values=self.past_key_values,
            return_dict=True,
            use_cache=self.use_cache,
        )
        logits, self.past_key_values = out.logits, out.past_key_values
        next_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
        self.generated_ids[:, self.seq_length] = next_token[:, 0]
        self.cache_position = torch.tensor([self.seq_length], device=self.device, dtype=torch.long)
        self.begin_gen_position = self.cache_position.item()
        return next_token

    # generate one token at a time
    def gen_next_token_raw(self, next_token):
        with sdpa_kernel([SDPBackend.MATH]):
            next_token = self.decode_one_token(
                next_token.clone(),
                None,
                cache_position=self.cache_position + 1,
                past_key_values=self.past_key_values,
                temperature=self.temperature,
                top_k=self.top_k,
            )
        self.cache_position += 1
        self.generated_ids[:, self.cache_position] = next_token.int()
        return next_token

    def gen_next_token(self, next_token):
        return self.gen_next_token_raw(next_token)

    def enable_cuda_graph(self, prompt = "Write an essay about large language models."):
        #Warm-up
        _ = self.generate(prompt, print_tokens=False)

        #Enable 
        self.gen_next_token = self.gen_next_token_withgraph_v2
        self.do_capture_graph = True

        #Capture
        _ = self.generate(prompt, print_tokens=False)

        return self

    def gen_next_token_withgraph_v2(self, next_token):
        if(self.do_capture_graph):
            self.static_input.copy_(next_token)
            self.stream = torch.cuda.Stream()
            torch.cuda.synchronize()

            #Warm-up
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    with sdpa_kernel([SDPBackend.MATH]):
                        _ = self.decode_one_token(
                            self.static_input,
                            None,
                            cache_position=self.cache_position + 1,
                            past_key_values=self.past_key_values,
                            temperature=self.temperature,
                            top_k=self.top_k,
                        )
                torch.cuda.synchronize()
  
            #Capture
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self.stream):
                self.cuda_graph.capture_begin()
                with sdpa_kernel([SDPBackend.MATH]):
                    out = self.decode_one_token(
                        self.static_input,
                        None,
                        cache_position=self.cache_position + 1,
                        past_key_values=self.past_key_values,
                        temperature=self.temperature,
                        top_k=self.top_k,
                    )
                    self.static_output.copy_(out)
                self.cuda_graph.capture_end()
            torch.cuda.synchronize()
            
            #Turn-off
            self.do_capture_graph = False
        else:
            self.static_input.copy_(next_token)
            self.cuda_graph.replay()

        next_token = self.static_output
        self.cache_position += 1
        self.generated_ids[:, self.cache_position] = next_token.int()
        return next_token

    def print_current_token(self, output_text_len):
        output_text = self.tokenizer.decode(self.generated_ids[0, self.begin_gen_position : self.cache_position + 1])
        printable_text = output_text[output_text_len:]
        output_text_len = len(output_text)
        print(printable_text, end="", flush=True)
        return output_text_len

    def next_token_iterator(self, next_token, max_new_tokens, verbose, print_tokens, cleanup=True):
        output_text, output_text_len = "", 0
        for i in tqdm(range(1, max_new_tokens), disable=(not verbose or print_tokens)):
            next_token = self.gen_next_token(next_token)

            if next_token[0].item() == self.tokenizer.eos_token_id:
                break

            # You need to keep track of the whole text, otherwise you lose spaces, makes everything much slower ¯\_(ツ)_/¯
            # https://github.com/huggingface/transformers/blob/b109257f4fb8b1166e7c53cc5418632014ed53a5/src/transformers/generation/streamers.py#L95-L114
            if print_tokens:
                output_text_len = self.print_current_token(output_text_len)

        input_tokens  = self.generated_ids[0, : self.begin_gen_position].cpu()
        output_tokens = self.generated_ids[0, self.begin_gen_position : self.cache_position].cpu()
        output_text   = self.tokenizer.decode(output_tokens)

        if cleanup:
            # model._reset_cache()
            del self.inputs, self.generated_ids, self.cache_position
            torch.cuda.empty_cache()

        return {
            "output_text": output_text,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
        }

    @torch.inference_mode()
    def generate(self, prompt, use_chat_template=True, verbose=True, print_tokens=False):
        self.setup(
            inputs=self.tokenize_prompt(prompt, use_chat_template=use_chat_template),
            max_new_tokens=self.max_new_tokens,
        )
        return self.next_token_iterator(self.prefill(), self.max_new_tokens, verbose, print_tokens)

    def generate_(self, prompt, use_chat_template=True, verbose=False, print_tokens=False):
        gen_out = self.model.generate(
            **self.tokenize_prompt(prompt, use_chat_template=use_chat_template),
            do_sample=self.do_sample,
            cache_implementation="static",
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=self.temperature,
            top_p=self.top_k,
            # use_cache=False,
        )[0]

        return {"output_text": self.tokenizer.decode(gen_out), "output_tokens": gen_out}

    def tokenize_prompt(self, prompt, use_chat_template=True):
        if use_chat_template:

            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        return self.tokenizer([prompt], return_tensors="pt").to(device=self.model.device)

def main(quantization="", mode=""):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_path = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="auto", 
        torch_dtype=torch.float16,
    ).to("cuda")

    model.eval()

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    def transform_fn(data, tokenizer):
            tokenized_text = tokenizer(data["text"], return_tensors="pt").to("cuda")
            inp = tokenized_text["input_ids"]
            attention_mask = tokenized_text["attention_mask"]

            position_ids = torch.cumsum(attention_mask, axis=1) - 1
            position_ids[attention_mask == 0] = 1

            return {
                "input_ids": inp,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
    
    quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))
    # model_input = model.dummy_inputs
    # quantization_dataset = nncf.Dataset([
    #     { k: v.to("cuda") for k, v in model_input.items() }
    # ])

    if mode == "int8_sym":
        mode = CompressWeightsMode.INT8_SYM

    elif mode == "int8_asym":
        mode = CompressWeightsMode.INT8_ASYM

    elif mode == "int4_sym":
        mode = CompressWeightsMode.INT4_SYM

    elif mode == "int4_asym":
        mode = CompressWeightsMode.INT4_ASYM

    if quantization == "gemlite":
        gemlite.set_autotune("max")
        config_file = f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
        gemlite.load_config(config_file)

        model = nncf.compress_weights(
            model,
            dataset=quantization_dataset,
            mode=mode,
            gemlite=True,
        )

    elif quantization == "nncf":
        model = nncf.compress_weights(
            model,
            dataset=quantization_dataset,
            mode=mode,
        )

    gen = HFGenerator(
        model,
        tokenizer,
        max_new_tokens=1024,
        do_sample=True,
    ).enable_cuda_graph()

    prompt = "What are the top 10 popular apps in China 2020?"

    for _ in range(3):
        out = gen.generate(prompt, print_tokens=False) 


    if quantization == "gemlite":
        gemlite.cache_config(config_file)

    for _ in range(5):
        t0 = time.perf_counter()
        out = gen.generate(prompt, print_tokens=False) 
        print(f"Time taken: {time.perf_counter() - t0:.2f}s")
    
    print(out["output_text"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression", type=str, default="", help="Compression, e.g. 'gemlite'")
    parser.add_argument("--mode", type=str, default="", help="Quantization mode, e.g. 'int8_sym'")
    args = parser.parse_args()
    main(quantization=args.quantization, mode=args.mode)