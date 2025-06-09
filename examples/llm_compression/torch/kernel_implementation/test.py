import argparse
import os
import pwd
import time
from functools import partial
import openvino as ov

import torch
from datasets import load_dataset
from nncf.parameters import CompressWeightsMode
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import datetime

from gemlite import GemLiteLinearTriton, DType
import gemlite
import nncf
from torch.profiler import profile, ProfilerActivity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_model_size(model):
    total_size = 0
    for param in model.parameters():
        total_size += param.numel() * param.element_size()
    return total_size / (1024 * 1024) 

def warmup(model, x):
    with torch.inference_mode():
        for _ in range(3):
            model(x["input_ids"])
        if device.type == "cuda":
            torch.cuda.synchronize()

def check(model, name, x):
    runs = 5
    with torch.inference_mode():
        for _ in range(runs):
            start = time.perf_counter()
            model(x["input_ids"])
            t = (time.perf_counter() - start)

    print(f"Avg over {runs} runs:")
    print(f"{name} model: {t*1e3:6.3f} ms")
    return t

def check_output(out_orig, out_comp):
    diff = (out_orig - out_comp).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    print(f"Output shapes: {out_orig.shape}, {out_comp.shape}")
    print(f"Max abs difference:  {max_abs_diff:.6e}")
    print(f"Mean abs difference: {mean_abs_diff:.6e}")

    # are_close = torch.allclose(out_orig, out_comp, rtol=tol_rtol, atol=tol_atol)
    # print(f"Allclose (rtol={tol_rtol}, atol={tol_atol}): {are_close}")

    print(out_orig.dtype)
    print(out_comp.dtype)


def main(quantization=""):
    log_file = "results.txt"
    with open(log_file, "a") as f:
        def log(msg):
            print(msg)
            f.write(f"{msg}\n")

        gemlite.set_packing_bitwidth(8)
        # gemlite.set_kernel_caching(True)
        # gemlite.set_acc_dtype(DType.FP16)

        MODEL_ID = "meta-llama/Llama-3.2-1B"
        log(f"\n=== Benchmark started: {datetime.datetime.now()} ===")
        log(f"Model ID: {MODEL_ID}")

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        model.to("cuda")

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

        inp = tokenizer("What is PyTorch?", return_tensors="pt").to('cuda')

        warmup(model, inp)
        t_orig = check(model, "Original", inp)
        log(f"Original model avg latency: {t_orig * 1e3:.3f} ms")

        size_before = calculate_model_size(model)
        log(f"Model size before compression: {size_before:.2f} MB")

        quantization_dataset = nncf.Dataset(dataset, partial(transform_fn, tokenizer=tokenizer))

        if quantization == "gemlite":
            model_g = nncf.compress_weights(
                model,
                dataset=quantization_dataset,
                mode=CompressWeightsMode.INT8_SYM,
                gemlite=True,
            )

            try:
                GemLiteLinearTriton.load_config(
                    f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json")
                log("GemLite kernel config loaded.")
                print(f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json")
            except:
                log("Failed to load GemLite kernel config.")

            warmup(model_g, inp)
            GemLiteLinearTriton.cache_config(
                f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
            )

            t_comp_1 = check(model_g, "NNCF Compressed with Gemlite", inp)
            log(f"NNCF Compressed model with Gemlite avg latency: {t_comp_1 * 1e3:.3f} ms")

            log(f"Speedup: {t_orig / t_comp_1:.2f}x")

            # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            # with profile(activities=activities, record_shapes=True) as prof:
            #     model_g(inp["input_ids"])
            # with profile(activities=activities, record_shapes=True) as prof:
            #     model_g(inp["input_ids"])
            # prof.export_chrome_trace("llama_3.2_1B_gemlite_trace.json")
            
            # print("Checking OV Conversion")

            # # ov1 = ov.convert_model(model_g, example_input=inp["input_ids"])

        else:
            model_c = nncf.compress_weights(
                model,
                dataset=quantization_dataset,
                mode=CompressWeightsMode.INT8_SYM,
            )

            size_after = calculate_model_size(model_c)
            log(f"Model size after compression: {size_after:.2f} MB")

            t_comp_2 = check(model_c, "NNCF Compressed", inp)
            log(f"Plain NNCF Compressed model avg latency: {t_comp_2 * 1e3:.3f} ms")

            log(f"Speedup: {t_orig / t_comp_2:.2f}x")

            # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            # with profile(activities=activities, record_shapes=True) as prof:
            #     model_c(inp["input_ids"])
            # with profile(activities=activities, record_shapes=True) as prof:
            #     model_c(inp["input_ids"])
            # prof.export_chrome_trace("llama_3.2_1B_trace.json")


        log("=== Benchmark finished ===\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", type=str, default="", help="Quantization mode, e.g. 'gemlite'")
    args = parser.parse_args()
    main(quantization=args.quantization)
