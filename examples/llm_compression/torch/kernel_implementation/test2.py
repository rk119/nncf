import os
import pwd
import time
import argparse
import torch
from nncf.parameters import CompressWeightsMode
import gemlite
from torch.profiler import profile as prof, record_function, ProfilerActivity
import numpy as np
import random
import nncf

# SEED = 42
# random.seed(SEED); np.random.seed(SEED)
# torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark     = False
# torch.use_deterministic_algorithms(True)

class SimpleLinearModel(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, dtype=torch.float16)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(output_dim, output_dim, dtype=torch.float16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = self.linear(x)
        l = self.relu(l)
        l = self.linear2(l)
        return l

def main(compression="", mode="", profile=False, cuda_graph=False):
    
    print(torch.__version__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size, input_dim, output_dim = 32, 256, 512
    model = SimpleLinearModel(input_dim, output_dim).to(device)

    model.eval()
    x = torch.randn(batch_size, input_dim, device=device, dtype=torch.float16)

    if cuda_graph:
        static_input  = torch.randn(batch_size, input_dim,  device=device, dtype=torch.float16)
        static_output = torch.empty(batch_size, output_dim, device=device, dtype=torch.float16)

    for _ in range(10):
        ch1 = model(x)

    if profile:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        with prof(activities=activities) as p:
            model(x) 
        
        p.export_chrome_trace("before_compress_torch.json")   

    if mode == "int8_sym":
        mode = CompressWeightsMode.INT8_SYM

    elif mode == "int8_asym":
        mode = CompressWeightsMode.INT8_ASYM

    elif mode == "int4_sym":
        mode = CompressWeightsMode.INT4_SYM

    elif mode == "int4_asym":
        mode = CompressWeightsMode.INT4_ASYM

    if compression == "gemlite":
        gemlite.set_autotune("max")
        config_file = f"/tmp/{pwd.getpwuid(os.getuid()).pw_gecos}_gemlite.json"
        gemlite.load_config(config_file)

        model = nncf.compress_weights(model, dataset=nncf.Dataset([x]), mode=mode, gemlite=True)
        
    if compression == "tinygemm":
        model = nncf.compress_weights(model, dataset=nncf.Dataset([x]), mode=mode, awq=True)

    elif compression == "nncf":
        model = nncf.compress_weights(model, dataset=nncf.Dataset([x]), mode=mode)
        
    with torch.inference_mode():
        for _ in range(50):
            t = time.perf_counter()
            _ = model(x)
            e = time.perf_counter()
            print(f"Warmup forward: {(e - t)*1e3:.3f} ms")
        torch.cuda.synchronize()

    if compression == "gemlite":
        gemlite.cache_config(config_file)     
    
    if cuda_graph:
        capture_stream = torch.cuda.Stream()
        g = torch.cuda.CUDAGraph()

        torch.cuda.synchronize()

        with torch.cuda.stream(capture_stream):
            g.capture_begin()
            tmp = model(static_input)      
            static_output.copy_(tmp)       
            g.capture_end()

        torch.cuda.synchronize()

        def infer_with_graph(x: torch.Tensor):
            static_input.copy_(x)           
            g.replay()                       
            return static_output            

        t0 = time.perf_counter(); out = infer_with_graph(x); torch.cuda.synchronize()
        print(f"First replay: {(time.perf_counter() - t0)*1e3:.3f} ms")

    times = []
    for _ in range(100):
        t = time.perf_counter()
        ch2 = model(x) if not cuda_graph else infer_with_graph(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    print(f"Avg inference: {sum(times)/len(times)*1e3:.3f} ms")

    if profile:
        with prof(activities=activities) as p:
            if cuda_graph:
                with record_function("cuda_graph_replay"):
                    _ = infer_with_graph(x)
            else:
                _ = model(x)

        p.export_chrome_trace(f"profile_{compression}_{mode}.json")
    
    # print(f"Output 1: {ch1}")
    # print(f"Output 2: {ch2}")

    diff = (ch1 - ch2).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    print(f"Output shapes: {ch1.shape}, {ch2.shape}")
    print(f"Max abs difference:  {max_abs_diff:.6e}")
    print(f"Mean abs difference: {mean_abs_diff:.6e}")
    
    # import openvino as ov
    
    # ov_model = ov.convert_model(model, example_input=x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compression", type=str, default="", help="Compression, e.g. 'gemlite', 'nncf'")
    parser.add_argument("--mode", type=str, default="", help="Quantization mode, e.g. 'int8_sym'")
    parser.add_argument("--profile", action="store_true", help="Whether to profile model and cuda graph")
    parser.add_argument("--cuda_graph", action="store_true", help="Whether to use cuda graph")
    args = parser.parse_args()
    main(compression=args.compression, mode=args.mode, profile=args.profile, cuda_graph=args.cuda_graph)
