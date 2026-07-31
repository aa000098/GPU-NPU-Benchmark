"""
RKNN-LLM Benchmark Script
RK3588에서 .rkllm 모델의 latency, throughput, memory 측정.
ctypes로 librkllmrt.so를 직접 호출.

Usage:
    python bench_rkllm.py --model_path ./model.rkllm --output_dir ./results/raw

    # All models in a directory:
    python bench_rkllm.py --model_dir ./rkllm_models --output_dir ./results/raw
"""
import ctypes
import sys
import os
import time
import json
import argparse
import threading
import subprocess
import re

# ── RKLLM C API bindings ──────────────────────────────────────────────
RKLLM_LIB_PATH = "/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so"

rkllm_lib = ctypes.CDLL(RKLLM_LIB_PATH)

RKLLM_Handle_t = ctypes.c_void_p

LLMCallState_NORMAL = 0
LLMCallState_WAITING = 1
LLMCallState_FINISH = 2
LLMCallState_ERROR = 3

RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0


class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("reserved", ctypes.c_uint8 * 106),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t),
    ]


class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t),
    ]


class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("input_mode", ctypes.c_int),
        ("input_data", RKLLMInputUnion),
    ]


class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [("lora_adapter_name", ctypes.c_char_p)]


class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int),
    ]


class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
    ]


# ── Benchmark Logic ──────────────────────────────────────────────────

# Global state for callback
_token_count = 0
_first_token_time = None
_generation_start_time = None
_finished = False


def _callback_impl(result, userdata, state):
    global _token_count, _first_token_time, _finished
    if state == LLMCallState_NORMAL:
        _token_count += 1
        if _first_token_time is None:
            _first_token_time = time.perf_counter()
    elif state == LLMCallState_FINISH:
        _finished = True
    elif state == LLMCallState_ERROR:
        _finished = True
        print("[ERROR] RKLLM inference error")


_callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
_callback = _callback_type(_callback_impl)


def get_memory_mb():
    """Get current process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB to MB
    except Exception:
        return 0


def generate_prompt(num_tokens_approx):
    """Generate a dummy prompt of approximately num_tokens_approx tokens."""
    # Rough approximation: 1 token ≈ 4 chars
    base = "The quick brown fox jumps over the lazy dog. "
    repeats = max(1, (num_tokens_approx * 4) // len(base))
    return base * repeats


def init_model(model_path, max_context_len=4096, max_new_tokens=256):
    """Initialize RKLLM model and return handle."""
    param = RKLLMParam()
    param.model_path = model_path.encode("utf-8")
    param.max_context_len = max_context_len
    param.max_new_tokens = max_new_tokens
    param.skip_special_token = True
    param.n_keep = -1
    param.top_k = 1
    param.top_p = 0.9
    param.temperature = 0.8
    param.repeat_penalty = 1.1
    param.frequency_penalty = 0.0
    param.presence_penalty = 0.0
    param.mirostat = 0
    param.mirostat_tau = 5.0
    param.mirostat_eta = 0.1
    param.is_async = False
    param.img_start = b""
    param.img_end = b""
    param.img_content = b""
    param.extend_param.base_domain_id = 0
    param.extend_param.enabled_cpus_num = 4
    param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)

    handle = RKLLM_Handle_t()

    rkllm_init = rkllm_lib.rkllm_init
    rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), _callback_type]
    rkllm_init.restype = ctypes.c_int

    ret = rkllm_init(ctypes.byref(handle), ctypes.byref(param), _callback)
    if ret != 0:
        raise RuntimeError(f"rkllm_init failed with ret={ret}")

    return handle


def run_inference(handle, prompt):
    """Run a single inference and return metrics."""
    global _token_count, _first_token_time, _generation_start_time, _finished

    _token_count = 0
    _first_token_time = None
    _finished = False

    rkllm_input = RKLLMInput()
    rkllm_input.input_mode = RKLLM_INPUT_PROMPT
    rkllm_input.input_data.prompt_input = prompt.encode("utf-8")

    infer_param = RKLLMInferParam()
    ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
    infer_param.mode = RKLLM_INFER_GENERATE
    infer_param.keep_history = 0

    rkllm_run = rkllm_lib.rkllm_run
    rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
    rkllm_run.restype = ctypes.c_int

    _generation_start_time = time.perf_counter()
    ret = rkllm_run(handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None)
    end_time = time.perf_counter()

    if ret != 0:
        print(f"[WARN] rkllm_run returned {ret}")

    total_time = end_time - _generation_start_time
    ttft = (_first_token_time - _generation_start_time) * 1000 if _first_token_time else None
    tokens_per_sec = _token_count / total_time if total_time > 0 else 0

    return {
        "ttft_ms": ttft,
        "tokens_per_sec": tokens_per_sec,
        "total_tokens": _token_count,
        "total_time_s": total_time,
    }


def destroy_model(handle):
    rkllm_destroy = rkllm_lib.rkllm_destroy
    rkllm_destroy.argtypes = [RKLLM_Handle_t]
    rkllm_destroy.restype = ctypes.c_int
    rkllm_destroy(handle)


def benchmark_model(model_path, prompt_lengths, gen_length, warmup, measure, output_dir):
    """Run full benchmark for a single model."""
    model_name = os.path.basename(model_path).replace(".rkllm", "")
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    mem_before = get_memory_mb()
    handle = init_model(model_path, max_new_tokens=gen_length)
    mem_after = get_memory_mb()
    model_mem_mb = mem_after - mem_before

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    results = {
        "model": model_name,
        "model_path": model_path,
        "model_size_mb": round(model_size_mb, 1),
        "model_memory_mb": round(model_mem_mb, 1),
        "framework": "rknn-llm",
        "backend": "npu",
        "benchmarks": [],
    }

    for pl in prompt_lengths:
        prompt = generate_prompt(pl)
        print(f"\n  Prompt length ~{pl} tokens:")

        # Warmup
        print(f"    Warmup ({warmup} runs)...", end="", flush=True)
        for _ in range(warmup):
            run_inference(handle, prompt)
        print(" done")

        # Measure
        metrics_list = []
        for i in range(measure):
            m = run_inference(handle, prompt)
            metrics_list.append(m)
            print(f"    Run {i+1}/{measure}: TTFT={m['ttft_ms']:.1f}ms, "
                  f"{m['tokens_per_sec']:.2f} tok/s, {m['total_tokens']} tokens")

        # Aggregate
        ttfts = [m["ttft_ms"] for m in metrics_list if m["ttft_ms"] is not None]
        tps_list = [m["tokens_per_sec"] for m in metrics_list]

        import statistics
        agg = {
            "prompt_length": pl,
            "generate_length": gen_length,
            "ttft_ms_mean": round(statistics.mean(ttfts), 2) if ttfts else None,
            "ttft_ms_std": round(statistics.stdev(ttfts), 2) if len(ttfts) > 1 else 0,
            "tokens_per_sec_mean": round(statistics.mean(tps_list), 2),
            "tokens_per_sec_std": round(statistics.stdev(tps_list), 2) if len(tps_list) > 1 else 0,
            "peak_memory_mb": round(get_memory_mb(), 1),
            "raw_runs": metrics_list,
        }
        results["benchmarks"].append(agg)

    destroy_model(handle)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rkllm_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="RKNN-LLM Benchmark")
    parser.add_argument("--model_path", type=str, help="Path to single .rkllm model")
    parser.add_argument("--model_dir", type=str, help="Directory containing .rkllm models")
    parser.add_argument("--output_dir", type=str, default="./results/raw")
    parser.add_argument("--prompt_lengths", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=10)
    args = parser.parse_args()

    if args.model_path:
        benchmark_model(args.model_path, args.prompt_lengths, args.gen_length,
                        args.warmup, args.measure, args.output_dir)
    elif args.model_dir:
        models = sorted([
            os.path.join(args.model_dir, f)
            for f in os.listdir(args.model_dir)
            if f.endswith(".rkllm")
        ])
        print(f"Found {len(models)} .rkllm models")
        all_results = []
        for mp in models:
            r = benchmark_model(mp, args.prompt_lengths, args.gen_length,
                                args.warmup, args.measure, args.output_dir)
            all_results.append(r)

        # Save combined summary
        summary_path = os.path.join(args.output_dir, "rkllm_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nAll results saved to {summary_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
