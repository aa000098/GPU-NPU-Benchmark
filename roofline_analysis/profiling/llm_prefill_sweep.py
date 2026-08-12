"""
LLM Prefill Throughput Sweep
시퀀스 길이별 prefill tok/s 측정 (CPU/GPU/NPU 3 백엔드).

MNN: llm_bench 바이너리로 측정
RKNN: ctypes librkllmrt.so로 TTFT 측정 → prefill_tok_s = seq_len / TTFT_s

Usage:
    python profiling/llm_prefill_sweep.py --backend cpu_mnn
    python profiling/llm_prefill_sweep.py --backend npu_rknn
    python profiling/llm_prefill_sweep.py  # all backends
"""
import os
import sys
import csv
import time
import json
import re
import argparse
import subprocess
import statistics
import yaml

os.environ['LOGLEVEL'] = 'WARNING'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "roofline_config.yaml")

LLM_BENCH_PATH = "/home/hyunho.son/install_files/MNN/build/llm_bench"
RKLLM_LIB_PATH = "/home/hyunho.son/install_files/rknn-llm/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so"


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def get_memory_mb():
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0


def find_mnn_model_dir(config_id):
    """Find MNN model directory by config ID (M1, M4, etc.)."""
    mnn_models_dir = os.path.join(os.path.dirname(PROJECT_DIR),
                                   "llm_quant_bench", "mnn_models")
    if not os.path.isdir(mnn_models_dir):
        return None
    for d in os.listdir(mnn_models_dir):
        if d.startswith(config_id + "_"):
            return os.path.join(mnn_models_dir, d)
    return None


def find_rkllm_model(config_id):
    """Find .rkllm model file by config ID (R1, R5, etc.)."""
    rkllm_models_dir = os.path.join(os.path.dirname(PROJECT_DIR),
                                     "llm_quant_bench", "rkllm_models")
    if not os.path.isdir(rkllm_models_dir):
        return None

    # Map config IDs to filename patterns
    config_patterns = {
        "R1": "W8A8_RK3588",
        "R2": "W8A8_G128_RK3588",
        "R3": "W8A8_G256_RK3588",
        "R4": "W8A8_G512_RK3588",
        "R5": "W4A16_RK3588",
        "R6": "W4A16_G32_RK3588",
        "R7": "W4A16_G64_RK3588",
        "R8": "W4A16_G128_RK3588",
    }

    pattern = config_patterns.get(config_id)
    if pattern:
        for f in os.listdir(rkllm_models_dir):
            if pattern in f and f.endswith(".rkllm"):
                return os.path.join(rkllm_models_dir, f)

    # Fallback: try prefix match
    for f in os.listdir(rkllm_models_dir):
        if f.startswith(config_id + "_") and f.endswith(".rkllm"):
            return os.path.join(rkllm_models_dir, f)
    return None


def measure_mnn_prefill(model_dir, backend_type, threads, seq_len, gen_length, measure_runs):
    """Measure prefill with MNN llm_bench."""
    config_path = None
    for name in ["config.json", "llm_config.json"]:
        p = os.path.join(model_dir, name)
        if os.path.exists(p):
            config_path = p
            break

    if config_path is None:
        return None

    backend_map = {"cpu": "cpu", "opencl": "opencl"}
    backend_arg = backend_map.get(backend_type, backend_type)

    cmd = [
        LLM_BENCH_PATH,
        "-m", config_path,
        "-a", backend_arg,
        "-t", str(threads),
        "-p", str(seq_len),
        "-n", str(gen_length),
        "-rep", str(measure_runs),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0:
        return None

    # Parse llm_bench table output
    # Format: | model | size | backend | threads | precision | quantAttn | test | t/s |
    # test: pp16 = prefill 16 tokens, tg64 = token gen 64 tokens
    metrics = {}
    for line in result.stdout.strip().split("\n"):
        line = line.strip()

        # Table format: | ... | ppN | XX.XX ± Y.YY |
        m = re.search(r'\|\s*pp(\d+)\s*\|\s*([\d.]+)\s*±', line)
        if m:
            pp_len = int(m.group(1))
            tok_s = float(m.group(2))
            metrics["prefill_tok_s"] = tok_s
            metrics["prefill_ms"] = pp_len / tok_s * 1000 if tok_s > 0 else 0

        # Table format: | ... | tgN | XX.XX ± Y.YY |
        m = re.search(r'\|\s*tg(\d+)\s*\|\s*([\d.]+)\s*±', line)
        if m:
            metrics["decode_tok_s"] = float(m.group(2))

        # Fallback: old format (prefill: X ms, decode: Y token/s)
        m = re.search(r'prefill\s*[:=]\s*([\d.]+)\s*ms', line, re.IGNORECASE)
        if m:
            metrics["prefill_ms"] = float(m.group(1))
            metrics["prefill_tok_s"] = seq_len / (float(m.group(1)) / 1000) if float(m.group(1)) > 0 else 0
        m = re.search(r'decode\s*[:=]\s*([\d.]+)\s*token', line, re.IGNORECASE)
        if m:
            metrics["decode_tok_s"] = float(m.group(1))
        m = re.search(r'memory\s*[:=]\s*([\d.]+)\s*MB', line, re.IGNORECASE)
        if m:
            metrics["memory_mb"] = float(m.group(1))

    if "memory_mb" not in metrics:
        metrics["memory_mb"] = 0

    return metrics if metrics else None


def measure_rkllm_prefill(model_path, seq_len, gen_length, measure_runs):
    """Measure prefill (TTFT) with RKNN-LLM ctypes."""
    import ctypes

    # Inline RKLLM bindings (copied from bench_rkllm.py)
    rkllm_lib = ctypes.CDLL(RKLLM_LIB_PATH)

    RKLLM_Handle_t = ctypes.c_void_p
    LLMCallState_NORMAL = 0
    LLMCallState_FINISH = 2
    LLMCallState_ERROR = 3

    class RKLLMExtendParam(ctypes.Structure):
        _fields_ = [("base_domain_id", ctypes.c_int32), ("embed_flash", ctypes.c_int8),
                     ("enabled_cpus_num", ctypes.c_int8), ("enabled_cpus_mask", ctypes.c_uint32),
                     ("reserved", ctypes.c_uint8 * 106)]

    class RKLLMParam(ctypes.Structure):
        _fields_ = [("model_path", ctypes.c_char_p), ("max_context_len", ctypes.c_int32),
                     ("max_new_tokens", ctypes.c_int32), ("top_k", ctypes.c_int32),
                     ("n_keep", ctypes.c_int32), ("top_p", ctypes.c_float),
                     ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float),
                     ("frequency_penalty", ctypes.c_float), ("presence_penalty", ctypes.c_float),
                     ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float),
                     ("mirostat_eta", ctypes.c_float), ("skip_special_token", ctypes.c_bool),
                     ("is_async", ctypes.c_bool), ("img_start", ctypes.c_char_p),
                     ("img_end", ctypes.c_char_p), ("img_content", ctypes.c_char_p),
                     ("extend_param", RKLLMExtendParam)]

    class RKLLMEmbedInput(ctypes.Structure):
        _fields_ = [("embed", ctypes.POINTER(ctypes.c_float)), ("n_tokens", ctypes.c_size_t)]

    class RKLLMTokenInput(ctypes.Structure):
        _fields_ = [("input_ids", ctypes.POINTER(ctypes.c_int32)), ("n_tokens", ctypes.c_size_t)]

    class RKLLMMultiModelInput(ctypes.Structure):
        _fields_ = [("prompt", ctypes.c_char_p), ("image_embed", ctypes.POINTER(ctypes.c_float)),
                     ("n_image_tokens", ctypes.c_size_t), ("n_image", ctypes.c_size_t),
                     ("image_width", ctypes.c_size_t), ("image_height", ctypes.c_size_t)]

    class RKLLMInputUnion(ctypes.Union):
        _fields_ = [("prompt_input", ctypes.c_char_p), ("embed_input", RKLLMEmbedInput),
                     ("token_input", RKLLMTokenInput), ("multimodal_input", RKLLMMultiModelInput)]

    class RKLLMInput(ctypes.Structure):
        _fields_ = [("input_mode", ctypes.c_int), ("input_data", RKLLMInputUnion)]

    class RKLLMLoraParam(ctypes.Structure):
        _fields_ = [("lora_adapter_name", ctypes.c_char_p)]

    class RKLLMPromptCacheParam(ctypes.Structure):
        _fields_ = [("save_prompt_cache", ctypes.c_int), ("prompt_cache_path", ctypes.c_char_p)]

    class RKLLMInferParam(ctypes.Structure):
        _fields_ = [("mode", ctypes.c_int), ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
                     ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
                     ("keep_history", ctypes.c_int)]

    class RKLLMResultLastHiddenLayer(ctypes.Structure):
        _fields_ = [("hidden_states", ctypes.POINTER(ctypes.c_float)),
                     ("embd_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]

    class RKLLMResultLogits(ctypes.Structure):
        _fields_ = [("logits", ctypes.POINTER(ctypes.c_float)),
                     ("vocab_size", ctypes.c_int), ("num_tokens", ctypes.c_int)]

    class RKLLMResult(ctypes.Structure):
        _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int),
                     ("last_hidden_layer", RKLLMResultLastHiddenLayer),
                     ("logits", RKLLMResultLogits)]

    # Callback state
    state = {"token_count": 0, "first_token_time": None, "finished": False, "start_time": None}

    def callback_impl(result, userdata, call_state):
        if call_state == LLMCallState_NORMAL:
            state["token_count"] += 1
            if state["first_token_time"] is None:
                state["first_token_time"] = time.perf_counter()
        elif call_state in (LLMCallState_FINISH, LLMCallState_ERROR):
            state["finished"] = True

    callback_type = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
    callback = callback_type(callback_impl)

    # Init model
    param = RKLLMParam()
    param.model_path = model_path.encode("utf-8")
    param.max_context_len = 4096
    param.max_new_tokens = gen_length
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
    rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
    rkllm_init.restype = ctypes.c_int
    ret = rkllm_init(ctypes.byref(handle), ctypes.byref(param), callback)
    if ret != 0:
        print(f"  [ERROR] rkllm_init failed (ret={ret})")
        return None

    # Generate prompt
    base = "The quick brown fox jumps over the lazy dog. "
    prompt = base * max(1, (seq_len * 4) // len(base))

    rkllm_run = rkllm_lib.rkllm_run
    rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput),
                           ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
    rkllm_run.restype = ctypes.c_int

    def do_inference():
        state["token_count"] = 0
        state["first_token_time"] = None
        state["finished"] = False

        rkllm_input = RKLLMInput()
        rkllm_input.input_mode = 0  # RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = prompt.encode("utf-8")

        infer_param = RKLLMInferParam()
        ctypes.memset(ctypes.byref(infer_param), 0, ctypes.sizeof(RKLLMInferParam))
        infer_param.mode = 0  # RKLLM_INFER_GENERATE
        infer_param.keep_history = 0

        start = time.perf_counter()
        state["start_time"] = start
        rkllm_run(handle, ctypes.byref(rkllm_input), ctypes.byref(infer_param), None)
        end = time.perf_counter()

        ttft_ms = (state["first_token_time"] - start) * 1000 if state["first_token_time"] else None
        total_time = end - start
        decode_tok_s = state["token_count"] / total_time if total_time > 0 else 0

        return {
            "ttft_ms": ttft_ms,
            "prefill_tok_s": seq_len / (ttft_ms / 1000) if ttft_ms and ttft_ms > 0 else 0,
            "decode_tok_s": decode_tok_s,
            "total_tokens": state["token_count"],
        }

    # Warmup
    for _ in range(2):
        do_inference()

    # Measure
    ttfts = []
    prefill_toks = []
    decode_toks = []
    for _ in range(measure_runs):
        m = do_inference()
        if m["ttft_ms"]:
            ttfts.append(m["ttft_ms"])
            prefill_toks.append(m["prefill_tok_s"])
        decode_toks.append(m["decode_tok_s"])

    # Destroy
    rkllm_destroy = rkllm_lib.rkllm_destroy
    rkllm_destroy.argtypes = [RKLLM_Handle_t]
    rkllm_destroy.restype = ctypes.c_int
    rkllm_destroy(handle)

    return {
        "prefill_ms": statistics.median(ttfts) if ttfts else None,
        "prefill_tok_s": statistics.median(prefill_toks) if prefill_toks else 0,
        "decode_tok_s": statistics.median(decode_toks) if decode_toks else 0,
        "memory_mb": get_memory_mb(),
    }


def main():
    parser = argparse.ArgumentParser(description="LLM Prefill Sweep")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--backend", type=str, default=None,
                        help="Specific backend (cpu_mnn, gpu_mnn, npu_rknn). Default: all")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm = cfg["llm_profiling"]
    seq_lengths = llm["seq_lengths"]
    gen_length = llm["generate_length"]
    measure_runs = llm["measure_runs"]

    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(output_dir, exist_ok=True)

    backends_to_run = [args.backend] if args.backend else list(llm["backends"].keys())
    results = []

    for backend_name in backends_to_run:
        backend_cfg = llm["backends"].get(backend_name)
        if backend_cfg is None:
            print(f"[WARN] Unknown backend: {backend_name}")
            continue

        framework = backend_cfg["framework"]
        configs = backend_cfg["configs"]

        for config in configs:
            config_id = config["id"]
            print(f"\n{'='*60}")
            print(f"Prefill Sweep: {backend_name} / {config_id} ({config['description']})")
            print(f"{'='*60}")

            for seq_len in seq_lengths:
                print(f"  seq_len={seq_len:5d} : ", end="", flush=True)

                if framework == "mnn":
                    model_dir = find_mnn_model_dir(config_id)
                    if model_dir is None:
                        print(f"MODEL NOT FOUND ({config_id})")
                        continue
                    metrics = measure_mnn_prefill(
                        model_dir, backend_cfg["backend_type"],
                        backend_cfg.get("threads", 4), seq_len, gen_length, measure_runs
                    )
                elif framework == "rknn-llm":
                    model_path = find_rkllm_model(config_id)
                    if model_path is None:
                        print(f"MODEL NOT FOUND ({config_id})")
                        continue
                    metrics = measure_rkllm_prefill(model_path, seq_len, gen_length, measure_runs)
                else:
                    print(f"UNSUPPORTED FRAMEWORK ({framework})")
                    continue

                if metrics:
                    row = {
                        "backend": backend_name,
                        "config_id": config_id,
                        "seq_len": seq_len,
                        "prefill_ms": metrics.get("prefill_ms"),
                        "prefill_tok_s": round(metrics.get("prefill_tok_s", 0), 2),
                        "decode_tok_s": round(metrics.get("decode_tok_s", 0), 2),
                        "memory_mb": round(metrics.get("memory_mb", 0), 1),
                    }
                    results.append(row)
                    print(f"prefill={row['prefill_tok_s']:.1f} tok/s, "
                          f"decode={row['decode_tok_s']:.1f} tok/s, "
                          f"mem={row['memory_mb']:.0f} MB")
                else:
                    print("FAILED")

    # Save CSV
    csv_path = os.path.join(output_dir, "prefill_sweep.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} results to {csv_path}")

    # Also save JSON
    json_path = os.path.join(output_dir, "prefill_sweep.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
