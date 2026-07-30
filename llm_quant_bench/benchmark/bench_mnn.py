"""
MNN LLM Benchmark Script
RK3588에서 MNN 모델의 latency, throughput, memory 측정.
llm_bench 바이너리를 사용하거나 Python MNN.llm API를 사용.

Usage:
    python bench_mnn.py --model_dir ./mnn_models/M1_q4_ch_direct --backend cpu
    python bench_mnn.py --all_models_dir ./mnn_models --output_dir ./results/raw
"""
import os
import sys
import json
import time
import argparse
import subprocess
import statistics
import re

LLM_BENCH_PATH = "/home/hyunho.son/install_files/MNN/build/llm_bench"
LLM_DEMO_PATH = "/home/hyunho.son/install_files/MNN/build/llm_demo"


def get_memory_mb():
    """Get current process RSS in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0


def get_model_size_mb(model_dir):
    """Calculate total model size in MB."""
    total = 0
    for f in os.listdir(model_dir):
        total += os.path.getsize(os.path.join(model_dir, f))
    return total / (1024 * 1024)


def find_config_json(model_dir):
    """Find the config.json or llm_config.json in model dir."""
    for name in ["config.json", "llm_config.json"]:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return None


def benchmark_with_llm_bench(model_dir, backend, threads, prompt_lengths,
                              gen_length, warmup, measure):
    """Use the compiled llm_bench tool."""
    config_path = find_config_json(model_dir)
    if config_path is None:
        print(f"[ERROR] No config.json found in {model_dir}")
        return None

    backend_map = {"cpu": "cpu", "opencl": "opencl", "gpu": "opencl"}
    backend_arg = backend_map.get(backend, backend)

    prompt_str = ",".join(str(p) for p in prompt_lengths)

    cmd = [
        LLM_BENCH_PATH,
        "-m", config_path,
        "-a", backend_arg,
        "-t", str(threads),
        "-p", prompt_str,
        "-n", str(gen_length),
        "-rep", str(measure),
    ]

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"[ERROR] llm_bench failed: {result.stderr[-300:]}")
        return None

    return parse_llm_bench_output(result.stdout, model_dir, backend)


def benchmark_with_python_api(model_dir, backend, threads, prompt_lengths,
                               gen_length, warmup, measure):
    """Use MNN Python API for benchmarking (fallback if llm_bench unavailable)."""
    try:
        import MNN.llm as mnnllm
    except ImportError:
        print("[ERROR] MNN Python module not available")
        return None

    config_path = find_config_json(model_dir)
    if config_path is None:
        print(f"[ERROR] No config.json found in {model_dir}")
        return None

    # Create model with backend config
    model = mnnllm.create(config_path)
    backend_config = {"backend_type": backend, "thread_num": threads}
    model.set_config(backend_config)
    model.load()

    results = {
        "model_dir": model_dir,
        "framework": "mnn",
        "backend": backend,
        "threads": threads,
        "model_size_mb": round(get_model_size_mb(model_dir), 1),
        "benchmarks": [],
    }

    for pl in prompt_lengths:
        # Generate dummy prompt
        prompt = "Hello " * (pl // 1)  # Rough approximation
        print(f"\n  Prompt length ~{pl} tokens (backend={backend}):")

        # Warmup
        print(f"    Warmup ({warmup} runs)...", end="", flush=True)
        for _ in range(warmup):
            model.generate_init()
            model.response(prompt, gen_length)
        print(" done")

        # Measure
        metrics_list = []
        for i in range(measure):
            model.generate_init()

            start = time.perf_counter()
            output = model.response(prompt, gen_length)
            end = time.perf_counter()

            total_time = end - start
            # Approximate token count from output
            token_count = len(output.split()) if output else 0
            tps = token_count / total_time if total_time > 0 else 0

            metrics_list.append({
                "total_time_s": round(total_time, 4),
                "tokens_per_sec": round(tps, 2),
                "total_tokens": token_count,
            })
            print(f"    Run {i+1}/{measure}: {tps:.2f} tok/s, {total_time:.2f}s")

        tps_list = [m["tokens_per_sec"] for m in metrics_list]
        agg = {
            "prompt_length": pl,
            "generate_length": gen_length,
            "tokens_per_sec_mean": round(statistics.mean(tps_list), 2),
            "tokens_per_sec_std": round(statistics.stdev(tps_list), 2) if len(tps_list) > 1 else 0,
            "peak_memory_mb": round(get_memory_mb(), 1),
            "raw_runs": metrics_list,
        }
        results["benchmarks"].append(agg)

    return results


def parse_llm_bench_output(stdout, model_dir, backend):
    """Parse llm_bench text output into structured results."""
    results = {
        "model_dir": model_dir,
        "framework": "mnn",
        "backend": backend,
        "model_size_mb": round(get_model_size_mb(model_dir), 1),
        "benchmarks": [],
        "raw_output": stdout,
    }

    # llm_bench output format varies; extract key metrics
    # Look for patterns like "prefill: X ms", "decode: Y token/s"
    lines = stdout.strip().split("\n")
    current_benchmark = {}

    for line in lines:
        line = line.strip()
        # Try to extract TTFT / prefill time
        m = re.search(r'prefill\s*[:=]\s*([\d.]+)\s*ms', line, re.IGNORECASE)
        if m:
            current_benchmark["ttft_ms"] = float(m.group(1))

        # Try to extract decode speed
        m = re.search(r'decode\s*[:=]\s*([\d.]+)\s*token', line, re.IGNORECASE)
        if m:
            current_benchmark["tokens_per_sec"] = float(m.group(1))

        # Try to extract memory
        m = re.search(r'memory\s*[:=]\s*([\d.]+)\s*MB', line, re.IGNORECASE)
        if m:
            current_benchmark["peak_memory_mb"] = float(m.group(1))

    if current_benchmark:
        results["benchmarks"].append(current_benchmark)

    return results


def benchmark_model_dir(model_dir, backends, threads, prompt_lengths,
                        gen_length, warmup, measure, output_dir):
    """Benchmark a single model on all specified backends."""
    dir_name = os.path.basename(model_dir.rstrip("/"))
    all_results = []

    for backend in backends:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {dir_name} on {backend}")
        print(f"{'='*60}")

        # Try llm_bench first, fallback to Python API
        if os.path.exists(LLM_BENCH_PATH):
            result = benchmark_with_llm_bench(
                model_dir, backend, threads, prompt_lengths,
                gen_length, warmup, measure
            )
        else:
            result = benchmark_with_python_api(
                model_dir, backend, threads, prompt_lengths,
                gen_length, warmup, measure
            )

        if result:
            all_results.append(result)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"mnn_{dir_name}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="MNN LLM Benchmark")
    parser.add_argument("--model_dir", type=str, help="Single MNN model directory")
    parser.add_argument("--all_models_dir", type=str,
                        help="Directory containing multiple MNN model dirs")
    parser.add_argument("--output_dir", type=str, default="./results/raw")
    parser.add_argument("--backends", nargs="+", default=["cpu"],
                        help="Backends to test (cpu, opencl)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prompt_lengths", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--measure", type=int, default=10)
    args = parser.parse_args()

    if args.model_dir:
        benchmark_model_dir(
            args.model_dir, args.backends, args.threads,
            args.prompt_lengths, args.gen_length,
            args.warmup, args.measure, args.output_dir
        )
    elif args.all_models_dir:
        model_dirs = sorted([
            os.path.join(args.all_models_dir, d)
            for d in os.listdir(args.all_models_dir)
            if os.path.isdir(os.path.join(args.all_models_dir, d))
            and not d.startswith(".")
        ])
        print(f"Found {len(model_dirs)} model directories")
        for md in model_dirs:
            benchmark_model_dir(
                md, args.backends, args.threads,
                args.prompt_lengths, args.gen_length,
                args.warmup, args.measure, args.output_dir
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
