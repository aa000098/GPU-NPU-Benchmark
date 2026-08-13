"""
LLM Decode Throughput Sweep
시퀀스 길이별 decode tok/s 측정.
prefill_sweep.py와 동일한 백엔드/모델 사용, decode에 집중.

Usage:
    python profiling/llm_decode_sweep.py --backend cpu_mnn
    python profiling/llm_decode_sweep.py  # all backends
"""
import os
import sys
import csv
import json
import argparse
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "roofline_config.yaml")

# Import from prefill_sweep (same measurement functions)
sys.path.insert(0, SCRIPT_DIR)
from llm_prefill_sweep import (
    load_config, find_mnn_model_dir, find_rkllm_model,
    measure_mnn_prefill, measure_rkllm_prefill, get_memory_mb
)


def main():
    parser = argparse.ArgumentParser(description="LLM Decode Sweep")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--backend", type=str, default=None)
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
            continue

        framework = backend_cfg["framework"]
        configs = backend_cfg["configs"]

        for config in configs:
            config_id = config["id"]
            print(f"\n{'='*60}")
            print(f"Decode Sweep: {backend_name} / {config_id} ({config['description']})")
            print(f"{'='*60}")

            for seq_len in seq_lengths:
                print(f"  context_len={seq_len:5d} : ", end="", flush=True)

                if framework == "mnn":
                    model_dir = find_mnn_model_dir(config_id)
                    if model_dir is None:
                        print("MODEL NOT FOUND")
                        continue
                    metrics = measure_mnn_prefill(
                        model_dir, backend_cfg["backend_type"],
                        backend_cfg.get("threads", 4), seq_len, gen_length, measure_runs
                    )
                elif framework == "rknn-llm":
                    model_path = find_rkllm_model(config_id)
                    if model_path is None:
                        print("MODEL NOT FOUND")
                        continue
                    metrics = measure_rkllm_prefill(model_path, seq_len, gen_length, measure_runs)
                else:
                    print("UNSUPPORTED")
                    continue

                if metrics:
                    row = {
                        "backend": backend_name,
                        "config_id": config_id,
                        "context_len": seq_len,
                        "decode_tok_s": round(metrics.get("decode_tok_s", 0), 2),
                        "memory_mb": round(metrics.get("memory_mb", 0), 1),
                    }
                    results.append(row)
                    print(f"decode={row['decode_tok_s']:.1f} tok/s, mem={row['memory_mb']:.0f} MB")
                else:
                    print("FAILED")

    # Save
    csv_path = os.path.join(output_dir, "decode_sweep.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} results to {csv_path}")

    json_path = os.path.join(output_dir, "decode_sweep.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
