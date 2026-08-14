"""
KV Cache Memory Tracking
시퀀스 길이 증가에 따른 메모리 사용량 추적.
이론적 KV 캐시 크기와 실측 메모리 비교.

Usage:
    python profiling/kv_cache_memory.py
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

sys.path.insert(0, SCRIPT_DIR)
from llm_prefill_sweep import (
    load_config, find_mnn_model_dir, find_rkllm_model,
    measure_mnn_prefill, measure_rkllm_prefill, get_memory_mb
)


def compute_theoretical_kv_cache_mb(seq_len, arch, bytes_per_kv):
    """
    Theoretical KV cache size in MB.
    KV cache = 2 (K+V) × num_layers × seq_len × num_kv_heads × head_dim × bytes_per_element
    """
    kv_bytes = (2 * arch["num_layers"] * seq_len *
                arch["num_kv_heads"] * arch["head_dim"] * bytes_per_kv)
    return kv_bytes / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="KV Cache Memory Tracking")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm = cfg["llm_profiling"]
    arch = cfg["model_arch"]
    quant_configs = cfg["quant_configs"]
    seq_lengths = llm["seq_lengths"]
    gen_length = 16  # Short generation to focus on memory, not throughput

    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(output_dir, exist_ok=True)

    results = []

    print(f"{'='*70}")
    print(f"KV Cache Memory Tracking — {arch['name']}")
    print(f"{'='*70}")

    # Compute theoretical KV cache sizes
    print(f"\nTheoretical KV Cache Sizes:")
    print(f"  {'Quant':8s} {'bytes/kv':>8s}", end="")
    for sl in seq_lengths:
        print(f" {sl:>8d}", end="")
    print(" (MB)")

    for qname, qcfg in quant_configs.items():
        print(f"  {qname:8s} {qcfg['bytes_per_kv']:>8.1f}", end="")
        for sl in seq_lengths:
            kv_mb = compute_theoretical_kv_cache_mb(sl, arch, qcfg["bytes_per_kv"])
            print(f" {kv_mb:>8.1f}", end="")
        print()

    # Measure actual memory across backends
    print(f"\n{'='*70}")
    print("Measured Memory Usage:")
    print(f"{'='*70}")

    for backend_name, backend_cfg in llm["backends"].items():
        framework = backend_cfg["framework"]
        configs = backend_cfg["configs"]

        for config in configs:
            config_id = config["id"]
            bytes_per_kv = config.get("bytes_per_kv", 2)  # default FP16
            print(f"\n  {backend_name} / {config_id}:")

            for seq_len in seq_lengths:
                print(f"    seq_len={seq_len:5d} : ", end="", flush=True)

                if framework == "mnn":
                    model_dir = find_mnn_model_dir(config_id)
                    if model_dir is None:
                        print("MODEL NOT FOUND")
                        continue
                    metrics = measure_mnn_prefill(
                        model_dir, backend_cfg["backend_type"],
                        backend_cfg.get("threads", 4), seq_len, gen_length, 3
                    )
                elif framework == "rknn-llm":
                    model_path = find_rkllm_model(config_id)
                    if model_path is None:
                        print("MODEL NOT FOUND")
                        continue
                    metrics = measure_rkllm_prefill(model_path, seq_len, gen_length, 3)
                else:
                    continue

                if metrics:
                    measured_mb = metrics.get("memory_mb", 0)
                    theoretical_kv_mb = compute_theoretical_kv_cache_mb(seq_len, arch, bytes_per_kv)

                    row = {
                        "backend": backend_name,
                        "config_id": config_id,
                        "seq_len": seq_len,
                        "measured_memory_mb": round(measured_mb, 1),
                        "theoretical_kv_mb": round(theoretical_kv_mb, 2),
                        "bytes_per_kv": bytes_per_kv,
                    }
                    results.append(row)
                    print(f"measured={measured_mb:.0f} MB, "
                          f"theoretical_kv={theoretical_kv_mb:.1f} MB")
                else:
                    print("FAILED")

    # Save
    csv_path = os.path.join(output_dir, "kv_cache_memory.csv")
    if results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} results to {csv_path}")

    json_path = os.path.join(output_dir, "kv_cache_memory.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
