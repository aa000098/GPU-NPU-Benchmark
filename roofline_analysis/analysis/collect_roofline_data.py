"""
Roofline Data Collector
하드웨어 피크 + LLM 프로파일링 raw CSV를 통합 데이터셋으로 집계.

Usage:
    python analysis/collect_roofline_data.py
"""
import os
import csv
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def collect_hardware_peaks(raw_dir):
    """Collect peak compute and bandwidth per backend."""
    peaks = {}

    for backend in ["cpu", "gpu", "npu"]:
        peaks[backend] = {"peak_gflops": 0, "peak_bandwidth_gb_s": 0}

        # Peak compute
        compute_csv = os.path.join(raw_dir, f"peak_compute_{backend}.csv")
        rows = load_csv(compute_csv)
        for r in rows:
            gf = float(r.get("gflops", 0))
            if gf > peaks[backend]["peak_gflops"]:
                peaks[backend]["peak_gflops"] = gf
                peaks[backend]["peak_compute_dtype"] = r.get("dtype", "")
                peaks[backend]["peak_compute_size"] = int(r.get("size", 0))

        # Peak bandwidth
        bw_csv = os.path.join(raw_dir, f"mem_bw_{backend}.csv")
        rows = load_csv(bw_csv)
        for r in rows:
            bw = float(r.get("bandwidth_gb_s", 0))
            if bw > peaks[backend]["peak_bandwidth_gb_s"]:
                peaks[backend]["peak_bandwidth_gb_s"] = bw
                peaks[backend]["peak_bw_operation"] = r.get("operation", "")

        # Ridge point
        if peaks[backend]["peak_bandwidth_gb_s"] > 0:
            peaks[backend]["ridge_point"] = round(
                peaks[backend]["peak_gflops"] / peaks[backend]["peak_bandwidth_gb_s"], 3
            )
        else:
            peaks[backend]["ridge_point"] = 0

    return peaks


def collect_profiling_data(raw_dir):
    """Collect LLM profiling results."""
    prefill = load_csv(os.path.join(raw_dir, "prefill_sweep.csv"))
    decode = load_csv(os.path.join(raw_dir, "decode_sweep.csv"))
    kv = load_csv(os.path.join(raw_dir, "kv_cache_memory.csv"))
    return {"prefill": prefill, "decode": decode, "kv_cache": kv}


def build_unified(peaks, profiling):
    """Build unified dataset combining hardware specs and profiling."""
    unified = []

    for row in profiling.get("prefill", []):
        backend_key = row["backend"].split("_")[0]  # cpu_mnn → cpu
        hw = peaks.get(backend_key, {})

        entry = {
            "backend": row["backend"],
            "config_id": row["config_id"],
            "seq_len": int(row["seq_len"]),
            "phase": "prefill",
            "throughput_tok_s": float(row.get("prefill_tok_s", 0)),
            "latency_ms": row.get("prefill_ms"),
            "memory_mb": float(row.get("memory_mb", 0)),
            "peak_gflops": hw.get("peak_gflops", 0),
            "peak_bandwidth_gb_s": hw.get("peak_bandwidth_gb_s", 0),
            "ridge_point": hw.get("ridge_point", 0),
        }
        unified.append(entry)

    for row in profiling.get("decode", []):
        backend_key = row["backend"].split("_")[0]
        hw = peaks.get(backend_key, {})

        entry = {
            "backend": row["backend"],
            "config_id": row["config_id"],
            "seq_len": int(row.get("context_len", 0)),
            "phase": "decode",
            "throughput_tok_s": float(row.get("decode_tok_s", 0)),
            "latency_ms": None,
            "memory_mb": float(row.get("memory_mb", 0)),
            "peak_gflops": hw.get("peak_gflops", 0),
            "peak_bandwidth_gb_s": hw.get("peak_bandwidth_gb_s", 0),
            "ridge_point": hw.get("ridge_point", 0),
        }
        unified.append(entry)

    return unified


def main():
    parser = argparse.ArgumentParser(description="Collect Roofline Data")
    parser.add_argument("--raw_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "results", "raw"))
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or args.raw_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Collecting hardware peaks...")
    peaks = collect_hardware_peaks(args.raw_dir)
    for backend, hw in peaks.items():
        print(f"  {backend}: {hw['peak_gflops']:.1f} GFLOPS, "
              f"{hw['peak_bandwidth_gb_s']:.1f} GB/s, "
              f"ridge={hw['ridge_point']:.2f} FLOP/Byte")

    print("\nCollecting profiling data...")
    profiling = collect_profiling_data(args.raw_dir)
    print(f"  Prefill entries: {len(profiling['prefill'])}")
    print(f"  Decode entries: {len(profiling['decode'])}")
    print(f"  KV cache entries: {len(profiling['kv_cache'])}")

    print("\nBuilding unified dataset...")
    unified = build_unified(peaks, profiling)
    print(f"  Total entries: {len(unified)}")

    # Save unified CSV
    csv_path = os.path.join(output_dir, "roofline_unified.csv")
    if unified:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=unified[0].keys())
            writer.writeheader()
            writer.writerows(unified)

    # Save hardware peaks JSON
    peaks_path = os.path.join(output_dir, "hardware_peaks.json")
    with open(peaks_path, "w") as f:
        json.dump(peaks, f, indent=2)

    # Save unified JSON
    json_path = os.path.join(output_dir, "roofline_unified.json")
    with open(json_path, "w") as f:
        json.dump(unified, f, indent=2, default=str)

    print(f"\nSaved to {csv_path}")
    print(f"Saved to {peaks_path}")


if __name__ == "__main__":
    main()
