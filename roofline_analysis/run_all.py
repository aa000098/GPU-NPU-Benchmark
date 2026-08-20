"""
3-Backend Roofline Analysis — Full Pipeline Orchestrator

Usage:
    python run_all.py                    # Run all steps
    python run_all.py --step hardware    # Hardware characterization only
    python run_all.py --step profiling   # LLM profiling only
    python run_all.py --step analysis    # Analysis only (needs raw data)
    python run_all.py --step viz         # Visualization only (needs analysis data)
"""
import os
import sys
import time
import subprocess
import argparse
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(description, script_path, extra_args=None, timeout=None):
    """Run a subprocess step with logging."""
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, timeout=timeout)
        elapsed = time.time() - start

        if result.returncode != 0:
            print(f"[FAIL] {description} (exit={result.returncode}, {elapsed:.0f}s)")
            return False
        else:
            print(f"[OK] {description} ({elapsed:.0f}s)")
            return True
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"[TIMEOUT] {description} ({elapsed:.0f}s)")
        return False
    except Exception as e:
        print(f"[ERROR] {description}: {e}")
        return False


def run_hardware(output_dir):
    """Step 1: Hardware characterization."""
    steps = [
        ("CPU Peak Compute", os.path.join(PROJECT_DIR, "hardware", "peak_compute_cpu.py")),
        ("GPU Peak Compute", os.path.join(PROJECT_DIR, "hardware", "peak_compute_gpu.py")),
        ("NPU Peak Compute", os.path.join(PROJECT_DIR, "hardware", "peak_compute_npu.py")),
        ("CPU Memory Bandwidth", os.path.join(PROJECT_DIR, "hardware", "mem_bandwidth_cpu.py")),
        ("GPU Memory Bandwidth", os.path.join(PROJECT_DIR, "hardware", "mem_bandwidth_gpu.py")),
        ("NPU Memory Bandwidth", os.path.join(PROJECT_DIR, "hardware", "mem_bandwidth_npu.py")),
    ]
    results = []
    for desc, script in steps:
        ok = run_step(desc, script, ["--output_dir", output_dir])
        results.append((desc, ok))
    return results


def run_profiling(output_dir):
    """Step 2: LLM profiling."""
    steps = [
        ("Prefill Sweep", os.path.join(PROJECT_DIR, "profiling", "llm_prefill_sweep.py")),
        ("Decode Sweep", os.path.join(PROJECT_DIR, "profiling", "llm_decode_sweep.py")),
        ("KV Cache Memory", os.path.join(PROJECT_DIR, "profiling", "kv_cache_memory.py")),
    ]
    results = []
    for desc, script in steps:
        ok = run_step(desc, script, ["--output_dir", output_dir])
        results.append((desc, ok))
    return results


def run_analysis(raw_dir, output_dir):
    """Step 3: Analysis."""
    steps = [
        ("Collect Roofline Data", os.path.join(PROJECT_DIR, "analysis", "collect_roofline_data.py"),
         ["--raw_dir", raw_dir, "--output_dir", output_dir]),
        ("Arithmetic Intensity", os.path.join(PROJECT_DIR, "analysis", "arithmetic_intensity.py"),
         ["--output_dir", output_dir]),
        ("Crossover Analysis", os.path.join(PROJECT_DIR, "analysis", "crossover_analysis.py"),
         ["--raw_dir", raw_dir, "--output_dir", output_dir]),
        ("Optimal Strategy", os.path.join(PROJECT_DIR, "analysis", "optimal_strategy.py"),
         ["--raw_dir", raw_dir, "--output_dir", output_dir]),
    ]
    results = []
    for desc, script, args in steps:
        ok = run_step(desc, script, args)
        results.append((desc, ok))
    return results


def run_viz(raw_dir, fig_dir):
    """Step 4: Visualization."""
    steps = [
        ("Roofline Plots", os.path.join(PROJECT_DIR, "visualization", "plot_roofline.py"),
         ["--raw_dir", raw_dir, "--output_dir", fig_dir]),
        ("Crossover Plots", os.path.join(PROJECT_DIR, "visualization", "plot_crossover.py"),
         ["--raw_dir", raw_dir, "--output_dir", fig_dir]),
        ("Heatmap Plots", os.path.join(PROJECT_DIR, "visualization", "plot_heatmap.py"),
         ["--raw_dir", raw_dir, "--output_dir", fig_dir]),
        ("KV Cache Plots", os.path.join(PROJECT_DIR, "visualization", "plot_kv_cache.py"),
         ["--raw_dir", raw_dir, "--output_dir", fig_dir]),
        ("Summary Figure", os.path.join(PROJECT_DIR, "visualization", "plot_summary.py"),
         ["--raw_dir", raw_dir, "--output_dir", fig_dir]),
    ]
    results = []
    for desc, script, args in steps:
        ok = run_step(desc, script, args)
        results.append((desc, ok))
    return results


def main():
    parser = argparse.ArgumentParser(description="3-Backend Roofline Analysis Pipeline")
    parser.add_argument("--step", type=str, default=None,
                        choices=["hardware", "profiling", "analysis", "viz"],
                        help="Run specific step only")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = args.output_dir or os.path.join(PROJECT_DIR, "results")
    raw_dir = os.path.join(base_output, "raw")
    fig_dir = os.path.join(base_output, f"figures_{timestamp}")
    os.makedirs(raw_dir, exist_ok=True)

    print(f"{'#'*70}")
    print(f"  3-Backend Roofline Analysis Pipeline")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output: {base_output}")
    print(f"{'#'*70}")

    all_results = []

    steps_to_run = [args.step] if args.step else ["hardware", "profiling", "analysis", "viz"]

    for step in steps_to_run:
        if step == "hardware":
            all_results.extend(run_hardware(raw_dir))
        elif step == "profiling":
            all_results.extend(run_profiling(raw_dir))
        elif step == "analysis":
            all_results.extend(run_analysis(raw_dir, raw_dir))
        elif step == "viz":
            os.makedirs(fig_dir, exist_ok=True)
            all_results.extend(run_viz(raw_dir, fig_dir))

    # Summary
    print(f"\n{'#'*70}")
    print("Pipeline Summary:")
    for name, ok in all_results:
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {name}")

    n_ok = sum(1 for _, ok in all_results if ok)
    n_total = len(all_results)
    print(f"\n  {n_ok}/{n_total} steps completed successfully")
    print(f"  Raw data: {raw_dir}")
    if "viz" in steps_to_run:
        print(f"  Figures:  {fig_dir}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
