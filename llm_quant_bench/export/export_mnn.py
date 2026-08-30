"""
MNN Model Export Script
RK3588에서 직접 실행 가능 (CPU-only export).
HuggingFace 모델을 MNN 포맷으로 다양한 양자화 설정으로 변환.

Usage:
    # Single config:
    python export_mnn.py --model_path /path/to/gemma-model --config M1

    # All configs:
    python export_mnn.py --model_path /path/to/gemma-model --run_all

    # Custom:
    python export_mnn.py --model_path /path/to/gemma-model \
        --quant_bit 4 --quant_block 64 --awq
"""
import os
import sys
import json
import argparse
import subprocess
import time

# llmexport.py 위치
LLMEXPORT_DIR = "/home/hyunho.son/install_files/MNN/transformers/llm/export"
LLMEXPORT_SCRIPT = os.path.join(LLMEXPORT_DIR, "llmexport.py")
MNNCONVERT_PATH = "/home/hyunho.son/install_files/MNN/build/MNNConvert"

EXPERIMENT_CONFIGS = {
    "M1": {"quant_bit": 4, "quant_block": 0,  "method": "direct",  "extra_flags": []},
    "M2": {"quant_bit": 4, "quant_block": 64, "method": "direct",  "extra_flags": []},
    "M3": {"quant_bit": 4, "quant_block": 128,"method": "direct",  "extra_flags": []},
    "M4": {"quant_bit": 8, "quant_block": 0,  "method": "direct",  "extra_flags": []},
    "M5": {"quant_bit": 8, "quant_block": 64, "method": "direct",  "extra_flags": []},
    "M6": {"quant_bit": 4, "quant_block": 64, "method": "awq",     "extra_flags": ["--awq"]},
    "M7": {"quant_bit": 4, "quant_block": 64, "method": "smooth",  "extra_flags": ["--smooth"]},
    "M8": {"quant_bit": 4, "quant_block": 64, "method": "hqq",     "extra_flags": ["--hqq"]},
}


def get_output_dirname(config_id, config):
    """Generate descriptive output directory name."""
    method = config["method"]
    bit = config["quant_bit"]
    block = config["quant_block"]
    block_str = f"b{block}" if block > 0 else "ch"
    return f"{config_id}_q{bit}_{block_str}_{method}"


def export_single(model_path, config_id, config, base_output_dir, model_type=None):
    dirname = get_output_dirname(config_id, config)
    dst_path = os.path.join(base_output_dir, dirname)

    # Check if already exported
    if os.path.exists(os.path.join(dst_path, "llm.mnn.weight")):
        print(f"[SKIP] {dirname} already exists")
        return dst_path

    print(f"\n{'='*60}")
    print(f"Exporting: {config_id} ({dirname})")
    print(f"  bit={config['quant_bit']}, block={config['quant_block']}, method={config['method']}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, LLMEXPORT_SCRIPT,
        "--path", model_path,
        "--export", "mnn",
        "--quant_bit", str(config["quant_bit"]),
        "--quant_block", str(config["quant_block"]),
        "--dst_path", dst_path,
        "--mnnconvert", MNNCONVERT_PATH,
    ]

    if model_type:
        cmd.extend(["--type", model_type])

    cmd.extend(config["extra_flags"])

    print(f"  CMD: {' '.join(cmd)}")
    start_time = time.time()

    result = subprocess.run(cmd, cwd=LLMEXPORT_DIR, capture_output=True, text=True)

    elapsed = time.time() - start_time
    print(f"  Time: {elapsed:.1f}s")

    if result.returncode != 0:
        print(f"[ERROR] Export failed for {config_id}")
        print(f"  STDOUT: {result.stdout[-500:]}")
        print(f"  STDERR: {result.stderr[-500:]}")
        return None

    # Measure model size
    total_size = 0
    if os.path.exists(dst_path):
        for f in os.listdir(dst_path):
            total_size += os.path.getsize(os.path.join(dst_path, f))

    print(f"[OK] Exported: {dst_path} ({total_size / (1024*1024):.1f} MB)")
    return dst_path


def export_all(model_path, base_output_dir, model_type=None, configs=None):
    configs_to_run = configs or list(EXPERIMENT_CONFIGS.keys())
    results = {}

    for config_id in configs_to_run:
        if config_id not in EXPERIMENT_CONFIGS:
            print(f"[WARN] Unknown config: {config_id}, skipping")
            continue

        config = EXPERIMENT_CONFIGS[config_id]
        path = export_single(model_path, config_id, config, base_output_dir, model_type)

        # Measure size
        total_size = 0
        if path and os.path.exists(path):
            for f in os.listdir(path):
                total_size += os.path.getsize(os.path.join(path, f))

        results[config_id] = {
            "path": path,
            "config": config,
            "size_mb": total_size / (1024*1024) if path else None,
        }

    # Save summary
    os.makedirs(base_output_dir, exist_ok=True)
    summary_path = os.path.join(base_output_dir, "export_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Export Gemma model to MNN format")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--output_dir", type=str, default="./mnn_models",
                        help="Base output directory for MNN models")
    parser.add_argument("--model_type", type=str, default=None,
                        help="Model type override (e.g., gemma3_text, gemma2)")

    # Single export
    parser.add_argument("--config", type=str, default=None,
                        help="Single config ID (e.g., M1)")
    parser.add_argument("--quant_bit", type=int, default=None)
    parser.add_argument("--quant_block", type=int, default=None)
    parser.add_argument("--awq", action="store_true")
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--hqq", action="store_true")

    # Batch export
    parser.add_argument("--run_all", action="store_true")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific config IDs (e.g., M1 M6 M8)")

    args = parser.parse_args()

    if args.run_all:
        export_all(args.model_path, args.output_dir, args.model_type)
    elif args.configs:
        export_all(args.model_path, args.output_dir, args.model_type, args.configs)
    elif args.config:
        if args.config not in EXPERIMENT_CONFIGS:
            print(f"Unknown config: {args.config}")
            sys.exit(1)
        config = EXPERIMENT_CONFIGS[args.config]
        export_single(args.model_path, args.config, config, args.output_dir, args.model_type)
    elif args.quant_bit is not None:
        custom_config = {
            "quant_bit": args.quant_bit,
            "quant_block": args.quant_block or 64,
            "method": "awq" if args.awq else "smooth" if args.smooth else "hqq" if args.hqq else "direct",
            "extra_flags": [],
        }
        if args.awq: custom_config["extra_flags"].append("--awq")
        if args.smooth: custom_config["extra_flags"].append("--smooth")
        if args.hqq: custom_config["extra_flags"].append("--hqq")
        export_single(args.model_path, "custom", custom_config, args.output_dir, args.model_type)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
