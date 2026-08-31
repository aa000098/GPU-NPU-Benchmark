"""
RKNN-LLM Model Export Script (Colab용)
Google Colab에서 실행하여 HuggingFace 모델을 .rkllm 포맷으로 변환.
변환된 파일을 RK3588로 전송하여 추론에 사용.

Usage:
    python export_rkllm.py --model_path /path/to/gemma-model \
        --quantized_dtype W8A8 --quantized_algorithm normal \
        --output_dir ./rkllm_models

    # Group-wise quantization:
    python export_rkllm.py --model_path /path/to/gemma-model \
        --quantized_dtype W4A16 --quantized_algorithm grq \
        --group_size 128 --output_dir ./rkllm_models

    # Run all 8 configs:
    python export_rkllm.py --model_path /path/to/gemma-model --run_all
"""
import os
import sys
import argparse
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EXPERIMENT_CONFIGS = {
    "R1": {"quantized_dtype": "W8A8",  "quantized_algorithm": "normal", "group_size": None},
    "R2": {"quantized_dtype": "W8A8",  "quantized_algorithm": "normal", "group_size": 128},
    "R3": {"quantized_dtype": "W8A8",  "quantized_algorithm": "normal", "group_size": 256},
    "R4": {"quantized_dtype": "W8A8",  "quantized_algorithm": "normal", "group_size": 512},
    "R5": {"quantized_dtype": "W4A16", "quantized_algorithm": "grq",    "group_size": None},
    "R6": {"quantized_dtype": "W4A16", "quantized_algorithm": "grq",    "group_size": 32},
    "R7": {"quantized_dtype": "W4A16", "quantized_algorithm": "grq",    "group_size": 64},
    "R8": {"quantized_dtype": "W4A16", "quantized_algorithm": "grq",    "group_size": 128},
}


def build_quantized_dtype_str(dtype, group_size):
    """Build the quantized_dtype string with optional group size."""
    if group_size is not None:
        return f"{dtype}_G{group_size}"
    return dtype


def export_single(model_path, quantized_dtype, quantized_algorithm, group_size,
                   output_dir, dataset_path, device="cuda", dtype="float16",
                   max_context=4096):
    from rkllm.api import RKLLM

    model_name = os.path.basename(model_path.rstrip('/'))
    dtype_str = build_quantized_dtype_str(quantized_dtype, group_size)
    output_filename = f"{model_name}_{dtype_str}_RK3588.rkllm"
    output_path = os.path.join(output_dir, output_filename)

    if os.path.exists(output_path):
        print(f"[SKIP] {output_path} already exists")
        return output_path

    print(f"\n{'='*60}")
    print(f"Exporting: {output_filename}")
    print(f"  dtype={quantized_dtype}, algo={quantized_algorithm}, group={group_size}")
    print(f"{'='*60}")

    llm = RKLLM()

    # Load model
    ret = llm.load_huggingface(
        model=model_path,
        model_lora=None,
        device=device,
        dtype=dtype,
        custom_config=None,
        load_weight=True
    )
    if ret != 0:
        print(f"[ERROR] Failed to load model (ret={ret})")
        return None

    # Build with quantization
    full_dtype = build_quantized_dtype_str(quantized_dtype, group_size)
    ret = llm.build(
        do_quantization=True,
        optimization_level=1,
        quantized_dtype=full_dtype,
        quantized_algorithm=quantized_algorithm,
        target_platform="RK3588",
        num_npu_core=3,
        extra_qparams=None,
        dataset=dataset_path,
        hybrid_rate=0,
        max_context=max_context,
    )
    if ret != 0:
        print(f"[ERROR] Failed to build model (ret={ret})")
        return None

    # Export
    os.makedirs(output_dir, exist_ok=True)
    ret = llm.export_rkllm(output_path)
    if ret != 0:
        print(f"[ERROR] Failed to export model (ret={ret})")
        return None

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[OK] Exported: {output_path} ({file_size_mb:.1f} MB)")
    return output_path


def export_all(model_path, output_dir, dataset_path, device="cuda", dtype="float16"):
    results = {}
    for config_id, config in EXPERIMENT_CONFIGS.items():
        print(f"\n>>> Processing {config_id} ...")
        path = export_single(
            model_path=model_path,
            quantized_dtype=config["quantized_dtype"],
            quantized_algorithm=config["quantized_algorithm"],
            group_size=config["group_size"],
            output_dir=output_dir,
            dataset_path=dataset_path,
            device=device,
            dtype=dtype,
        )
        results[config_id] = {
            "path": path,
            "config": config,
            "size_mb": os.path.getsize(path) / (1024*1024) if path and os.path.exists(path) else None,
        }

    # Save summary
    summary_path = os.path.join(output_dir, "export_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Export Gemma model to RKLLM format")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model directory")
    parser.add_argument("--output_dir", type=str, default="./rkllm_models",
                        help="Output directory for .rkllm files")
    parser.add_argument("--dataset", type=str, default="./data_quant.json",
                        help="Path to calibration dataset JSON")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device for model loading")
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max_context", type=int, default=4096)

    # Single export mode
    parser.add_argument("--quantized_dtype", type=str, choices=["W8A8", "W4A16"])
    parser.add_argument("--quantized_algorithm", type=str, choices=["normal", "grq"])
    parser.add_argument("--group_size", type=int, default=None)

    # Batch export mode
    parser.add_argument("--run_all", action="store_true",
                        help="Run all 8 experiment configurations")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Specific config IDs to run (e.g., R1 R5 R8)")

    args = parser.parse_args()

    if args.run_all:
        export_all(args.model_path, args.output_dir, args.dataset,
                   args.device, args.dtype)
    elif args.configs:
        for cid in args.configs:
            if cid not in EXPERIMENT_CONFIGS:
                print(f"[WARN] Unknown config: {cid}, skipping")
                continue
            cfg = EXPERIMENT_CONFIGS[cid]
            export_single(
                model_path=args.model_path,
                output_dir=args.output_dir,
                dataset_path=args.dataset,
                device=args.device,
                dtype=args.dtype,
                **cfg,
            )
    elif args.quantized_dtype and args.quantized_algorithm:
        export_single(
            model_path=args.model_path,
            quantized_dtype=args.quantized_dtype,
            quantized_algorithm=args.quantized_algorithm,
            group_size=args.group_size,
            output_dir=args.output_dir,
            dataset_path=args.dataset,
            device=args.device,
            dtype=args.dtype,
            max_context=args.max_context,
        )
    else:
        parser.print_help()
        print("\nError: Specify --run_all, --configs, or --quantized_dtype + --quantized_algorithm")
        sys.exit(1)


if __name__ == "__main__":
    main()
