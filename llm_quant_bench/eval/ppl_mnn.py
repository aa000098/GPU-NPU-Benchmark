"""
MNN Perplexity Evaluator Wrapper
MNN의 evaluate_perplexity.py를 래핑하여 여러 모델에 대해 실행하고 결과를 수집.

Usage:
    python ppl_mnn.py --model_dir ./mnn_models/M1_q4_ch_direct
    python ppl_mnn.py --all_models_dir ./mnn_models --output_dir ./results/raw
"""
import os
import sys
import json
import argparse
import subprocess
import re

MNN_EVAL_SCRIPT = "/home/hyunho.son/install_files/MNN/transformers/llm/eval/evaluate_perplexity.py"
MNN_PPL_EVAL_BIN = "/home/hyunho.son/install_files/MNN/build/ppl_eval"


def find_config_json(model_dir):
    for name in ["config.json", "llm_config.json"]:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            return path
    return None


def evaluate_with_python(model_dir, dataset="wikitext/wikitext-2-raw-v1", quant_qkv=8):
    """Use MNN Python API for perplexity evaluation."""
    config_path = find_config_json(model_dir)
    if config_path is None:
        print(f"[ERROR] No config.json in {model_dir}")
        return None

    cmd = [
        sys.executable, MNN_EVAL_SCRIPT,
        "-m", config_path,
        "-d", dataset,
        "--quant-qkv", str(quant_qkv),
    ]

    print(f"  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)

    if result.returncode != 0:
        print(f"[ERROR] PPL eval failed: {result.stderr[-300:]}")
        return None

    # Parse perplexity from output
    for line in result.stdout.strip().split("\n"):
        m = re.search(r'Perplexity:\s*([\d.]+)', line)
        if m:
            return float(m.group(1))

    print(f"[WARN] Could not parse perplexity from output:\n{result.stdout[-300:]}")
    return None


def evaluate_with_binary(model_dir, dataset_path=None):
    """Use compiled ppl_eval binary (if available)."""
    if not os.path.exists(MNN_PPL_EVAL_BIN):
        return None

    config_path = find_config_json(model_dir)
    if config_path is None:
        return None

    cmd = [MNN_PPL_EVAL_BIN, config_path]
    if dataset_path:
        cmd.append(dataset_path)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=None)
    if result.returncode != 0:
        return None

    for line in result.stdout.strip().split("\n"):
        m = re.search(r'[Pp]erplexity\s*[:=]\s*([\d.]+)', line)
        if m:
            return float(m.group(1))
    return None


def evaluate_model(model_dir, output_dir, dataset="wikitext/wikitext-2-raw-v1"):
    dir_name = os.path.basename(model_dir.rstrip("/"))
    print(f"\n{'='*60}")
    print(f"Evaluating PPL: {dir_name}")
    print(f"{'='*60}")

    # Try Python API first
    ppl = evaluate_with_python(model_dir, dataset)
    if ppl is None:
        # Fallback to binary
        ppl = evaluate_with_binary(model_dir)

    if ppl is not None:
        result = {
            "model": dir_name,
            "framework": "mnn",
            "perplexity": round(ppl, 4),
            "dataset": dataset,
        }

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"ppl_mnn_{dir_name}.json")
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Perplexity: {ppl:.4f}")
        print(f"  Saved to {output_path}")
        return result
    else:
        print(f"  [FAIL] Could not evaluate perplexity")
        return None


def main():
    parser = argparse.ArgumentParser(description="MNN Perplexity Evaluator")
    parser.add_argument("--model_dir", type=str, help="Single MNN model directory")
    parser.add_argument("--all_models_dir", type=str,
                        help="Directory containing multiple MNN model dirs")
    parser.add_argument("--output_dir", type=str, default="./results/raw")
    parser.add_argument("--dataset", type=str, default="wikitext/wikitext-2-raw-v1")
    args = parser.parse_args()

    if args.model_dir:
        evaluate_model(args.model_dir, args.output_dir, args.dataset)
    elif args.all_models_dir:
        model_dirs = sorted([
            os.path.join(args.all_models_dir, d)
            for d in os.listdir(args.all_models_dir)
            if os.path.isdir(os.path.join(args.all_models_dir, d))
            and not d.startswith(".")
        ])
        print(f"Found {len(model_dirs)} model directories")
        results = []
        for md in model_dirs:
            r = evaluate_model(md, args.output_dir, args.dataset)
            if r:
                results.append(r)

        # Summary
        summary_path = os.path.join(args.output_dir, "ppl_mnn_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
