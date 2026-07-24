"""
Merge multiple per-context NPU verify probe JSON files into one simulator-ready JSON.

Input files are produced by benchmark/npu_verify_probe.py --oracle_json

Example:
    python analysis/merge_verify_profiles.py \
        --inputs \
          results/raw/npu_verify_ctx32_oracle.json \
          results/raw/npu_verify_ctx256_oracle.json \
          results/raw/npu_verify_ctx1024_oracle.json \
          results/raw/npu_verify_ctx4096_oracle.json \
        --output results/raw/npu_verify_merged.json
"""
import argparse
import json
import os


def merge_inputs(paths):
    merged = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for ctx, inner in data.items():
            if ctx not in merged:
                merged[ctx] = {}
            merged[ctx].update(inner)
    return dict(sorted(merged.items(), key=lambda kv: int(kv[0])))


def main():
    parser = argparse.ArgumentParser(description="Merge per-context verify probe outputs")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input oracle JSON files")
    parser.add_argument("--output", required=True, help="Merged output JSON")
    args = parser.parse_args()

    merged = merge_inputs(args.inputs)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(args.inputs)} files -> {args.output}")
    print("Contexts:", ", ".join(sorted(merged.keys(), key=int)))


if __name__ == "__main__":
    main()
