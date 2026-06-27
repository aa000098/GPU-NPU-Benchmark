"""
결과 수집 및 통합 스크립트
PPL 로그 파일과 벤치마크 데이터를 파싱하여 full_results.csv 생성.

Usage:
    python analysis/collect_results.py
"""
import os
import re
import csv
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "raw")
MNN_MODELS_DIR = os.path.join(BASE_DIR, "mnn_models")

# Benchmark results (llm_bench output, manually recorded)
# TODO: 추후 bench_mnn.py가 자동으로 JSON 저장하면 그걸 파싱하도록 변경
BENCHMARK_DATA = {
    "M1_q4_ch_direct":  {"pp64": 130.13, "pp256": 115.68, "tg256": 29.89},
    "M2_q4_b64_direct": {"pp64":  94.00, "pp256":  87.19, "tg256": 24.59},
    "M3_q4_b128_direct":{"pp64": 112.22, "pp256":  99.03, "tg256": 25.61},
    "M4_q8_ch_direct":  {"pp64": 152.53, "pp256": 144.28, "tg256": 15.63},
    "M5_q8_b64_direct": {"pp64": 109.53, "pp256": 104.37, "tg256": 14.01},
    "M6_q4_b64_awq":    {"pp64":  96.96, "pp256":  87.32, "tg256": 24.23},
    "M7_q4_b64_smooth":  {"pp64":  91.39, "pp256":  78.07, "tg256": 23.83},
    "M8_q4_b64_hqq":    {"pp64":  93.56, "pp256":  85.87, "tg256": 26.95},
}

# RKNN-LLM benchmark results (NPU, 3 cores)
RKNN_BENCHMARK_DATA = {
    "R1_W8A8":      {"size_mb": 1704, "ttft_p64": 49.4, "tg_p64": 19.97, "ttft_p256": 51.5, "tg_p256": 19.07, "ppl": 16.93},
    "R2_W8A8_G128": {"size_mb": 1798, "ttft_p64": 78.3, "tg_p64": 12.39, "ttft_p256": 80.4, "tg_p256": 12.16, "ppl": 16.04},
    "R3_W8A8_G256": {"size_mb": 1750, "ttft_p64": 63.9, "tg_p64": 15.22, "ttft_p256": 66.2, "tg_p256": 14.76, "ppl": 16.11},
    "R4_W8A8_G512": {"size_mb": 1727, "ttft_p64": 59.5, "tg_p64": 16.30, "ttft_p256": 62.2, "tg_p256": 15.71, "ppl": 16.22},
}


def parse_model_info(model_name):
    """모델 이름에서 양자화 정보 추출."""
    info = {"quant_bit": "", "block_size": "", "method": ""}

    if "q4" in model_name:
        info["quant_bit"] = "4bit"
    elif "q8" in model_name:
        info["quant_bit"] = "8bit"

    if "_ch_" in model_name:
        info["block_size"] = "channel"
    else:
        m = re.search(r"_b(\d+)_", model_name)
        if m:
            info["block_size"] = m.group(1)

    for method in ["direct", "awq", "smooth", "hqq"]:
        if method in model_name:
            info["method"] = method
            break

    return info


def get_model_size_mb(model_name):
    """MNN 모델 디렉토리의 총 크기 계산."""
    model_dir = os.path.join(MNN_MODELS_DIR, model_name)
    if not os.path.isdir(model_dir):
        return 0
    total = 0
    for f in os.listdir(model_dir):
        fpath = os.path.join(model_dir, f)
        if os.path.isfile(fpath):
            total += os.path.getsize(fpath)
    return total / (1024 * 1024)


def parse_ppl_log(log_path):
    """PPL 로그 파일에서 Perplexity 값 추출."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        for line in f:
            m = re.search(r"Perplexity:\s*([\d.]+)", line)
            if m:
                return float(m.group(1))
    return None


def collect_all():
    """모든 결과를 수집하여 통합."""
    results = []

    # MNN 모델 디렉토리 스캔
    if os.path.isdir(MNN_MODELS_DIR):
        model_dirs = sorted([
            d for d in os.listdir(MNN_MODELS_DIR)
            if os.path.isdir(os.path.join(MNN_MODELS_DIR, d)) and d.startswith("M")
        ])
    else:
        model_dirs = list(BENCHMARK_DATA.keys())

    for model_name in model_dirs:
        info = parse_model_info(model_name)
        size_mb = get_model_size_mb(model_name)

        # PPL from log files
        ppl_log = os.path.join(RESULTS_DIR, f"ppl_{model_name}.log")
        ppl = parse_ppl_log(ppl_log)

        # Benchmark data
        bench = BENCHMARK_DATA.get(model_name, {})

        results.append({
            "model": model_name,
            "framework": "mnn",
            "backend": "cpu",
            "size_mb": round(size_mb, 1),
            "quant_bit": info["quant_bit"],
            "block_size": info["block_size"],
            "method": info["method"],
            "prefill_64_tok_s": bench.get("pp64"),
            "prefill_256_tok_s": bench.get("pp256"),
            "decode_256_tok_s": bench.get("tg256"),
            "perplexity": round(ppl, 2) if ppl else None,
        })

    # RKNN-LLM 결과 추가
    for model_name, bench in RKNN_BENCHMARK_DATA.items():
        results.append({
            "model": model_name,
            "framework": "rknn-llm",
            "backend": "npu",
            "size_mb": bench["size_mb"],
            "quant_bit": "8bit",
            "block_size": model_name.split("_")[-1] if "G" in model_name else "channel",
            "method": "normal",
            "prefill_64_tok_s": None,  # RKNN uses TTFT instead
            "prefill_256_tok_s": None,
            "decode_256_tok_s": bench["tg_p64"],
            "perplexity": bench.get("ppl"),
            "ttft_64_ms": bench["ttft_p64"],
            "ttft_256_ms": bench["ttft_p256"],
            "decode_p256_tok_s": bench["tg_p256"],
        })

    return results


def save_results(results):
    """결과를 CSV와 JSON으로 저장."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CSV - collect all fieldnames from all results
    csv_path = os.path.join(RESULTS_DIR, "full_results.csv")
    with open(csv_path, "w", newline="") as f:
        if results:
            all_keys = []
            for r in results:
                for k in r.keys():
                    if k not in all_keys:
                        all_keys.append(k)
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
    print(f"CSV saved: {csv_path}")

    # JSON (per-model)
    for r in results:
        name = r["model"]
        json_path = os.path.join(RESULTS_DIR, f"mnn_{name}.json")
        with open(json_path, "w") as f:
            json.dump(r, f, indent=2)

    # Combined JSON
    combined_path = os.path.join(RESULTS_DIR, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON saved: {combined_path}")


def print_summary(results):
    """결과 요약 출력."""
    # MNN results
    mnn = [r for r in results if r["framework"] == "mnn"]
    rknn = [r for r in results if r["framework"] == "rknn-llm"]

    print()
    print("=" * 80)
    print("MNN CPU (4 threads) — Llama 3.2 1B on RK3588")
    print("=" * 80)
    print(f"{'Model':<22} {'Size':>7} {'PP64':>7} {'PP256':>7} {'TG256':>7} {'PPL':>7}")
    print("-" * 80)
    for r in mnn:
        pp64 = f"{r['prefill_64_tok_s']:.1f}" if r["prefill_64_tok_s"] else "N/A"
        pp256 = f"{r['prefill_256_tok_s']:.1f}" if r["prefill_256_tok_s"] else "N/A"
        tg = f"{r['decode_256_tok_s']:.2f}" if r["decode_256_tok_s"] else "N/A"
        ppl = f"{r['perplexity']:.2f}" if r["perplexity"] else "N/A"
        print(f"{r['model']:<22} {r['size_mb']:>6.0f}M {pp64:>7} {pp256:>7} {tg:>7} {ppl:>7}")

    if rknn:
        print()
        print("=" * 80)
        print("RKNN-LLM NPU (3 cores) — Llama 3.2 1B on RK3588")
        print("=" * 80)
        print(f"{'Model':<22} {'Size':>7} {'TTFT64':>8} {'TG64':>7} {'TTFT256':>8} {'TG256':>7}")
        print("-" * 80)
        for r in rknn:
            ttft64 = f"{r.get('ttft_64_ms', 0):.1f}ms" if r.get("ttft_64_ms") else "N/A"
            tg64 = f"{r['decode_256_tok_s']:.2f}" if r["decode_256_tok_s"] else "N/A"
            ttft256 = f"{r.get('ttft_256_ms', 0):.1f}ms" if r.get("ttft_256_ms") else "N/A"
            tg256 = f"{r.get('decode_p256_tok_s', 0):.2f}" if r.get("decode_p256_tok_s") else "N/A"
            print(f"{r['model']:<22} {r['size_mb']:>6.0f}M {ttft64:>8} {tg64:>7} {ttft256:>8} {tg256:>7}")
    print()


def main():
    results = collect_all()
    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()
