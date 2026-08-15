"""
Arithmetic Intensity Calculator
LLM 연산별 산술 밀도(FLOPS/Byte) 해석적 계산.
Prefill / Decode 단계 × 양자화 설정별 분석.

Usage:
    python analysis/arithmetic_intensity.py
"""
import os
import csv
import json
import argparse
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, "config", "roofline_config.yaml")


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


def compute_prefill_intensity(seq_len, arch, quant):
    """
    Prefill 단계 산술 밀도 계산.

    Linear projection (QKV + O + FFN):
      FLOPS = 2 * seq_len * hidden * out_dim
      Bytes = weights + input_activations + output_activations

    Attention (QK^T, softmax*V):
      FLOPS = 2 * num_heads * seq_len^2 * head_dim * 2 (QK^T + AV)
      Bytes = Q + K + V + output

    Returns dict with per-operation and total intensity.
    """
    d = arch["hidden_dim"]
    h = arch["num_heads"]
    hd = arch["head_dim"]
    kv_h = arch["num_kv_heads"]
    L = arch["num_layers"]
    ff = arch["intermediate_size"]
    N = seq_len

    bw = quant["bytes_per_weight"]
    ba = quant["bytes_per_activation"]
    bkv = quant["bytes_per_kv"]

    results = []

    # --- Per layer operations ---

    # 1. QKV projection: [N, d] × [d, 3d] (or d + 2*kv_h*hd for GQA)
    qkv_out_dim = d + 2 * kv_h * hd  # Q: d, K: kv_h*hd, V: kv_h*hd
    qkv_flops = 2 * N * d * qkv_out_dim
    qkv_bytes = (d * qkv_out_dim * bw +  # weights
                 N * d * ba +              # input
                 N * qkv_out_dim * ba)     # output
    qkv_intensity = qkv_flops / qkv_bytes if qkv_bytes > 0 else 0
    results.append({
        "operation": "qkv_proj", "phase": "prefill", "seq_len": N,
        "flops": qkv_flops * L, "bytes": qkv_bytes * L,
        "intensity": round(qkv_intensity, 3),
    })

    # 2. Attention: QK^T → [N, N] per head
    attn_qkt_flops = 2 * h * N * N * hd
    attn_qkt_bytes = (N * d * ba +      # Q
                      N * kv_h * hd * bkv)  # K
    attn_av_flops = 2 * h * N * N * hd
    attn_av_bytes = (h * N * N * ba +    # attention weights
                     N * kv_h * hd * bkv +  # V
                     N * d * ba)         # output
    attn_flops = attn_qkt_flops + attn_av_flops
    attn_bytes = attn_qkt_bytes + attn_av_bytes
    attn_intensity = attn_flops / attn_bytes if attn_bytes > 0 else 0
    results.append({
        "operation": "attention", "phase": "prefill", "seq_len": N,
        "flops": attn_flops * L, "bytes": attn_bytes * L,
        "intensity": round(attn_intensity, 3),
    })

    # 3. Output projection: [N, d] × [d, d]
    o_flops = 2 * N * d * d
    o_bytes = d * d * bw + N * d * ba + N * d * ba
    o_intensity = o_flops / o_bytes if o_bytes > 0 else 0
    results.append({
        "operation": "o_proj", "phase": "prefill", "seq_len": N,
        "flops": o_flops * L, "bytes": o_bytes * L,
        "intensity": round(o_intensity, 3),
    })

    # 4. FFN: up [N, d] × [d, ff], gate [N, d] × [d, ff], down [N, ff] × [ff, d]
    ffn_flops = 2 * N * d * ff * 3  # up + gate + down
    ffn_bytes = (3 * d * ff * bw +    # weights (up, gate, down)
                 N * d * ba +          # input
                 N * ff * ba * 2 +     # intermediate (up, gate outputs)
                 N * d * ba)           # output
    ffn_intensity = ffn_flops / ffn_bytes if ffn_bytes > 0 else 0
    results.append({
        "operation": "ffn", "phase": "prefill", "seq_len": N,
        "flops": ffn_flops * L, "bytes": ffn_bytes * L,
        "intensity": round(ffn_intensity, 3),
    })

    # Total prefill
    total_flops = (qkv_flops + attn_flops + o_flops + ffn_flops) * L
    total_bytes = (qkv_bytes + attn_bytes + o_bytes + ffn_bytes) * L
    total_intensity = total_flops / total_bytes if total_bytes > 0 else 0
    results.append({
        "operation": "total", "phase": "prefill", "seq_len": N,
        "flops": total_flops, "bytes": total_bytes,
        "intensity": round(total_intensity, 3),
    })

    return results


def compute_decode_intensity(seq_len, arch, quant):
    """
    Decode 단계 산술 밀도 (배치 크기 = 1).
    seq_len = 현재까지의 KV 캐시 길이.
    """
    d = arch["hidden_dim"]
    h = arch["num_heads"]
    hd = arch["head_dim"]
    kv_h = arch["num_kv_heads"]
    L = arch["num_layers"]
    ff = arch["intermediate_size"]
    S = seq_len  # KV cache length

    bw = quant["bytes_per_weight"]
    ba = quant["bytes_per_activation"]
    bkv = quant["bytes_per_kv"]

    results = []

    # 1. QKV projection: [1, d] × [d, qkv_out]
    qkv_out_dim = d + 2 * kv_h * hd
    qkv_flops = 2 * 1 * d * qkv_out_dim
    qkv_bytes = d * qkv_out_dim * bw + 1 * d * ba + 1 * qkv_out_dim * ba
    results.append({
        "operation": "qkv_proj", "phase": "decode", "seq_len": S,
        "flops": qkv_flops * L, "bytes": qkv_bytes * L,
        "intensity": round(qkv_flops / qkv_bytes, 3) if qkv_bytes > 0 else 0,
    })

    # 2. Attention with KV cache: Q[1,hd] × K[S,hd]^T → [1,S], then ×V[S,hd]
    attn_flops = 2 * h * 1 * S * hd * 2  # QK^T + AV
    attn_bytes = (1 * d * ba +              # Q
                  2 * S * kv_h * hd * bkv + # K + V cache read
                  1 * d * ba)               # output
    results.append({
        "operation": "attention_kv", "phase": "decode", "seq_len": S,
        "flops": attn_flops * L, "bytes": attn_bytes * L,
        "intensity": round(attn_flops / attn_bytes, 3) if attn_bytes > 0 else 0,
    })

    # 3. O projection: [1, d] × [d, d]
    o_flops = 2 * 1 * d * d
    o_bytes = d * d * bw + 1 * d * ba + 1 * d * ba
    results.append({
        "operation": "o_proj", "phase": "decode", "seq_len": S,
        "flops": o_flops * L, "bytes": o_bytes * L,
        "intensity": round(o_flops / o_bytes, 3) if o_bytes > 0 else 0,
    })

    # 4. FFN
    ffn_flops = 2 * 1 * d * ff * 3
    ffn_bytes = 3 * d * ff * bw + 1 * d * ba + 1 * ff * ba * 2 + 1 * d * ba
    results.append({
        "operation": "ffn", "phase": "decode", "seq_len": S,
        "flops": ffn_flops * L, "bytes": ffn_bytes * L,
        "intensity": round(ffn_flops / ffn_bytes, 3) if ffn_bytes > 0 else 0,
    })

    # Total decode
    total_flops = (qkv_flops + attn_flops + o_flops + ffn_flops) * L
    total_bytes = (qkv_bytes + attn_bytes + o_bytes + ffn_bytes) * L
    results.append({
        "operation": "total", "phase": "decode", "seq_len": S,
        "flops": total_flops, "bytes": total_bytes,
        "intensity": round(total_flops / total_bytes, 3) if total_bytes > 0 else 0,
    })

    return results


def main():
    parser = argparse.ArgumentParser(description="Arithmetic Intensity Calculator")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    arch = cfg["model_arch"]
    quant_configs = cfg["quant_configs"]
    seq_lengths = cfg["llm_profiling"]["seq_lengths"]

    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "results", "raw")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    print(f"{'='*70}")
    print(f"Arithmetic Intensity Analysis — {arch['name']}")
    print(f"{'='*70}")

    for qname, qcfg in quant_configs.items():
        print(f"\n  Quantization: {qname} ({qcfg['description']})")
        print(f"  {'Phase':8s} {'SeqLen':>7s} {'Total FLOPS':>14s} {'Total Bytes':>14s} {'Intensity':>10s}")

        for seq_len in seq_lengths:
            # Prefill
            prefill_results = compute_prefill_intensity(seq_len, arch, qcfg)
            for r in prefill_results:
                r["quant"] = qname
            all_results.extend(prefill_results)

            total_p = [r for r in prefill_results if r["operation"] == "total"][0]
            print(f"  {'prefill':8s} {seq_len:>7d} {total_p['flops']:>14.2e} "
                  f"{total_p['bytes']:>14.2e} {total_p['intensity']:>10.3f}")

            # Decode
            decode_results = compute_decode_intensity(seq_len, arch, qcfg)
            for r in decode_results:
                r["quant"] = qname
            all_results.extend(decode_results)

            total_d = [r for r in decode_results if r["operation"] == "total"][0]
            print(f"  {'decode':8s} {seq_len:>7d} {total_d['flops']:>14.2e} "
                  f"{total_d['bytes']:>14.2e} {total_d['intensity']:>10.3f}")

    # Save CSV
    csv_path = os.path.join(output_dir, "arithmetic_intensity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "quant", "operation", "phase", "seq_len", "flops", "bytes", "intensity"
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved {len(all_results)} entries to {csv_path}")

    # Save JSON
    json_path = os.path.join(output_dir, "arithmetic_intensity.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
