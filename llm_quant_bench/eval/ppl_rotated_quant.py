"""Wikitext-2 perplexity with rotated scalar KV quantization (TurboQuant style).

Patches HF Llama-3.2-1B to apply rotation + scalar quantization on cached K, V
during attention. Compared against fp16 baseline. Sweeps bit-rates 4/3/2.
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.sched_setaffinity(0, {4, 5, 6, 7})

HF_MODEL = ROOT / "llm_quant_bench/models/Llama-3.2-1B-Instruct"
HEAD_DIM = 64


def hadamard_matrix(d):
    H = np.array([[1.0]])
    while H.shape[0] < d:
        H = np.block([[H, H], [H, -H]])
    return H


def make_rotation(d, seed=42):
    H = hadamard_matrix(d) / np.sqrt(d)
    rng = np.random.default_rng(seed)
    D = rng.choice([-1.0, 1.0], size=d)
    return (H * D).astype(np.float32)


def lloyd_max_gaussian(bits, n_samples=200000, n_iter=200, seed=0):
    K = 2 ** bits
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(n_samples).astype(np.float32)
    samples.sort()
    edges = np.quantile(samples, np.linspace(0, 1, K + 1))
    centroids = np.zeros(K, dtype=np.float32)
    for i in range(K):
        if i < K - 1:
            mask = (samples >= edges[i]) & (samples < edges[i+1])
        else:
            mask = samples >= edges[i]
        centroids[i] = samples[mask].mean() if mask.any() else (edges[i] + edges[i+1]) / 2
    for _ in range(n_iter):
        b = (centroids[:-1] + centroids[1:]) / 2
        idx = np.searchsorted(b, samples)
        new_c = np.array([samples[idx == i].mean() if (idx == i).any() else centroids[i]
                          for i in range(K)], dtype=np.float32)
        if np.allclose(new_c, centroids, atol=1e-6):
            break
        centroids = new_c
    return np.sort(centroids)


def quantize_rotated_torch(X, R_t, centroids_t):
    """Apply rotation, normalize, scalar quantize via nearest centroid.
    X shape (..., d)."""
    import torch
    orig_shape = X.shape
    Xf = X.reshape(-1, orig_shape[-1]).float()
    Y = Xf @ R_t.t()
    norms = Y.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    Y_n = Y / norms
    # nearest centroid via boundaries
    bnd = (centroids_t[:-1] + centroids_t[1:]) / 2
    idx = torch.searchsorted(bnd, Y_n.contiguous())
    idx = idx.clamp(0, centroids_t.numel() - 1)
    Y_q = centroids_t[idx]
    Y_dq = Y_q * norms
    X_dq = Y_dq @ R_t  # rotate back (R orthonormal)
    return X_dq.reshape(orig_shape).to(X.dtype)


def patch_attention_with_rotated_quant(model, R_t, centroids_t):
    """Wrap each layer's forward to quantize newly-appended K, V tokens.

    Tracks per-layer last quantized seq_len; only quantizes the new tail each
    call. This matches the production-grade incremental index update flow.
    """
    import torch
    handles = []
    last_len = [0] * len(model.model.layers)

    for L, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        orig_forward = attn.forward

        def make_wrapped(orig, layer_idx):
            def wrapped(*args, **kwargs):
                out = orig(*args, **kwargs)
                pkv = kwargs.get("past_key_value", None)
                if pkv is not None and hasattr(pkv, "key_cache"):
                    if layer_idx < len(pkv.key_cache):
                        k_t = pkv.key_cache[layer_idx]
                        v_t = pkv.value_cache[layer_idx]
                        cur = k_t.shape[-2]
                        prev = last_len[layer_idx]
                        if cur > prev:
                            new_k = k_t[..., prev:, :]
                            new_v = v_t[..., prev:, :]
                            new_k_q = quantize_rotated_torch(new_k, R_t, centroids_t)
                            new_v_q = quantize_rotated_torch(new_v, R_t, centroids_t)
                            k_t[..., prev:, :] = new_k_q
                            v_t[..., prev:, :] = new_v_q
                            last_len[layer_idx] = cur
                        elif cur < prev:
                            last_len[layer_idx] = 0  # cache reset
                return out
            return wrapped

        attn.forward = make_wrapped(orig_forward, L)
        handles.append((attn, orig_forward))
    return handles


def restore_attention(handles):
    for attn, orig in handles:
        attn.forward = orig


def evaluate_ppl(model, tok, dataset_text, max_length=512, stride=256):
    """Stride-based perplexity following HF's eval recipe."""
    import torch
    encodings = tok(dataset_text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = encodings.input_ids[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(input_ids, labels=target_ids)
            neg_log_likelihood = out.loss
        nlls.append(neg_log_likelihood.float() * trg_len)
        prev_end = end
        if end == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / end).item()
    return ppl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=str, default="4,3,2")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--n_chars", type=int, default=20000, help="chars from wikitext")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import load_dataset

    print("[load] Llama-3.2-1B...")
    tok = AutoTokenizer.from_pretrained(str(HF_MODEL))
    model = AutoModelForCausalLM.from_pretrained(str(HF_MODEL), torch_dtype=torch.float16)
    model.eval()

    print("[load] wikitext-2 test split...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([s["text"] for s in ds if len(s["text"].strip()) > 0])
    text = text[:args.n_chars]
    print(f"  using {len(text)} chars")

    print("\n[setup] rotation + codebooks (data-oblivious — no real KV data used)...")
    R = make_rotation(HEAD_DIM, seed=42)
    R_t = torch.from_numpy(R)

    bit_rates = [int(b) for b in args.bits.split(",")]
    codebooks_t = {}
    for b in bit_rates:
        cb = lloyd_max_gaussian(b) / math.sqrt(HEAD_DIM)
        codebooks_t[b] = torch.from_numpy(cb)

    print("\n=== Wikitext-2 Perplexity ===")
    print(f"  max_length={args.max_length}, stride={args.stride}")
    print(f"\n{'config':>20} {'PPL':>10} {'rel Δ':>10}")
    t0 = time.perf_counter()
    ppl_baseline = evaluate_ppl(model, tok, text, args.max_length, args.stride)
    print(f"{'fp16 baseline':>20} {ppl_baseline:>10.3f}  {'(ref)':>10}  ({time.perf_counter()-t0:.1f}s)")

    for b in bit_rates:
        cb_t = codebooks_t[b]
        handles = patch_attention_with_rotated_quant(model, R_t, cb_t)
        try:
            t0 = time.perf_counter()
            ppl_q = evaluate_ppl(model, tok, text, args.max_length, args.stride)
        finally:
            restore_attention(handles)
        rel = (ppl_q - ppl_baseline) / ppl_baseline * 100
        print(f"{f'rot+{b}bit ({2**b} lvls)':>20} {ppl_q:>10.3f} {rel:>+9.2f}%  ({time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
