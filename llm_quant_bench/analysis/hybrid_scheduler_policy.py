"""
Reusable scheduling policy for hybrid speculative decoding.

This module turns measured profiles into runtime decisions:
- CPU-only
- NPU-only
- Hybrid with draft length k

It is designed to be shared by:
- offline oracle analysis
- future end-to-end runtime implementation
"""
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Baseline decode measurements from current characterization
CPU_DECODE_MS = {
    32: 64.6,
    64: 60.4,
    128: 63.5,
    256: 59.9,
    512: 60.2,
    1024: 61.1,
    2048: 60.8,
    4096: 63.4,
}
NPU_DECODE_MS = {
    32: 50.0,
    64: 50.0,
    128: 51.0,
    256: 54.9,
    512: 55.2,
    1024: 58.8,
    2048: 68.5,
    4096: 88.5,
}
DRAFT_MS_PER_TOKEN = {
    32: 3.0,
    64: 3.0,
    128: 3.1,
    256: 3.2,
    512: 3.3,
    1024: 3.5,
    2048: 3.7,
    4096: 4.0,
}
DEFAULT_VERIFY_MS = {
    32:   {1: 50.0, 2: 54.0, 4: 60.0, 8: 73.0, 16: 98.0},
    64:   {1: 50.0, 2: 54.0, 4: 60.0, 8: 73.0, 16: 98.0},
    128:  {1: 51.0, 2: 55.0, 4: 61.0, 8: 74.0, 16: 100.0},
    256:  {1: 54.9, 2: 59.0, 4: 66.0, 8: 81.0, 16: 110.0},
    512:  {1: 55.2, 2: 60.0, 4: 68.0, 8: 85.0, 16: 116.0},
    1024: {1: 58.8, 2: 64.0, 4: 74.0, 8: 93.0, 16: 128.0},
    2048: {1: 68.5, 2: 74.0, 4: 86.0, 8: 110.0, 16: 153.0},
    4096: {1: 88.5, 2: 95.0, 4: 112.0, 8: 145.0, 16: 204.0},
}


@dataclass
class PolicyDecision:
    context: int
    best_mode: str
    best_k: Optional[int]
    expected_ms_per_token: float
    cpu_ms_per_token: float
    npu_ms_per_token: float
    hybrid_ms_per_token: Optional[float]
    hybrid_speedup_vs_cpu: Optional[float]
    hybrid_speedup_vs_npu: Optional[float]


def load_verify_profile(path: Optional[str]) -> Dict[int, Dict[int, float]]:
    if path is None:
        return DEFAULT_VERIFY_MS
    with open(path) as f:
        data = json.load(f)
    return {
        int(ctx): {int(k): float(v) for k, v in inner.items()}
        for ctx, inner in data.items()
    }


def nearest_context(context: int, available_contexts: List[int]) -> int:
    return min(available_contexts, key=lambda x: abs(x - context))


def correction_overhead_ms(context: int, npu_decode_ms: float) -> float:
    # Simple approximation until runtime measurements become available
    _ = context
    return 0.25 * npu_decode_ms


def hybrid_ms_per_token(
    context: int,
    k: int,
    acceptance: float,
    verify_ms: float,
    draft_ms_per_token: float,
    npu_decode_ms: float,
) -> float:
    acceptance = max(1e-4, min(0.999, acceptance))
    expected_accepted = sum(acceptance ** i for i in range(1, k + 1))
    draft_total_ms = draft_ms_per_token * k
    corr_ms = correction_overhead_ms(context, npu_decode_ms) * (1.0 - acceptance)
    return (draft_total_ms + verify_ms + corr_ms) / max(expected_accepted, 1e-6)


def choose_policy(
    context: int,
    acceptance: float,
    verify_profile: Dict[int, Dict[int, float]],
    draft_ms_per_token_profile: Dict[int, float] = DRAFT_MS_PER_TOKEN,
    cpu_decode_profile: Dict[int, float] = CPU_DECODE_MS,
    npu_decode_profile: Dict[int, float] = NPU_DECODE_MS,
) -> PolicyDecision:
    available_contexts = sorted(verify_profile.keys())
    ctx = nearest_context(context, available_contexts)

    cpu_ms = cpu_decode_profile[ctx]
    npu_ms = npu_decode_profile[ctx]
    draft_ms = draft_ms_per_token_profile[ctx]

    best_hybrid_ms = None
    best_k = None
    for k, verify_ms in sorted(verify_profile[ctx].items()):
        ms = hybrid_ms_per_token(
            context=ctx,
            k=k,
            acceptance=acceptance,
            verify_ms=verify_ms,
            draft_ms_per_token=draft_ms,
            npu_decode_ms=npu_ms,
        )
        if best_hybrid_ms is None or ms < best_hybrid_ms:
            best_hybrid_ms = ms
            best_k = k

    candidates: List[Tuple[str, float, Optional[int]]] = [
        ("CPU-only", cpu_ms, None),
        ("NPU-only", npu_ms, None),
    ]
    if best_hybrid_ms is not None:
        candidates.append(("Hybrid", best_hybrid_ms, best_k))

    best_mode, expected_ms, best_mode_k = min(candidates, key=lambda x: x[1])
    hybrid_speedup_vs_cpu = None if best_hybrid_ms is None else cpu_ms / best_hybrid_ms
    hybrid_speedup_vs_npu = None if best_hybrid_ms is None else npu_ms / best_hybrid_ms

    return PolicyDecision(
        context=ctx,
        best_mode=best_mode,
        best_k=best_mode_k if best_mode == "Hybrid" else None,
        expected_ms_per_token=expected_ms,
        cpu_ms_per_token=cpu_ms,
        npu_ms_per_token=npu_ms,
        hybrid_ms_per_token=best_hybrid_ms,
        hybrid_speedup_vs_cpu=hybrid_speedup_vs_cpu,
        hybrid_speedup_vs_npu=hybrid_speedup_vs_npu,
    )


def print_decision(decision: PolicyDecision, acceptance: float):
    print(
        f"context={decision.context} acceptance={acceptance:.3f} -> "
        f"{decision.best_mode}"
        + (f"(k={decision.best_k})" if decision.best_k is not None else "")
        + f", expected_ms/tok={decision.expected_ms_per_token:.2f}, "
        f"cpu={decision.cpu_ms_per_token:.2f}, npu={decision.npu_ms_per_token:.2f}, "
        f"hybrid={decision.hybrid_ms_per_token:.2f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Inspect hybrid scheduling policy")
    parser.add_argument("--verify_json", type=str, default=None,
                        help="Merged verify profile JSON")
    parser.add_argument("--contexts", nargs="+", type=int,
                        default=[32, 256, 1024, 2048, 4096])
    parser.add_argument("--acceptances", nargs="+", type=float,
                        default=[0.95, 0.90, 0.85, 0.80, 0.75])
    args = parser.parse_args()

    verify_profile = load_verify_profile(args.verify_json)
    print("=== Hybrid Scheduler Policy Sweep ===")
    for context in args.contexts:
        for acceptance in args.acceptances:
            decision = choose_policy(
                context=context,
                acceptance=acceptance,
                verify_profile=verify_profile,
            )
            print_decision(decision, acceptance)


if __name__ == "__main__":
    main()
