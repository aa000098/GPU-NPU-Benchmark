"""Compute integral-form cross-over n_gen for each (initial_ctx, backend pair).

Replaces the constant-t_decode model (eq 3) with the effective-ctx integral
model (eq 3'), where decode cost depends on instantaneous effective ctx
(initial_ctx + tokens_generated_so_far) via piecewise-linear interpolation
of t_dec measurements (Table 2 of paper).
"""
import csv
from pathlib import Path

# Measured at 5 ctx values: TTFT (sec), decode tok/s
# From v7_paper.md tables 1, 2.
CTX = [15, 512, 1024, 2048, 3500]
TTFT = {  # seconds
    "NPU": [0.128, 1.95, 4.42, 10.85, 23.06],
    "CPU": [0.201, 3.55, 8.10, 21.74, 38.42],
    "GPU": [0.269, 2.87, 5.99, 14.34, 28.83],
}
TOKS = {  # tok/s
    "NPU": [20.80, 15.09, 12.20, 8.46, 6.12],
    "CPU": [16.18, 15.01, 14.80, 13.84, 13.05],
    "GPU": [14.93, 13.76, 13.04, 12.68, 11.41],
}
TDEC_MS = {b: [1000.0/t for t in TOKS[b]] for b in TOKS}  # ms / token


def t_dec(b, c):
    pts = list(zip(CTX, TDEC_MS[b]))
    if c <= pts[0][0]:
        x0, y0 = pts[0]; x1, y1 = pts[1]
        return y0 + (c - x0) * (y1 - y0) / (x1 - x0)
    if c >= pts[-1][0]:
        x0, y0 = pts[-2]; x1, y1 = pts[-1]
        return y1 + (c - x1) * (y1 - y0) / (x1 - x0)
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]; x1, y1 = pts[i + 1]
        if x0 <= c <= x1:
            return y0 + (c - x0) * (y1 - y0) / (x1 - x0)


def integral_t_dec(b, a, b_end):
    """Integrate t_dec_b(c) from c=a to c=b_end (ms·tokens), piecewise linear."""
    breakpoints = sorted({a, b_end} | {x for x in CTX if a < x < b_end})
    total = 0.0
    for i in range(len(breakpoints) - 1):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        avg = (t_dec(b, lo) + t_dec(b, hi)) / 2.0
        total += avg * (hi - lo)
    return total


def overall_throughput(b, ic, n_gen):
    """tokens / total_time, where total_time = TTFT(ic) + ∫t_dec(c)dc."""
    ttft_ms = TTFT[b][CTX.index(ic)] * 1000.0
    decode_ms = integral_t_dec(b, ic, ic + n_gen)
    return n_gen / ((ttft_ms + decode_ms) / 1000.0)


def find_crossover(b1, b2, ic, n_gen_max=10000):
    """Find smallest n_gen > 0 where overall_b1(ic, n_gen) == overall_b2(ic, n_gen).

    If the two backends' overall throughput curves never cross in (0, n_gen_max], return None.
    """
    def f(n):
        if n == 0:
            return TTFT[b2][CTX.index(ic)] - TTFT[b1][CTX.index(ic)]
        # equivalently, compare (b1 total time per token) vs (b2 total time per token)
        ttft1 = TTFT[b1][CTX.index(ic)] * 1000.0
        ttft2 = TTFT[b2][CTX.index(ic)] * 1000.0
        d1 = integral_t_dec(b1, ic, ic + n)
        d2 = integral_t_dec(b2, ic, ic + n)
        return (ttft1 + d1) - (ttft2 + d2)

    f0 = f(1)
    f_max = f(n_gen_max)
    if f0 * f_max > 0:
        return None
    lo, hi = 1, n_gen_max
    for _ in range(60):
        mid = (lo + hi) / 2
        fm = f(mid)
        if fm * f0 <= 0:
            hi = mid
        else:
            lo = mid
            f0 = fm
    return (lo + hi) / 2


def best_backend(ic, n_gen):
    return max(["NPU", "CPU", "GPU"], key=lambda b: overall_throughput(b, ic, n_gen))


def main():
    out_dir = Path(__file__).resolve().parents[1] / "paper/v7/v7_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print("== Cross-over n_gen by initial_ctx (effective-ctx integral, eq 3') ==")
    print(f"{'ic':>5} {'NPU↔GPU':>10} {'NPU↔CPU':>10} {'GPU↔CPU':>10}")
    for ic in CTX:
        ng = find_crossover("NPU", "GPU", ic)
        nc = find_crossover("NPU", "CPU", ic)
        gc = find_crossover("GPU", "CPU", ic)

        def fmt(x):
            return f"{x:.0f}" if x is not None else "–"
        print(f"{ic:>5} {fmt(ng):>10} {fmt(nc):>10} {fmt(gc):>10}")
        rows.append({"ctx": ic, "NPU_vs_GPU": fmt(ng),
                     "NPU_vs_CPU": fmt(nc), "GPU_vs_CPU": fmt(gc)})

    with open(out_dir / "v7_crossover_integral.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["ctx", "NPU_vs_GPU", "NPU_vs_CPU", "GPU_vs_CPU"])
        w.writeheader(); w.writerows(rows)

    # Best-backend regions per ctx using cross-over numbers
    print("\n== Best backend regions per ctx ==")
    for ic in CTX:
        ng_set = {}
        for n in [10, 50, 100, 200, 400, 800, 1500, 3000, 6000]:
            ng_set[n] = best_backend(ic, n)
        # also evaluate at cross-over candidates
        crosses = sorted(x for x in [
            find_crossover("NPU", "GPU", ic),
            find_crossover("NPU", "CPU", ic),
            find_crossover("GPU", "CPU", ic),
        ] if x is not None)
        bests = []
        prev_b = None
        sample_pts = sorted(set([1, 5, 10, 20, 50, 100, 200, 400, 800, 1500, 3000, 6000] +
                                 [int(c) for c in crosses for _ in [0]] +
                                 [int(c) - 5 for c in crosses] + [int(c) + 5 for c in crosses]))
        prev = 1
        for n in sample_pts:
            if n <= 0: continue
            b = best_backend(ic, n)
            if b != prev_b:
                if prev_b is not None:
                    bests.append((prev, n - 1, prev_b))
                prev = n
                prev_b = b
        bests.append((prev, sample_pts[-1], prev_b))
        print(f"ic={ic:>5}: ", " | ".join(f"[{a},{b}]→{c}" for a, b, c in bests))

    # Overall throughput predictions vs §4.4 measurements (ctx=3500)
    print("\n== Validation: overall throughput at ctx=3500 (predicted vs measured) ==")
    measured = {32: (1.146, 0.788, 1.031),
                128: (2.996, 2.648, 3.323),
                256: (4.063, 4.385, 5.117),
                512: (4.912, 6.519, 7.154),
                1024: (5.179, 8.659, 8.701),
                2048: (5.519, 10.084, 9.228)}
    print(f"{'n_gen':>6} | {'NPU pred':>9} {'NPU meas':>9} | {'CPU pred':>9} {'CPU meas':>9} | {'GPU pred':>9} {'GPU meas':>9}")
    for n, (mn, mc, mg) in measured.items():
        pn = overall_throughput("NPU", 3500, n)
        pc = overall_throughput("CPU", 3500, n)
        pg = overall_throughput("GPU", 3500, n)
        print(f"{n:>6} | {pn:>9.3f} {mn:>9.3f} | {pc:>9.3f} {mc:>9.3f} | {pg:>9.3f} {mg:>9.3f}")


if __name__ == "__main__":
    main()
