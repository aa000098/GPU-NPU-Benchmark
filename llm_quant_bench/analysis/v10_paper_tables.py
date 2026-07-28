"""Generate v10 paper tables from comprehensive measurement data.

Reads:
- comprehensive_baselines.csv (CPU/GPU rates per ctx)
- npu_uniform.csv (NPU rates per ctx)
- v10_workload_results.csv (workload validations)

Produces markdown tables for v10 paper §4.1 (TTFT/TPOT) and §4.3 (workload routing).
"""
import csv
from pathlib import Path

DATA = Path("/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/paper/v10/v10_data")

CONFIG_ORDER = ['cpu_t4_c47', 'cpu_t2_c45', 'cpu_t2_c67', 'gpu_low', 'gpu_high', 'npu_rkllm']
CONFIG_LABEL = {
    'cpu_t4_c47': 'cpu_t4',
    'cpu_t2_c45': 'cpu_t2_c45',
    'cpu_t2_c67': 'cpu_t2_c67',
    'gpu_low':    'gpu_low',
    'gpu_high':   'gpu_high',
    'npu_rkllm':  'npu_rkllm',
}
CTX_ORDER = [64, 256, 512, 1024, 2048, 3500]


def load_rates():
    rates = {}  # cfg -> {ctx: (pf, dc)}
    for fname in ['comprehensive_baselines.csv', 'npu_uniform.csv', 'gpu_idp_baselines.csv']:
        fpath = DATA / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for row in csv.DictReader(f):
                cid = row.get('config_id', '')
                try:
                    ctx = int(row.get('ctx', 0))
                    pf = float(row.get('prefill_tok_s', 0))
                    dc = float(row.get('decode_tok_s', 0))
                except ValueError:
                    continue
                if pf > 0 and dc > 0:
                    rates.setdefault(cid, {})[ctx] = (pf, dc)
    return rates


def fmt_ttft(rates, cfg, ctx):
    """TTFT in seconds = ctx / prefill_tok_s."""
    if cfg not in rates or ctx not in rates[cfg]:
        return '—'
    pf, _ = rates[cfg][ctx]
    return f"{ctx / pf:.2f}"


def fmt_tpot(rates, cfg, ctx):
    """TPOT in ms/token = 1000 / decode_tok_s."""
    if cfg not in rates or ctx not in rates[cfg]:
        return '—'
    _, dc = rates[cfg][ctx]
    return f"{1000 / dc:.1f}"


def best_idx(values):
    """Return index of min numeric value (treat '—' as inf)."""
    best = float('inf'); best_i = -1
    for i, v in enumerate(values):
        try:
            f = float(v)
            if f < best:
                best = f; best_i = i
        except (ValueError, TypeError):
            pass
    return best_i


def gen_table_ttft(rates):
    print("표 1. 백엔드별 TTFT (n_p, 초)")
    header = "| n_p | " + " | ".join(CONFIG_LABEL[c] for c in CONFIG_ORDER) + " |"
    sep = "|---:|" + "---:|" * len(CONFIG_ORDER)
    print(header)
    print(sep)
    for ctx in CTX_ORDER:
        vals = [fmt_ttft(rates, c, ctx) for c in CONFIG_ORDER]
        bi = best_idx(vals)
        if bi >= 0:
            vals[bi] = f"**{vals[bi]}**"
        print(f"| {ctx} | " + " | ".join(vals) + " |")
    print()


def gen_table_tpot(rates):
    print("표 2. 백엔드별 TPOT (n_p, ms/token)")
    header = "| n_p | " + " | ".join(CONFIG_LABEL[c] for c in CONFIG_ORDER) + " |"
    sep = "|---:|" + "---:|" * len(CONFIG_ORDER)
    print(header)
    print(sep)
    for ctx in CTX_ORDER:
        vals = [fmt_tpot(rates, c, ctx) for c in CONFIG_ORDER]
        bi = best_idx(vals)
        if bi >= 0:
            vals[bi] = f"**{vals[bi]}**"
        print(f"| {ctx} | " + " | ".join(vals) + " |")
    print()


def gen_workload_table():
    fpath = DATA / "v10_workload_results.csv"
    if not fpath.exists():
        print("(workload validation CSV not yet generated)")
        return
    print("표 3. 워크로드별 라우터 결정과 측정된 가속비")
    print("| (n_p, n_g) | 라우터 선택 | 예측 시간(s) | 실측 시간(s) | Baseline cpu_t4(s) | Speedup |")
    print("|---|---|---:|---:|---:|---:|")
    with open(fpath) as f:
        for row in csv.DictReader(f):
            try:
                p = int(row['prompt']); g = int(row['gen'])
                cfg = row['router_cfg']; pred = float(row['predicted_s'])
                actual = float(row['router_actual_s']); cpu = float(row['cpu_t4_s'])
                sp = float(row['speedup'])
            except (ValueError, KeyError):
                continue
            mark = "**" if sp >= 1.4 else ""
            print(f"| ({p}, {g}) | {cfg} | {pred:.2f} | {actual:.2f} | {cpu:.2f} | {mark}{sp:.2f}×{mark} |")


if __name__ == '__main__':
    rates = load_rates()
    print("=== Available backends in data ===")
    for c in rates:
        print(f"  {c}: ctx ∈ {sorted(rates[c].keys())}")
    print()

    gen_table_ttft(rates)
    print()
    gen_table_tpot(rates)
    print()
    gen_workload_table()
