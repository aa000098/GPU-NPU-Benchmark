#!/bin/bash
# CPU(OpenBLAS 4T NEON) vs GPU(Mali-G610 OpenCL) 공정 비교
# 같은 행렬 크기, warmup 10회, 측정 10회 median
set -e
cd /home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench

OUTPUT="./results/raw/gpu_analysis"
mkdir -p "$OUTPUT"

echo "=========================================================================="
echo "CPU(OpenBLAS 4T NEON) vs GPU(Mali-G610 OpenCL) GEMM — FP32, M=N=K"
echo "Warmup: 10 | Measure: 10 (median)"
echo "=========================================================================="

echo ""
echo "[1/2] CPU (OpenBLAS, 4 threads, NEON)..."
OPENBLAS_NUM_THREADS=4 python3 -c "
import numpy as np, time, os, json
os.environ['OPENBLAS_NUM_THREADS'] = '4'

sizes = [32, 64, 128, 256, 512, 1024, 2048]
results = {}
for S in sizes:
    A = np.random.randn(S, S).astype(np.float32)
    B = np.random.randn(S, S).astype(np.float32)
    for _ in range(10): np.matmul(A, B)
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        np.matmul(A, B)
        times.append((time.perf_counter() - t0) * 1000)
    ms = float(np.median(times))
    gf = (2.0*S*S*S) / (ms/1000) / 1e9
    results[S] = {'ms': ms, 'gflops': gf}
    print('  %4d x %4d : %8.3f ms  (%6.2f GFLOPS)' % (S, S, ms, gf))

with open('$OUTPUT/cpu_gemm.json', 'w') as f:
    json.dump(results, f)
"

echo ""
echo "[2/2] GPU (OpenCL, Mali-G610)..."
./gpu_gemm_bench 2>&1 | tee "$OUTPUT/gpu_gemm_raw.txt"

echo ""
echo "=========================================================================="
echo "비교 결과:"
echo "=========================================================================="

python3 -c "
import json

with open('$OUTPUT/cpu_gemm.json') as f:
    cpu = json.load(f)

# Parse GPU results
gpu = {}
with open('$OUTPUT/gpu_gemm_raw.txt') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('=') or line.startswith('-') or line.startswith('Size') \
           or line.startswith('CPU') or line.startswith('Open') or line.startswith('arm'):
            continue
        parts = line.split('|')
        if len(parts) >= 4:
            try:
                size = int(parts[0].strip())
                gpu_part = parts[2].strip().split()
                gpu[str(size)] = {'ms': float(gpu_part[0]), 'gflops': float(gpu_part[1])}
            except (ValueError, IndexError):
                continue

print('%-6s | %10s %10s | %10s %10s | %s' % ('Size', 'CPU(ms)', 'GFLOPS', 'GPU(ms)', 'GFLOPS', 'Winner'))
print('-' * 75)

crossover = None
for S in [32, 64, 128, 256, 512, 1024, 2048]:
    c = cpu.get(str(S))
    g = gpu.get(str(S))
    if c and g:
        if g['ms'] < c['ms']:
            winner = '<< GPU'
            ratio = c['ms'] / g['ms']
            if crossover is None:
                crossover = S
        else:
            winner = 'CPU >>'
            ratio = g['ms'] / c['ms']
        print('%-6d | %8.3f %8.2f  | %8.3f %8.2f  | %s (%.1fx)' % (
            S, c['ms'], c['gflops'], g['ms'], g['gflops'], winner, ratio))

print('=' * 75)
if crossover:
    print('GPU 역전 교차점: M=N=K=%d' % crossover)
else:
    print('테스트 범위에서 GPU가 이기는 구간 없음')
print()
print('참고: LLM decode는 M=1 (행렬x벡터), prefill은 M=토큰수')
print('      MNN CPU는 OpenBLAS와 유사한 NEON 최적화 사용')
"
