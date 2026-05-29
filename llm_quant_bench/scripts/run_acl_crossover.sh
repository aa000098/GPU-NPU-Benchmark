#!/bin/bash
# CPU(ACL NEON) vs GPU(ACL OpenCL) vs NPU GEMM 공정 비교
# ACL은 Mali GPU에 최적화된 GEMM 커널을 사용
set -e

ACL="/home/hyunho.son/install_files/ComputeLibrary/build"
export LD_LIBRARY_PATH="$ACL:$LD_LIBRARY_PATH"

OUTPUT="/home/hyunho.son/Desktop/project/GPU-NPU-Benchmark/llm_quant_bench/results/raw/gpu_analysis"
mkdir -p "$OUTPUT"

SIZES="32 64 128 256 512 1024 2048"

echo "=========================================================================="
echo "CPU(ACL NEON) vs GPU(ACL OpenCL) GEMM Crossover — FP32"
echo "ARM Compute Library 최적화 커널 사용"
echo "=========================================================================="

echo ""
echo "[1/2] GPU (ACL cl_sgemm — Mali-G610 최적화 GEMM)..."
echo "Size,GPU_ms" > "$OUTPUT/acl_gpu.csv"
for S in $SIZES; do
    echo -n "  ${S}x${S}: "
    # cl_sgemm M N K 형식. 시간 측정을 위해 time 사용
    start=$(date +%s%N)
    $ACL/examples/cl_sgemm $S $S $S 2>&1 > /dev/null
    end=$(date +%s%N)
    ms=$(echo "scale=3; ($end - $start) / 1000000" | bc)
    gflops=$(echo "scale=2; 2 * $S * $S * $S / ($ms / 1000) / 1000000000" | bc 2>/dev/null || echo "N/A")
    echo "${ms} ms (${gflops} GFLOPS)"
    echo "$S,$ms" >> "$OUTPUT/acl_gpu.csv"
done

echo ""
echo "[2/2] CPU (ACL neon_sgemm — NEON SIMD 최적화 GEMM)..."
echo "Size,CPU_ms" > "$OUTPUT/acl_cpu.csv"
for S in $SIZES; do
    echo -n "  ${S}x${S}: "
    start=$(date +%s%N)
    $ACL/examples/neon_sgemm --m=$S --n=$S --k=$S 2>&1 > /dev/null
    end=$(date +%s%N)
    ms=$(echo "scale=3; ($end - $start) / 1000000" | bc)
    gflops=$(echo "scale=2; 2 * $S * $S * $S / ($ms / 1000) / 1000000000" | bc 2>/dev/null || echo "N/A")
    echo "${ms} ms (${gflops} GFLOPS)"
    echo "$S,$ms" >> "$OUTPUT/acl_cpu.csv"
done

echo ""
echo "=========================================================================="
echo "비교 결과:"
echo "=========================================================================="

python3 -c "
import csv

cpu, gpu = {}, {}
with open('$OUTPUT/acl_cpu.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cpu[int(row['Size'])] = float(row['CPU_ms'])
with open('$OUTPUT/acl_gpu.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gpu[int(row['Size'])] = float(row['GPU_ms'])

print('%-6s | %12s | %12s | %s' % ('Size', 'CPU NEON(ms)', 'GPU CL(ms)', 'Winner'))
print('-' * 60)
for S in sorted(cpu.keys()):
    c, g = cpu.get(S), gpu.get(S)
    if c and g:
        if g < c:
            winner = '<< GPU (%.1fx)' % (c/g)
        else:
            winner = 'CPU >> (%.1fx)' % (g/c)
        ops = 2.0 * S * S * S
        c_gf = ops / (c/1000) / 1e9
        g_gf = ops / (g/1000) / 1e9
        print('%-6d | %8.1f ms  | %8.1f ms  | %s' % (S, c, g, winner))
print('=' * 60)
"

echo ""
echo "=== Done: $(date) ==="
