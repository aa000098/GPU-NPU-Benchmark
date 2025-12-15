import pyopencl as cl
import numpy as np
import time
import os

from opencl.util import get_opencl_context 

HERE = os.path.dirname(os.path.abspath(__file__))
MATMUL_CL_PATH = os.path.join(HERE, "matmul.cl")

TILE_M = 16
TILE_N = 16

def matmul_opencl(M, N, K):
    ctx, queue = get_opencl_context()

    with open(MATMUL_CL_PATH, "r") as f:
        prog_src = f.read()

    # 빌드 옵션도 한 번 같이 줘보자 (빠른 math)
    prog = cl.Program(ctx, prog_src).build(
        options=["-cl-fast-relaxed-math", "-cl-mad-enable"]
    )

    # 데이터 준비 (row-major)
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    C = np.zeros((M, N), dtype=np.float16)

    mf = cl.mem_flags
    dA = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    dB = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    dC = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

    kernel = prog.matmul_fp16

    # 타일 기준으로 global/local 사이즈 맞추기
    global_M = ((M + TILE_M - 1) // TILE_M) * TILE_M
    global_N = ((N + TILE_N - 1) // TILE_N) * TILE_N

    global_size = (global_M, global_N)
    local_size  = (TILE_M, TILE_N)

    # warm-up
    for _ in range(5):
        kernel(queue, global_size, local_size,
               np.int32(M), np.int32(N), np.int32(K),
               dA, np.int32(K),
               dB, np.int32(N),
               dC, np.int32(N))
    queue.finish()

    # 측정
    start = time.time()
    for _ in range(50):
        kernel(queue, global_size, local_size,
               np.int32(M), np.int32(N), np.int32(K),
               dA, np.int32(K),
               dB, np.int32(N),
               dC, np.int32(N))
    queue.finish()
    end = time.time()

    latency_ms = (end - start) * 1000 / 50
    return latency_ms

