import pyopencl as cl
import numpy as np
import time
import os

from opencl.util import get_opencl_context 

HERE = os.path.dirname(os.path.abspath(__file__))
MATMUL_CL_PATH = os.path.join(HERE, "softmax.cl")

def softmax_opencl(L, D):
    ctx, queue = get_opencl_context()

    prog_src = open(MATMUL_CL_PATH).read()
    prog = cl.Program(ctx, prog_src).build()

    X = np.random.rand(L, D).astype(np.float16)
    Y = np.zeros((L, D), dtype=np.float16)

    dX = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=X)
    dY = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, Y.nbytes)

    kernel = prog.softmax_fp16
    global_size = (L,)

    # warmup
    for _ in range(5):
        kernel(queue, global_size, None, dX, dY, np.int32(D))

    # 측정
    start = time.time()
    for _ in range(30):
        kernel(queue, global_size, None, dX, dY, np.int32(D))
    queue.finish()
    end = time.time()

    latency_ms = (end - start) * 1000 / 30
    return latency_ms

