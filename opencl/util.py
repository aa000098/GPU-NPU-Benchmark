import pyopencl as cl

def get_opencl_context():
    platforms = cl.get_platforms()

    # Rockchip RK3588 → 대부분 "ARM Platform"만 존재함
    platform = None
    for p in platforms:
        if "ARM" in p.name:
            platform = p
            break

    if platform is None:
        platform = platforms[0]

    # GPU 디바이스 선택 (Mali-G610)
    devices = platform.get_devices(device_type=cl.device_type.GPU)
    if len(devices) == 0:
        devices = platform.get_devices(device_type=cl.device_type.ALL)

    ctx = cl.Context([devices[0]])
    queue = cl.CommandQueue(ctx)

    return ctx, queue
