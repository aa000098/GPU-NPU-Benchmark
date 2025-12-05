import numpy as np
import MNN   # pip install mnn / MNN 한 거면 이 이름이 보통 맞음

mnn_model_path = "mha.mnn"

# 1) Interpreter 생성
interpreter = MNN.Interpreter(mnn_model_path)

# 2) 세션 설정 (GPU 시도, 안 되면 내부에서 CPU로 fallback 될 수 있음)
config = {
    # backend 지정 방법은 빌드 옵션/버전에 따라 조금 다르지만 보통:
    # "CPU", "OPENCL", "AUTO" 같은 문자열이나 0, 3 같은 정수 사용
    "backend": "OPENCL",   # 안 되면 "AUTO" 혹은 "CPU"로 바꿔 테스트
    "numThread": 4,        # 쓰레드 수
}
session = interpreter.createSession(config)

# 3) 입력/출력 텐서
input_tensor  = interpreter.getSessionInput(session)
output_tensor = interpreter.getSessionOutput(session)

# 4) 입력 데이터 준비 (B, T, D) = (32, 128, 512)
B, T, D = 32, 128, 512
x = np.random.randn(B, T, D).astype(np.float32)

# MNN Tensor로 옮기기
tmp = MNN.Tensor(
    input_tensor.getShape(),              # 또는 (B, T, D)
    input_tensor.getDataType(),
    x,
    input_tensor.getDimensionType()
)
input_tensor.copyFromHostTensor(tmp)

# 5) 실행
interpreter.runSession(session)

# 6) 출력 Host로 복사
host = MNN.Tensor(output_tensor, output_tensor.getDimensionType())
output_tensor.copyToHostTensor(host)
out_np = np.array(host.getData()).reshape(host.getShape())

print("output shape:", out_np.shape)
print("output sample:", out_np.reshape(-1)[:10])

