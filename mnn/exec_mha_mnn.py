import numpy as np
import MNN

mnn_path = "mha.mnn"

# Interpreter 생성
interpreter = MNN.Interpreter(mnn_path)

# 세션 생성 (기본 backend; GPU 쓰고 싶으면 config 건드릴 수 있음)
session = interpreter.createSession()

# 입력 텐서 핸들 얻기
input_tensor = interpreter.getSessionInput(session)

# PyTorch 쪽과 동일한 shape
batch_size = 32
seq_length = 128
d_model = 512

x = np.random.randn(batch_size, seq_length, d_model).astype(np.float32)

# MNN Tensor로 래핑해서 복사
tmp = MNN.Tensor(
    (batch_size, seq_length, d_model),
    MNN.Halide_Type_Float,
    x,
    MNN.Tensor_DimensionType_Caffe  # (N, H, W, C) / (N, C, H, W)와 관련된 설정인데 여기선 단순 배치로 사용
)
input_tensor.copyFrom(tmp)

# 세션 실행
interpreter.runSession(session)

# 출력 텐서 가져오기
output_tensor = interpreter.getSessionOutput(session)
output_np = np.empty(output_tensor.getShape(), dtype=np.float32)

# Host로 복사
host_tensor = MNN.Tensor(
    output_tensor.getShape(),
    MNN.Halide_Type_Float,
    output_np,
    MNN.Tensor_DimensionType_Caffe
)
output_tensor.copyToHostTensor(host_tensor)

print("output shape:", host_tensor.getShape())
print("output sample:", output_np.flatten()[:10])

