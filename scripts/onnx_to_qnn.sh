


qnn-onnx-converter   --input_network ../pytorch/mha.onnx   --output_path ./models/mha_qnn   --backend gpu   --op_package_config "${QNN_SDK_ROOT}/examples/OpPackage/qnn-op-pkgs.json"
