

 # python ${QNN_SDK_ROOT}/bin/aarch64-oe-linux-gcc11.2/qnn-tensorflow-converter \
 qnn-tensorflow-converter \
    --input_network "${QNN_SDK_ROOT}/examples/Models/InceptionV3/tensorflow/inception_v3_2016_08_28_frozen.pb" \
    --input_dim input 1,299,299,3 \
    --out_node "InceptionV3/Predictions/Reshape_1" \
--output_path "${QNN_SDK_ROOT}/examples/Models/InceptionV3/model/Inception_v3.cpp"
