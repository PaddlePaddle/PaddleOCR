#!/bin/bash

# 本脚本用于测试PPOCRV4_det系列模型的自动压缩功能
## 运行脚本前，请确保处于以下环境：
## CUDA11.7+TensorRT8.4.2.4+Paddle2.5.2

model_type="$1"

if [ "$model_type" = "mobile" ]; then
    echo "test ppocrv4_det_mobile model......"
    ## 启动自动化压缩训练
    CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/det_mobile_qat --config_path configs/ppocrv4/ppocrv4_det_qat_dist.yaml

    ## GPU指标测试
    ### 量化前，预期指标：hmean:72.71%;time:4.7ms
    python test_ocr.py --model_path ./models/ch_PP-OCRv4_det_infer --config ./configs/ppocrv4/ppocrv4_det_qat_dist.yaml --precision fp32 --use_trt True
    ### 量化后，预期指标：hmean:71.38%;time:3.3ms
    python test_ocr.py --model_path ./models/det_mobile_qat --config ./configs/ppocrv4/ppocrv4_det_qat_dist.yaml --precision int8 --use_trt True

    ## CPU指标测试
    ### 量化前，预期指标：hmean:72.71%;time:198.4ms
    python test_ocr.py --model_path ./models/ch_PP-OCRv4_det_infer --config ./configs/ppocrv4/ppocrv4_det_qat_dist.yaml --precision fp32 --use_mkldnn True --device CPU --cpu_threads 12
    ### 量化后，预期指标：hmean:72.30%;time:205.2ms
    python test_ocr.py --model_path ./models/det_mobile_qat --config ./configs/ppocrv4/ppocrv4_det_qat_dist.yaml --precision int8 --use_mkldnn True --device CPU --cpu_threads 12

    # 量化前模型推理
    # GPU
    python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/models/ch_PP-OCRv4_det_infer \
        --benchmark True --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset --use_gpu True \
        --use_tensorrt True --warmup True --precision fp32

    # CPU
    python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/models/ch_PP-OCRv4_det_infer \
        --benchmark True --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset --use_gpu False \
        --enable_mkldnn True --warmup True --precision fp32

    # 量化后模型推理
    # GPU
    python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/models/det_mobile_qat \
        --benchmark True --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset --use_gpu True \
        --use_tensorrt True --warmup True --precision int8

    # CPU
    python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/models/det_mobile_qat \
        --benchmark True --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset --use_gpu False \
        --enable_mkldnn True --warmup True --precision int8

elif [ "$model_type" = "server" ]; then
    echo "test ppocrv4_det_server model......"
    ## 启动自动化压缩训练
    CUDA_VISIBLE_DEVICES=0 python run.py --save_dir ./models/det_server_qat --config_path configs/ppocrv4/ppocrv4_det_server_qat_dist.yaml

    ## GPU指标测试
    ### 量化前，预期指标：hmean:79.77%;time:50.0ms
    python test_ocr.py --model_path ./models/ch_PP-OCRv4_det_server_infer --config ./configs/ppocrv4/ppocrv4_det_server_qat_dist.yaml --precision fp32 --use_trt True
    ### 量化后，预期指标：hmean:79.81%;time:42.4ms
    python test_ocr.py --model_path ./models/det_server_qat --config ./configs/ppocrv4/ppocrv4_det_server_qat_dist.yaml --precision int8 --use_trt True

    ## CPU指标测试
    ### 量化前，预期指标：hmean:79.77%;time:2159.4ms
    python test_ocr.py --model_path ./models/ch_PP-OCRv4_det_server_infer --config ./configs/ppocrv4/ppocrv4_det_server_qat_dist.yaml --precision fp32 --use_mkldnn True --device CPU --cpu_threads 12
    ### 量化后，预期指标：hmean:79.69%;time:1834.8ms
    python test_ocr.py --model_path ./models/det_server_qat --config ./configs/ppocrv4/ppocrv4_det_server_qat_dist.yaml --precision int8 --use_mkldnn True --device CPU --cpu_threads 12

    ## 量化前模型推理
    ### GPU
    python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/models/ch_PP-OCRv4_det_server_infer \
        --benchmark True --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset --use_gpu True \
        --use_tensorrt True --warmup True --precision fp32

    ### CPU
    python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/models/ch_PP-OCRv4_det_server_infer \
        --benchmark True --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset --use_gpu False \
        --enable_mkldnn True --warmup True --precision fp32

    ## 量化后模型推理
    ### GPU
    python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/models/det_server_qat \
        --benchmark True --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset --use_gpu True \
        --use_tensorrt True --warmup True --precision int8

    ### CPU
    python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/models/det_server_qat \
        --benchmark True --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset --use_gpu False \
        --enable_mkldnn True --warmup True --precision int8
else
    echo "unrecgnized model_type"
fi
