
python run.py --save_dir save_quant_ppocrv4_rec --config_path configs/ppocrv4/ppocrv4_rec_qat_dist.yaml


## 检测模型推理
# 量化前模型
python test_ocr.py  --model_path ch_PP-OCRv4_det_infer --config_path configs/ppocrv4/ppocrv4_det_qat_dist.yaml\
 --device GPU  --use_trt True  --precision fp32

python test_ocr.py  --model_path ch_PP-OCRv4_det_infer --config_path configs/ppocrv4/ppocrv4_det_qat_dist.yaml --device CPU  --use_mkldnn True  --precision fp32 --cpu_threads 10

# 量化模型
python test_ocr.py  --model_path save_quant_ppocrv4_det --config_path configs/ppocrv4/ppocrv4_rec_qat_dist.yaml --device GPU  --use_trt True  --precision int8

python test_ocr.py  --model_path save_quant_ppocrv4_det  --config_path configs/ppocrv4/ppocrv4_det_qat_dist.yaml --device CPU  --use_mkldnn True  --precision int8 --cpu_threads 10



# 检测量化模型推理
# GPU
python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/save_quant_ppocrv4_det \
     --benchmark True  --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset  --use_gpu True \
     --use_tensorrt True  --warmup True --precision int8

# CPU
python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/save_quant_ppocrv4_det \
     --benchmark True  --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset  --use_gpu False \
     --enable_mkldnn True  --warmup True --precision int8

# 量化前模型
# GPU
python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/ch_PP-OCRv4_det_infer \
     --benchmark True  --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset  --use_gpu True \
     --use_tensorrt True  --warmup True --precision fp32

E0920 14:07:27.288496 35875 helper.h:131] Assertion failed: tensor.region->getDimensions(true) == tensor.extent
../builder/cudnnBuilderWeightConverters.cpp:645
Aborting...
E0920 14:07:27.297259 35875 helper.h:131] ../builder/cudnnBuilderWeightConverters.cpp (645) - Assertion Error in assertRegionTightlyFitsTensor: 0 (tensor.region->getDimensions(true) == tensor.extent)
Traceback (most recent call last):
  File "tools/infer/predict_det.py", line 291, in <module>
    text_detector = TextDetector(args)
  File "tools/infer/predict_det.py", line 143, in __init__
    args, 'det', logger)
  File "/ssd2/tangshiyu/Code/PaddleOCR/tools/infer/utility.py", line 281, in create_predictor
    predictor = inference.create_predictor(config)
SystemError: (Fatal) Build TensorRT cuda engine failed! Please recheck you configurations related to paddle-TensorRT.
  [Hint: infer_engine_ should not be null.] (at ../paddle/fluid/inference/tensorrt/engine.cc:382)

# CPU
python tools/infer/predict_det.py --det_model_dir deploy/slim/auto_compression/ch_PP-OCRv4_det_infer \
     --benchmark True  --image_dir deploy/slim/auto_compression/datasets/v4_4_test_dataset  --use_gpu False \
     --enable_mkldnn True  --warmup True --precision fp32


# 报错
python test_ocr.py  --model_path save_quant_ppocrv4_det --config_path configs/ppocrv4/ppocrv4_det_qat_dist.yaml \
        --device GPU  --use_trt True  --precision int8

# 识别模型推理
python test_ocr.py  --model_path ch_PP-OCRv4_rec_infer --config_path configs/ppocrv4/ppocrv4_rec_qat_dist.yaml --device GPU  --use_trt True  --precision fp32


python tools/infer/predict_rec.py --rec_model_dir deploy/slim/auto_compression/ch_PP-OCRv4_rec_infer  --benchmark True  --image_dir deploy/slim/auto_compression/data_pubtabnet/real_data/lsvt/lsvt_val_images/  --use_gpu True --use_tensorrt True  --warmup True
