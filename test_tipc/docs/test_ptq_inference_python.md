# Linux GPU/CPU KL离线量化训练推理测试

Linux GPU/CPU KL离线量化训练推理测试的主程序为`test_ptq_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总
- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 |
|  :----: |   :----:  |    :----:  |
|    | model_name | KL离线量化训练 |

- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|     |  model_name |  支持 | 支持 | 1 |

## 2. 测试流程

### 2.1 准备数据和模型

先运行`prepare.sh`准备数据和模型，然后运行`test_ptq_inference_python.sh`进行测试，最终在```test_tipc/output/{model_name}/whole_infer```目录下生成`python_infer_*.log`后缀的日志文件。

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/ch_PP-OCRv2_det/train_ptq_infer_python.txt "whole_infer"

# 用法:
bash test_tipc/test_ptq_inference_python.sh ./test_tipc/configs/ch_PP-OCRv2_det/train_ptq_infer_python.txt "whole_infer"
```

#### 运行结果

各测试的运行情况会打印在 `test_tipc/output/{model_name}/paddle2onnx/results_paddle2onnx.log` 中：
运行成功时会输出：

```
Run successfully with command - ch_PP-OCRv2_det_KL - python3.7 deploy/slim/quantization/quant_kl.py -c configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml -o Global.pretrained_model=./inference/ch_PP-OCRv2_det_infer/ Global.save_inference_dir=./inference/ch_PP-OCRv2_det_infer/_klquant > ./test_tipc/output/ch_PP-OCRv2_det_KL/whole_infer/whole_infer_export_0.log 2>&1 !
Run successfully with command - ch_PP-OCRv2_det_KL - python3.7 tools/infer/predict_det.py --use_gpu=False --enable_mkldnn=False --cpu_threads=6 --det_model_dir=./inference/ch_PP-OCRv2_det_infer/_klquant --rec_batch_num=1   --image_dir=./inference/ch_det_data_50/all-sum-510/   --precision=int8   > ./test_tipc/output/ch_PP-OCRv2_det_KL/whole_infer/python_infer_cpu_usemkldnn_False_threads_6_precision_int8_batchsize_1.log 2>&1 !
Run successfully with command - ch_PP-OCRv2_det_KL - python3.7 tools/infer/predict_det.py --use_gpu=True --use_tensorrt=False --precision=int8 --det_model_dir=./inference/ch_PP-OCRv2_det_infer/_klquant --rec_batch_num=1 --image_dir=./inference/ch_det_data_50/all-sum-510/       > ./test_tipc/output/ch_PP-OCRv2_det_KL/whole_infer/python_infer_gpu_usetrt_False_precision_int8_batchsize_1.log 2>&1 !
```

运行失败时会输出：

```
Run failed with command - ch_PP-OCRv2_det_KL - python3.7 deploy/slim/quantization/quant_kl.py -c configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_cml.yml -o Global.pretrained_model=./inference/ch_PP-OCRv2_det_infer/ Global.save_inference_dir=./inference/ch_PP-OCRv2_det_infer/_klquant > ./test_tipc/output/ch_PP-OCRv2_det_KL/whole_infer/whole_infer_export_0.log 2>&1 !
...
```

## 3. 更多教程

本文档为功能测试用，更详细的量化使用教程请参考：[量化](../../deploy/slim/quantization/README.md)
