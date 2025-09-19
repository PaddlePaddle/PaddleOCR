---
comments: true
---

# PaddleOCR多硬件使用指南

本文档主要针对昇腾 NPU、昆仑 XPU 等硬件平台，介绍 PaddleOCR 使用指南。

## 1、安装

### 1.1 PaddlePaddle安装

首先请您根据所属硬件平台，完成飞桨 PaddlePaddle 的安装，各硬件的飞桨安装教程如下：

昇腾 NPU：[昇腾 NPU 飞桨安装教程](./paddlepaddle_install_NPU.md)

昆仑 XPU：[昆仑 XPU 飞桨安装教程](./paddlepaddle_install_XPU.md)

### 1.2 PaddleOCR安装

请参考[PaddleOCR安装教程](../installation.md)安装 PaddleOCR。

## 2、使用

基于昇腾 NPU、昆仑 XPU 等硬件平台的 PaddleOCR 训练、推理使用方法与 GPU 相同，只需根据所属硬件平台，修改配置设备的参数。在这两款硬件上，支持 PaddleOCR 三大特色能力的快速推理和模型微调，包括 文字识别模型 PP-OCRv5、文档解析方案 PP-StructureV3 和 PP-ChatOCRv4。

### 2.1 快速推理

使用一行命令即可快速体验：

* OCR 产线推理

```bash
# 默认使用 PP-OCRv5 模型
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device npu:0 # 将设备名修改为 npu或xpu
```

* PP-StructureV3 产线推理

```bash
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --device npu:0
```

在项目中，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

* OCR 产线推理
```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(device="npu:0")

result = ocr.predict("./general_ocr_002.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

* PP-StructureV3 产线推理
```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(device="npu:0")
output = pipeline.predict("./pp_structure_v3_demo.png")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output") ## 保存当前图像的markdown格式的结果
```

更多关于 OCR 产线推理的使用说明，请参考[通用OCR产线使用教程](../pipeline_usage/OCR.md)。
更多关于 PP-StructureV3 产线推理的使用说明，请参考[PP-StructureV3产线使用教程](../pipeline_usage/PP-StructureV3.md)。

### 2.2 模型微调

如果对预训练模型的效果不满意，可以对产线上的模型进行模型微调，示例如下：

* 昇腾 NPU
```bash
export FLAGS_npu_storage_format=0
export FLAGS_npu_jit_compile=0
export FLAGS_use_stride_kernel=0
export FLAGS_allocator_strategy=auto_growth
export FLAGS_npu_split_aclnn=True
export FLAGS_npu_scale_aclnn=True
export CUSTOM_DEVICE_BLACK_LIST=pad3d,pad3d_grad
python3 -m paddle.distributed.launch --devices '0,1,2,3' \
        tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
        -o Global.use_gpu=False Global.use_npu=True
```

* 昆仑 XPU
```bash
export FLAGS_use_stride_kernel=0
export BKCL_FORCE_SYNC=1
export BKCL_TIMEOUT=1800
export XPU_BLACK_LIST=pad3d,pad3d_grad
python3 -m paddle.distributed.launch --devices '0,1,2,3' \
        tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
        -o Global.use_gpu=False Global.use_xpu=True
```

### 2.3 其它方式推理

在昇腾 NPU 上，对于少量推理样本，使用上述产线推理方式存在结果异常的可能（主要是 PP-StructureV3 产线），针对这种情况，我们支持使用 ONNX 模型进行推理，保证推理结果正确。

使用如下命令可以将 Paddle 模型转换为 ONNX 模型：

```bash
paddlex --install paddle2onnx
paddlex --paddle2onnx --paddle_model_dir /paddle_model_dir --onnx_model_dir /onnx_model_dir --opset_version 7
```

同时，部分模型支持昇腾离线 OM 推理，有效优化推理性能和内存占用。使用 OM + ONNX 格式的模型进行产线推理，能够同时保障精度和速度。

使用 atc 转换工具将 ONNX 模型转换为 OM 模型：

```bash
atc --model=inference.onnx --framework=5 --output=inference --soc_version="your_device_type" --input_shape "your_input_shape"
```

我们已将 ONNX 和 OM 模型深度集成进了 PaddleX 高性能推理，修改产线配置文件，配置模型推理后端为 ONNX 或 OM 即可使用 PaddleX 高性能推理 API 进行推理。

具体修改方法、推理代码以及更多的使用方法请参考[昇腾 NPU 高性能推理教程](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/practical_tutorials/high_performance_npu_tutorial.md)。

## 3、常见问题
### 1. 使用 PP-StructureV3 产线推理结果不正确
该产线上的部分模型在少量 case 上存在精度误差，可以尝试调整配置文件中的模型，或者使用 ONNX+OM 模型进行推理。
