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

基于昇腾 NPU、昆仑 XPU 等硬件平台的 PaddleOCR 训练、推理使用方法与 GPU 相同，只需根据所属硬件平台，修改配置设备的参数。在这两款硬件上，支持 PaddleOCR 三大特色能力的快速推理和模型微调，包括 文字识别模型 PP-OCRv5、文档解析方案 PP-StructureV3 和 PP-ChatOCRv4。下面以 PP-OCRv5 模型为例，介绍如何使用。

### 2.1 快速推理

使用一行命令即可快速体验：

```bash
# 默认使用 PP-OCRv5 模型
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device npu:0 # 将设备名修改为 npu或xpu
```

在项目中，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(device="npu:0")

result = ocr.predict("./general_ocr_002.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

更多关于 OCR 产线推理的使用说明，请参考[通用OCR产线使用教程](../pipeline_usage/OCR.md)。

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

更多关于表格结构识别模块的使用说明，请参考[表格结构识别模块使用教程](../module_usage/table_structure_recognition.md),其它模块的介绍可参考表格结构识别同级目录下的其它文档。

### 2.3 其它方式推理

在昇腾 NPU 上，对于少量推理样本，使用上述产线推理方式存在结果异常的可能（主要是 PP-StructureV3 产线），针对这种情况，我们支持使用 ONNX 模型进行推理，保证推理结果正确。同时，部分模型支持昇腾离线 OM 推理，有效优化推理性能和内存占用。使用 OM + ONNX 格式的模型进行产线推理，能够同时保障精度和速度。我们已将 ONNX 和 OM 模型深度集成进了 PaddleX 高性能推理，详细使用方法参考[昇腾 NPU 高性能推理教程](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/practical_tutorials/high_performance_npu_tutorial.md)。
