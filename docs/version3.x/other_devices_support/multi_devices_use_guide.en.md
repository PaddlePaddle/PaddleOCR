---
comments: true
---

# PaddleOCR Multi-Devices Usage Guide

This document focuses on the usage guide of PaddleX for Huawei Ascend NPU and Kunlun XPU hardware platforms.

## 1、Installation

### 1.1 PaddlePaddle Installation

First, please complete the installation of PaddlePaddle according to your hardware platform. The installation tutorials for each hardware are as follows:

Ascend NPU: [Ascend NPU PaddlePaddle Installation Guide](./paddlepaddle_install_NPU.en.md)

Kunlun XPU: [Kunlun XPU PaddlePaddle Installation Guide](./paddlepaddle_install_XPU.en.md)

### 1.2 PaddleOCR Installation

Please refer to [PaddleOCR Installation Guide](../installation.en.md) to install PaddleOCR。

## 2、Usage

The methods for training and inference of PaddleOCR on hardware platforms such as Ascend NPU and Kunlun XPU are the same as those on GPU. You only need to modify the configuration parameters according to the specific hardware platform.
On these two hardware platforms, quick inference and model fine-tuning are supported for the three major features of PaddleOCR, including the text recognition model PP-OCRv5, the document parsing solution PP-StructureV3, and PP-ChatOCRv4.

### 2.1 Quick Inference

You can quickly experience the OCR pipeline inference with a single command:

* OCR pipeline inference

```bash
# The default model used is PP-OCRv5
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device npu:0
```

* PP-StructureV3 pipeline inference

```bash
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --device npu:0
```

To perform quick inference using the production line in your project, you can achieve this with just a few lines of code. Here’s an example of how you might set it up:

* OCR pipeline inference

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(device="npu:0")

result = ocr.predict("./general_ocr_002.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

* PP-StructureV3 pipeline inference

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(device="npu:0")
output = pipeline.predict("./pp_structure_v3_demo.png")
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output")
```

If you want to know more about OCR pipeline inference,please refer to [general OCR pipeline use guide](../pipeline_usage/OCR.en.md).
If you want to know more about PPStructureV3 pipeline inference,please refer to [PP-StructureV3 pipeline use guide](../pipeline_usage/PP-StructureV3.en.md).

### 2.2 Model Fine-tuning

If you are not satisfied with the performance of the pre-trained model, you can fine-tune it.

* train on Ascend NPU
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

* train on Kunlun XPU
```bash
export FLAGS_use_stride_kernel=0
export BKCL_FORCE_SYNC=1
export BKCL_TIMEOUT=1800
export XPU_BLACK_LIST=pad3d,pad3d_grad
python3 -m paddle.distributed.launch --devices '0,1,2,3' \
        tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
        -o Global.use_gpu=False Global.use_xpu=True
```

### 2.3 Other Inference methods

On the Ascend NPU, for a small number of inference samples, using the aforementioned pipeline inference method may result in abnormal outcomes (primarily with the PP-StructureV3 pipeline). To address this issue, we support using ONNX models for inference to ensure correct results.

You can use the following command to convert a Paddle model to an ONNX model:

```bash
paddlex --install paddle2onnx
paddlex --paddle2onnx --paddle_model_dir /paddle_model_dir --onnx_model_dir /onnx_model_dir --opset_version 7
```

Meanwhile, some models support Ascend offline OM inference, effectively optimizing inference performance and memory usage. Using models in OM + ONNX format for pipeline inference can ensure both accuracy and speed.

Use the ATC conversion tool to convert an ONNX model to an OM model:

```bash
atc --model=inference.onnx --framework=5 --output=inference --soc_version="your_device_type" --input_shape "your_input_shape"
```

We have deeply integrated ONNX and OM models into PaddleX for high-performance inference. By modifying the pipeline configuration file to set the model inference backend to ONNX or OM, you can use the PaddleX high-performance inference API for inference.

For specific modification methods, inference code, and more usage instructions, please refer to [Ascend NPU High-Performance Inference Tutorial](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/practical_tutorials/high_performance_npu_tutorial.en.md)。

## 3、 FAQ
### 1.The inference results using PP-StructureV3 on the production line are incorrect. 
Some models on this line have precision errors in a small number of cases. You can try adjusting the model in the configuration file or use ONNX+OM models for inference.
