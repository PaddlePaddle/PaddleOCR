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

### 2.1 Quick Inference

Taking table structure recognition as an example, you can quickly experience it with a single command:

```bash
paddleocr table_structure_recognition -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device npu:0 # change device to npu or xpu
```

Integrate the model inference from the table structure recognition module into your project using the following code.Please download [demo image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg) to your device first。

```python
from paddleocr import TableStructureRecognition

model = TableStructureRecognition(model_name="SLANet", device="npu:0") # change device to npu or xpu
output = model.predict("table_recognition.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

### 2.2 Model Fine-tuning

If you are not satisfied with the performance of the pre-trained model, you can fine-tune it.：

```bash
# train on  Ascend NPU 
export FLAGS_npu_storage_format=0
export FLAGS_npu_jit_compile=0
export FLAGS_use_stride_kernel=0
export FLAGS_allocator_strategy=auto_growth
export FLAGS_npu_split_aclnn=True
export FLAGS_npu_scale_aclnn=True
export CUSTOM_DEVICE_BLACK_LIST=pad3d,pad3d_grad
python3 -m paddle.distributed.launch --devices '0,1,2,3' \
        tools/train.py -c configs/table/SLANet.yml \
        -o Global.use_gpu=False Global.use_npu=True

# train on Kunlun XPU
export FLAGS_use_stride_kernel=0
export BKCL_FORCE_SYNC=1
export BKCL_TIMEOUT=1800
export XPU_BLACK_LIST=pad3d,pad3d_grad
python3 -m paddle.distributed.launch --devices '0,1,2,3' \
        tools/train.py -c configs/table/SLANet.yml \
        -o Global.use_gpu=False Global.use_xpu=True
```

For more information on using the table structure recognition module, please refer to [Table Structure Recognition Module Usage Tutorial](../module_usage/table_structure_recognition.en.md).Introductions to other modules can be found in the documents located in the same directory as the table structure recognition module.

### 2.3 Pipeline Inference

On hardware platforms such as Ascend NPU and Kunlun XPU, you can also perform pipeline inference like OCR and PP-StructureV3. The method is similar to using GPU, requiring only a change in the device.

You can quickly experience the OCR pipeline inference with a single command:

```bash
# The default model used is PP-OCRv5
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device npu:0
```

To perform quick inference using the production line in your project, you can achieve this with just a few lines of code. Here’s an example of how you might set it up:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(device="npu:0")

result = ocr.predict("./general_ocr_002.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

If you want to know more about OCR pipeline inference,please refer to [general OCR pipeline use guide](../pipeline_usage/OCR.en.md)。

In addition, the Ascend NPU supports high-performance inference with PaddleX. Both single OCR models and production lines can utilize models in OM + ONNX format for enhanced performance, offering optimized inference speed and accuracy. For detailed instructions on how to use this feature, please refer to the [Ascend NPU High-Performance Inference Tutorial](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/practical_tutorials/high_performance_npu_tutorial.en.md)。
