---
comments: true
---

# PP-OCR模型量化

复杂的模型有利于提高模型的性能，但也导致模型中存在一定冗余，模型量化将全精度缩减到定点数减少这种冗余，达到减少模型计算复杂度，提高模型推理性能的目的。
模型量化可以在基本不损失模型的精度的情况下，将FP32精度的模型参数转换为Int8精度，减小模型参数大小并加速计算，使用量化后的模型在移动端等部署时更具备速度优势。

本教程将介绍如何使用飞桨模型压缩库PaddleSlim做PaddleOCR模型的压缩。
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 集成了模型剪枝、量化（包括量化训练和离线量化）、蒸馏和神经网络搜索等多种业界常用且领先的模型压缩功能，如果您感兴趣，可以关注并了解。

在开始本教程之前，建议先了解[PaddleOCR模型的训练方法](../model_train/training.md)以及[PaddleSlim](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)

## 快速开始

量化多适用于轻量模型在移动端的部署，当训练出一个模型后，如果希望进一步的压缩模型大小并加速预测，可使用量化的方法压缩模型。

模型量化主要包括五个步骤：

1. 安装 PaddleSlim
2. 准备训练好的模型
3. 量化训练
4. 导出量化推理模型
5. 量化模型预测部署

### 1. 安装PaddleSlim

```bash linenums="1"
pip3 install paddleslim==2.3.2
```

### 2. 准备训练好的模型

PaddleOCR提供了一系列训练好的[模型](../model_list.md)，如果待量化的模型不在列表中，需要按照[常规训练](../quick_start.md)方法得到训练好的模型。

### 3. 量化训练

量化训练包括离线量化训练和在线量化训练，在线量化训练效果更好，需加载预训练模型，在定义好量化策略后即可对模型进行量化。

量化训练的代码位于slim/quantization/quant.py 中，比如训练检测模型，以PPOCRv3检测模型为例，训练指令如下：

```bash linenums="1"
# 下载检测预训练模型：
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar xf ch_PP-OCRv3_det_distill_train.tar

python deploy/slim/quantization/quant.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml -o Global.pretrained_model='./ch_PP-OCRv3_det_distill_train/best_accuracy'   Global.save_model_dir=./output/quant_model_distill/
```

如果要训练识别模型的量化，修改配置文件和加载的模型参数即可。

### 4. 导出模型

在得到量化训练保存的模型后，我们可以将其导出为inference_model，用于预测部署：

```bash linenums="1"
python deploy/slim/quantization/export_model.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml -o Global.checkpoints=output/quant_model/best_accuracy Global.save_inference_dir=./output/quant_inference_model
```

### 5. 量化模型部署

上述步骤导出的量化模型，参数精度仍然是FP32，但是参数的数值范围是int8，导出的模型可以通过PaddleLite的opt模型转换工具完成模型转换。

量化模型移动端部署的可参考 [移动端模型部署](../infer_deploy/lite.md)

备注：量化训练后的模型参数是float32类型，转inference model预测时相对不量化无加速效果，原因是量化后模型结构之间存在量化和反量化算子，如果要使用量化模型部署，建议使用TensorRT并设置precision为INT8加速量化模型的预测时间。
