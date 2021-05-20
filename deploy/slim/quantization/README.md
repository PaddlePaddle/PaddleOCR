
## 介绍
复杂的模型有利于提高模型的性能，但也导致模型中存在一定冗余，模型量化将全精度缩减到定点数减少这种冗余，达到减少模型计算复杂度，提高模型推理性能的目的。
模型量化可以在基本不损失模型的精度的情况下，将FP32精度的模型参数转换为Int8精度，减小模型参数大小并加速计算，使用量化后的模型在移动端等部署时更具备速度优势。

本教程将介绍如何使用飞桨模型压缩库PaddleSlim做PaddleOCR模型的压缩。
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) 集成了模型剪枝、量化（包括量化训练和离线量化）、蒸馏和神经网络搜索等多种业界常用且领先的模型压缩功能，如果您感兴趣，可以关注并了解。

在开始本教程之前，建议先了解[PaddleOCR模型的训练方法](../../../doc/doc_ch/quickstart.md)以及[PaddleSlim](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)


## 快速开始
量化多适用于轻量模型在移动端的部署，当训练出一个模型后，如果希望进一步的压缩模型大小并加速预测，可使用量化的方法压缩模型。

模型量化主要包括五个步骤：
1. 安装 PaddleSlim
2. 准备训练好的模型
3. 量化训练
4. 导出量化推理模型
5. 量化模型预测部署

### 1. 安装PaddleSlim

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python setup.py install
```

### 2. 准备训练好的模型

PaddleOCR提供了一系列训练好的[模型](../../../doc/doc_ch/models_list.md)，如果待量化的模型不在列表中，需要按照[常规训练](../../../doc/doc_ch/quickstart.md)方法得到训练好的模型。

### 3. 量化训练
量化训练包括离线量化训练和在线量化训练，在线量化训练效果更好，需加载预训练模型，在定义好量化策略后即可对模型进行量化。


量化训练的代码位于slim/quantization/quant.py 中，比如训练检测模型，训练指令如下：
```bash
python deploy/slim/quantization/quant.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model='your trained model'   Global.save_model_dir=./output/quant_model

# 比如下载提供的训练模型
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
tar -xf ch_ppocr_mobile_v2.0_det_train.tar
python deploy/slim/quantization/quant.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model=./ch_ppocr_mobile_v2.0_det_train/best_accuracy   Global.save_inference_dir=./output/quant_inference_model

```
如果要训练识别模型的量化，修改配置文件和加载的模型参数即可。

### 4. 导出模型

在得到量化训练保存的模型后，我们可以将其导出为inference_model，用于预测部署：

```bash
python deploy/slim/quantization/export_model.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=output/quant_model/best_accuracy Global.save_model_dir=./output/quant_inference_model
```

### 5. 量化模型部署

上述步骤导出的量化模型，参数精度仍然是FP32，但是参数的数值范围是int8，导出的模型可以通过PaddleLite的opt模型转换工具完成模型转换。
量化模型部署的可参考 [移动端模型部署](../../lite/readme.md)
