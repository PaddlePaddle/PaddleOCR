
## 介绍

复杂的模型有利于提高模型的性能，但也导致模型中存在一定冗余，模型裁剪通过移出网络模型中的子模型来减少这种冗余，达到减少模型计算复杂度，提高模型推理性能的目的。

本教程将介绍如何使用PaddleSlim量化PaddleOCR的模型。

在开始本教程之前，建议先了解
1. [PaddleOCR模型的训练方法](../../../doc/doc_ch/quickstart.md)
2. [分类模型裁剪教程](https://paddlepaddle.github.io/PaddleSlim/tutorials/pruning_tutorial/)
3. [PaddleSlim 裁剪压缩API](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/)


## 快速开始

模型裁剪主要包括五个步骤：
1. 安装 PaddleSlim
2. 准备训练好的模型
3. 敏感度分析、训练
4. 模型裁剪训练
5. 导出模型、预测部署

### 1. 安装PaddleSlim

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python setup.py install
```

### 2. 获取预训练模型
模型裁剪需要加载事先训练好的模型，PaddleOCR也提供了一系列模型[../../../doc/doc_ch/models_list.md]，开发者可根据需要自行选择模型或使用自己的模型。

### 3. 敏感度分析训练

加载预训练模型后，通过对现有模型的每个网络层进行敏感度分析，得到敏感度文件：sensitivities_0.data，可以通过PaddleSlim提供的[接口](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/sensitive.py#L221)加载文件，获得各网络层在不同裁剪比例下的精度损失。从而了解各网络层冗余度，决定每个网络层的裁剪比例。
敏感度分析的具体细节见：[敏感度分析](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/image_classification_sensitivity_analysis_tutorial.md)

进入PaddleOCR根目录，通过以下命令对模型进行敏感度分析训练：
```bash
python deploy/slim/prune/sensitivity_anal.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights="your trained model" Global.test_batch_size_per_card=1
```

### 4. 模型裁剪训练
裁剪时通过之前的敏感度分析文件决定每个网络层的裁剪比例。在具体实现时，为了尽可能多的保留从图像中提取的低阶特征，我们跳过了backbone中靠近输入的4个卷积层。同样，为了减少由于裁剪导致的模型性能损失，我们通过之前敏感度分析所获得的敏感度表，人工挑选出了一些冗余较少，对裁剪较为敏感的[网络层](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/slim/prune/pruning_and_finetune.py#L41)（指对其进行较低比例裁剪就会导致模型性能显著下降的网络层），并在之后的裁剪过程中选择避开这些网络层。裁剪过后finetune的过程沿用OCR检测模型原始的训练策略。

```bash
python deploy/slim/prune/pruning_and_finetune.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1
```
通过对比可以发现，经过裁剪训练保存的模型更小。

### 5. 导出模型、预测部署

在得到裁剪训练保存的模型后，我们可以将其导出为inference_model：
```bash
python deploy/slim/prune/export_prune_model.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./output/det_db/best_accuracy Global.test_batch_size_per_card=1 Global.save_inference_dir=inference_model
```

inference model的预测和部署参考：
1. [inference model python端预测](../../../doc/doc_ch/inference.md)
2. [inference model C++预测](../../cpp_infer/readme.md)
3. [inference model在移动端部署](../../lite/readme.md)
