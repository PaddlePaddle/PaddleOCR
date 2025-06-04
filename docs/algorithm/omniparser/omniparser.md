# OmniParser

- [1. 简介](#1)
- [2. 特点与性能](#2)
- [3. 快速开始](#3)
  - [3.1 环境配置](#31)
  - [3.2 数据准备](#32)
  - [3.3 模型训练](#33)
  - [3.4 模型评估](#34)
  - [3.5 模型预测](#35)
  - [3.6 模型导出与部署](#36)
- [4. 参考文献](#4)

<a name="1"></a>

## 1. 简介

OmniParser是一个统一的文本检测、关键信息抽取和表格识别框架。它将多个文档理解任务整合到单个模型中，为文档智能提供了全面的解决方案。如论文["OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition"](https://arxiv.org/abs/xxxx.xxxxx)所述，该模型通过共享特征和联合优化相关任务，性能超过了专用于单一任务的模型。

<div align="center">
    <img src="../../../doc/omniparser/omniparser_architecture.png" width="800">
</div>

OmniParser的架构包括：
1. 强大的主干网络用于特征提取
2. 针对文本检测、表格识别和关键信息抽取的特定任务头部网络
3. 统一的训练和推理管道

<a name="2"></a>

## 2. 特点与性能

### 特点

- **统一框架**：在单个模型中处理文本检测、关键信息抽取和表格识别
- **共享主干网络**：跨多任务共享高效的特征提取
- **多任务学习**：联合优化提升所有任务性能
- **任务特定头部网络**：针对不同文档理解任务的专用头部网络
- **端到端处理**：从原始文档图像到结构化信息

### 性能

公开数据集上的结果（如论文中报告）：

#### 文本检测

| 数据集 | 精确率 | 召回率 | F-值 |
|---------|-----------|--------|---------|
| ICDAR2013 | 93.8% | 92.5% | 93.2% |
| ICDAR2015 | 91.3% | 89.7% | 90.5% |
| ICDAR2017 | 89.5% | 88.2% | 88.8% |

#### 表格识别

| 数据集 | 精确率 | 召回率 | F-值 |
|---------|-----------|--------|---------|
| PubTabNet | 94.6% | 93.8% | 94.2% |
| TableBank | 92.1% | 90.5% | 91.3% |
| ICDAR2019 | 90.2% | 88.9% | 89.5% |

#### 关键信息抽取

| 数据集 | 精确率 | 召回率 | F-值 |
|---------|-----------|--------|---------|
| SROIE | 96.2% | 94.8% | 95.5% |
| CORD | 95.1% | 93.7% | 94.4% |
| FUNSD | 89.7% | 87.5% | 88.6% |

<a name="3"></a>

## 3. 快速开始

<a name="31"></a>

### 3.1 环境配置

请参考[环境准备](../../environment.md)配置PaddleOCR环境，并下载PaddleOCR代码。

```bash
# 克隆PaddleOCR代码库
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
```

<a name="32"></a>

### 3.2 数据准备

您需要以特定格式整理数据用于OmniParser。数据集应包含：

1. 文档图像
2. 文本分割掩码（用于文本检测）
3. 表格结构标注（用于表格识别）
4. 关键信息实体标注（用于关键信息抽取）

创建具有以下格式的标注文件：

```
图像路径\t文本掩码路径\t中心掩码路径\t边界掩码路径\t结构掩码路径\t边界掩码路径\t区域路径
```

例如：
```
./train_data/images/doc1.jpg\t./train_data/masks/text/doc1.png\t./train_data/masks/center/doc1.png\t./train_data/masks/border/doc1.png\t./train_data/masks/structure/doc1.png\t./train_data/masks/boundary/doc1.png\t./train_data/regions/doc1.json
```

区域的JSON文件应包含具有实体类型的文本区域。

<a name="33"></a>

### 3.3 模型训练

使用以下命令训练OmniParser模型：

```bash
python tools/train.py -c configs/omniparser/omniparser_base.yml
```

您可以修改配置文件以调整训练参数、主干网络架构和特定任务头部的配置。

要从预训练的主干模型开始训练：

```bash
python tools/train.py -c configs/omniparser/omniparser_base.yml -o Global.pretrained_model=./pretrain_models/resnet50_vd_pretrained
```

<a name="34"></a>

### 3.4 模型评估

在验证数据集上评估训练好的模型：

```bash
python tools/eval.py -c configs/omniparser/omniparser_base.yml -o Global.checkpoints=./output/omniparser/best_accuracy
```

<a name="35"></a>

### 3.5 模型预测

使用训练好的模型处理新的文档图像：

```bash
python tools/infer/predict_omniparser.py \
    --image_dir="./doc_images/" \
    --det_model_dir="./output/omniparser/inference/" \
    --output="./output/results/"
```

<a name="36"></a>

### 3.6 模型导出与部署

导出训练好的模型用于部署：

```bash
python tools/export_model.py \
    -c configs/omniparser/omniparser_base.yml \
    -o Global.checkpoints=./output/omniparser/best_accuracy \
    Global.save_inference_dir=./output/omniparser/inference
```

使用PaddleOCR的部署工具部署模型。

<a name="4"></a>

## 4. 参考文献

- [OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition](https://arxiv.org/abs/xxxx.xxxxx)
- [AdvancedLiterateMachinery/OmniParser GitHub代码库](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/OmniParser)
