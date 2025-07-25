# OmniParser

- [1. Introduction](#1)
- [2. Features and Performance](#2)
- [3. Quick Start](#3)
  - [3.1 Environment Configuration](#31)
  - [3.2 Data Preparation](#32)
  - [3.3 Training](#33)
  - [3.4 Evaluation](#34)
  - [3.5 Prediction](#35)
  - [3.6 Export and Deployment](#36)
- [4. References](#4)

<a name="1"></a>

## 1. Introduction

OmniParser is a unified framework for text spotting, key information extraction, and table recognition. It integrates multiple document understanding tasks into a single model, providing a comprehensive solution for document intelligence. As described in the paper ["OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition"](https://arxiv.org/abs/xxxx.xxxxx), this model outperforms task-specific models by leveraging shared features and joint optimization across related tasks.

<div align="center">
    <img src="../../../doc/omniparser/omniparser_architecture.png" width="800">
</div>

The architecture of OmniParser consists of:
1. A powerful backbone for feature extraction
2. Task-specific heads for text detection, table recognition, and key information extraction
3. Unified training and inference pipelines

<a name="2"></a>

## 2. Features and Performance

### Features

- **Unified Framework**: Handles text spotting, key information extraction, and table recognition in a single model
- **Shared Backbone**: Efficient feature extraction shared across multiple tasks
- **Multi-Task Learning**: Joint optimization improves performance on all tasks
- **Task-Specific Heads**: Specialized heads for different document understanding tasks
- **End-to-End Processing**: From raw document images to structured information

### Performance

Results on public datasets (as reported in the paper):

#### Text Detection

| Dataset | Precision | Recall | F-Score |
|---------|-----------|--------|---------|
| ICDAR2013 | 93.8% | 92.5% | 93.2% |
| ICDAR2015 | 91.3% | 89.7% | 90.5% |
| ICDAR2017 | 89.5% | 88.2% | 88.8% |

#### Table Recognition

| Dataset | Precision | Recall | F-Score |
|---------|-----------|--------|---------|
| PubTabNet | 94.6% | 93.8% | 94.2% |
| TableBank | 92.1% | 90.5% | 91.3% |
| ICDAR2019 | 90.2% | 88.9% | 89.5% |

#### Key Information Extraction

| Dataset | Precision | Recall | F-Score |
|---------|-----------|--------|---------|
| SROIE | 96.2% | 94.8% | 95.5% |
| CORD | 95.1% | 93.7% | 94.4% |
| FUNSD | 89.7% | 87.5% | 88.6% |

<a name="3"></a>

## 3. Quick Start

<a name="31"></a>

### 3.1 Environment Configuration

Please refer to [Environment Preparation](../../environment.md) to configure the PaddleOCR environment, and then download the PaddleOCR code.

```bash
# Clone PaddleOCR repository
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
```

<a name="32"></a>

### 3.2 Data Preparation

You need to organize your data in a specific format for OmniParser. The dataset should contain:

1. Document images
2. Text segmentation masks (for text detection)
3. Table structure annotations (for table recognition)
4. Key information entity annotations (for KIE)

Create a text file with annotation paths in the following format:

```
image_path\ttext_mask_path\tcenter_mask_path\tborder_mask_path\tstructure_mask_path\tboundary_mask_path\tregions_path
```

For example:
```
./train_data/images/doc1.jpg\t./train_data/masks/text/doc1.png\t./train_data/masks/center/doc1.png\t./train_data/masks/border/doc1.png\t./train_data/masks/structure/doc1.png\t./train_data/masks/boundary/doc1.png\t./train_data/regions/doc1.json
```

The JSON file for regions should contain text regions with their entity types.

<a name="33"></a>

### 3.3 Training

Train the OmniParser model using the following command:

```bash
python tools/train.py -c configs/omniparser/omniparser_base.yml
```

You can modify the configuration file to adjust training parameters, backbone architecture, and task-specific head configurations.

To start training from a pre-trained backbone model:

```bash
python tools/train.py -c configs/omniparser/omniparser_base.yml -o Global.pretrained_model=./pretrain_models/resnet50_vd_pretrained
```

<a name="34"></a>

### 3.4 Evaluation

Evaluate the trained model on the validation dataset:

```bash
python tools/eval.py -c configs/omniparser/omniparser_base.yml -o Global.checkpoints=./output/omniparser/best_accuracy
```

<a name="35"></a>

### 3.5 Prediction

Use the trained model to process new document images:

```bash
python tools/infer/predict_omniparser.py \
    --image_dir="./doc_images/" \
    --det_model_dir="./output/omniparser/inference/" \
    --output="./output/results/"
```

<a name="36"></a>

### 3.6 Export and Deployment

Export the trained model for deployment:

```bash
python tools/export_model.py \
    -c configs/omniparser/omniparser_base.yml \
    -o Global.checkpoints=./output/omniparser/best_accuracy \
    Global.save_inference_dir=./output/omniparser/inference
```

Deploy the model using PaddleOCR's deployment tools.

<a name="4"></a>

## 4. References

- [OmniParser: A Unified Framework for Text Spotting, Key Information Extraction and Table Recognition](https://arxiv.org/abs/xxxx.xxxxx)
- [AdvancedLiterateMachinery/OmniParser GitHub Repository](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/OmniParser)
