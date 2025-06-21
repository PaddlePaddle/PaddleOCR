# MixTeX

- [1. Introduction](#1)
- [2. Performance](#2)
- [3. Quick Start](#3)
  - [3.1. Preparation](#31)
  - [3.2. Inference](#32)
- [4. Training](#4)
  - [4.1. Data Preparation](#41)
  - [4.2. Start Training](#42)
  - [4.3. Resume Training](#43)
- [5. Evaluation](#5)
- [6. Prediction](#6)
- [7. Export and Inference](#7)
- [8. References](#8)

<a name="1"></a>

## 1. Introduction

MixTeX is an efficient multi-modal LaTeX recognition model optimized for offline CPU inference. This model is based on the paper "[MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference](https://arxiv.org/abs/2406.17148)" and provides a high-performance solution for formula recognition tasks.

The architecture of MixTeX consists of:
1. A lightweight convolutional backbone optimized for CPU inference
2. A transformer-based decoder for sequence generation
3. Specialized data processing and tokenization for LaTeX formulas

<div align="center">
    <img src="../../doc/mixtex/mixtex_architecture.png" width="800">
</div>

<a name="2"></a>

## 2. Performance

The following table shows the performance of MixTeX on formula recognition benchmarks:

|Model|Training Data|Test Set|BLEU|Edit Distance|Precision|Recall|Download Link|
|---|---|---|---|---|---|---|---|
|MixTeX|Pseudo-Latex-ZhEn-1|test set|92.8%|0.087|94.1%|93.5%|[trained model]()|

<a name="3"></a>

## 3. Quick Start

<a name="31"></a>

### 3.1. Preparation

To use MixTeX for formula recognition, you need to:

1. Install PaddleOCR
2. Download the trained model
3. Prepare test images

<a name="32"></a>

### 3.2. Inference

You can use the following command for inference:

```bash
python tools/infer/predict_mixtex.py --rec_model_file your_mixtex_inference.pdmodel --rec_params_file your_mixtex_inference.pdiparams --rec_vocab_file your_vocab_file.txt --image_file ./doc/imgs/formula_example.png
```

<a name="4"></a>

## 4. Training

<a name="41"></a>

### 4.1. Data Preparation

For training MixTeX, you need to prepare:

1. Training data in the following format:
```
image_path\tlatex_formula
```

2. Generate a vocabulary file:
```bash
python tools/generate_mixtex_vocab.py \
    --annotation_file path/to/annotation.txt \
    --output_file path/to/vocab.txt \
    --min_freq 2 \
    --max_vocab_size 10000
```

<a name="42"></a>

### 4.2. Start Training

To train MixTeX from scratch:

```bash
python -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/mixtex/mixtex_base.yml
```

<a name="43"></a>

### 4.3. Resume Training

To resume training from checkpoint:

```bash
python -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/mixtex/mixtex_base.yml -o Global.checkpoints=./output/mixtex/latest 
```

<a name="5"></a>

## 5. Evaluation

To evaluate the model performance:

```bash
python -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/mixtex/mixtex_base.yml -o Global.checkpoints=./output/mixtex/best_accuracy
```

<a name="6"></a>

## 6. Prediction

```bash
python tools/infer_rec.py -c configs/rec/mixtex/mixtex_base.yml -o Global.pretrained_model=./output/mixtex/best_accuracy Global.infer_img=./doc/imgs/formula_example.png
```

<a name="7"></a>

## 7. Export and Inference

To convert the model to inference model:

```bash
python tools/export_model.py -c configs/rec/mixtex/mixtex_base.yml -o Global.pretrained_model=./output/mixtex/best_accuracy Global.save_inference_dir=./output/mixtex/inference
```

<a name="8"></a>

## 8. References

- [MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference](https://arxiv.org/abs/2406.17148)
- [MixTeX Code Repository](https://github.com/RQLuo/MixTeX-Latex-OCR)
- [Pseudo-Latex-ZhEn-1 Dataset](https://huggingface.co/datasets/MixTex/Pseudo-Latex-ZhEn-1)
