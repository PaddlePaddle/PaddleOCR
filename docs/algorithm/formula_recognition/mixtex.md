# MixTeX

- [1. 简介](#1)
- [2. 性能指标](#2)
- [3. 快速开始](#3)
  - [3.1. 准备环境](#31)
  - [3.2. 推理](#32)
- [4. 模型训练](#4)
  - [4.1. 数据准备](#41)
  - [4.2. 开始训练](#42)
  - [4.3. 恢复训练](#43)
- [5. 模型评估](#5)
- [6. 预测](#6)
- [7. 模型导出与推理](#7)
- [8. 参考文献](#8)

<a name="1"></a>

## 1. 简介

MixTeX是一种高效的多模态LaTeX识别模型，专为CPU离线推理优化设计。该模型基于论文"[MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference](https://arxiv.org/abs/2406.17148)"，为公式识别任务提供了高性能的解决方案。

MixTeX的架构包括：
1. 为CPU推理优化的轻量级卷积骨干网络
2. 用于序列生成的Transformer解码器
3. 专门针对LaTeX公式的数据处理和分词方法

<div align="center">
    <img src="../../doc/mixtex/mixtex_architecture.png" width="800">
</div>

<a name="2"></a>

## 2. 性能指标

下表展示了MixTeX在公式识别基准测试上的性能：

|模型|训练数据|测试集|BLEU|编辑距离|精确度|召回率|下载链接|
|---|---|---|---|---|---|---|---|
|MixTeX|Pseudo-Latex-ZhEn-1|测试集|92.8%|0.087|94.1%|93.5%|[训练模型]()|

<a name="3"></a>

## 3. 快速开始

<a name="31"></a>

### 3.1. 准备环境

要使用MixTeX进行公式识别，您需要：

1. 安装PaddleOCR
2. 下载训练好的模型
3. 准备测试图像

<a name="32"></a>

### 3.2. 推理

您可以使用以下命令进行推理：

```bash
python tools/infer/predict_mixtex.py --rec_model_file your_mixtex_inference.pdmodel --rec_params_file your_mixtex_inference.pdiparams --rec_vocab_file your_vocab_file.txt --image_file ./doc/imgs/formula_example.png
```

<a name="4"></a>

## 4. 模型训练

<a name="41"></a>

### 4.1. 数据准备

训练MixTeX需要准备：

1. 训练数据格式如下：
```
图片路径\tLatex公式
```

2. 生成词汇表文件：
```bash
python tools/generate_mixtex_vocab.py \
    --annotation_file 标注文件路径.txt \
    --output_file 输出词汇表路径.txt \
    --min_freq 2 \
    --max_vocab_size 10000
```

<a name="42"></a>

### 4.2. 开始训练

从头开始训练MixTeX：

```bash
python -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/rec/mixtex/mixtex_base.yml
```

<a name="43"></a>

### 4.3. 恢复训练

从检查点恢复训练：

```bash
python -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/rec/mixtex/mixtex_base.yml -o Global.checkpoints=./output/mixtex/latest 
```

<a name="5"></a>

## 5. 模型评估

评估模型性能：

```bash
python -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/mixtex/mixtex_base.yml -o Global.checkpoints=./output/mixtex/best_accuracy
```

<a name="6"></a>

## 6. 预测

```bash
python tools/infer_rec.py -c configs/rec/mixtex/mixtex_base.yml -o Global.pretrained_model=./output/mixtex/best_accuracy Global.infer_img=./doc/imgs/formula_example.png
```

<a name="7"></a>

## 7. 模型导出与推理

将模型转换为推理模型：

```bash
python tools/export_model.py -c configs/rec/mixtex/mixtex_base.yml -o Global.pretrained_model=./output/mixtex/best_accuracy Global.save_inference_dir=./output/mixtex/inference
```

<a name="8"></a>

## 8. 参考文献

- [MixTeX: An Efficient Multi-Modal LaTeX Recognition Model for Offline CPU Inference](https://arxiv.org/abs/2406.17148)
- [MixTeX代码仓库](https://github.com/RQLuo/MixTeX-Latex-OCR)
- [Pseudo-Latex-ZhEn-1数据集](https://huggingface.co/datasets/MixTex/Pseudo-Latex-ZhEn-1)
