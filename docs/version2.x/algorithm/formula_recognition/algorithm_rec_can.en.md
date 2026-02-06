---
comments: true
---

## 1. Introduction

Paper:
> [When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2207.11463)
> Bohan Li, Ye Yuan, Dingkang Liang, Xiao Liu, Zhilong Ji, Jinfeng Bai, Wenyu Liu, Xiang Bai
> ECCV, 2022

Using CROHME handwrittem mathematical expression recognition datasets for training, and evaluating on its test sets, the algorithm reproduction effect is as follows:

|Model|Backbone|config|exprate|Download link|
| --- | --- | --- | --- | --- |
|CAN|DenseNet|[rec_d28_can.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_d28_can.yml)|51.72%|[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_d28_can_train.tar)|

## 2. Environment

Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md)to clone the project code.

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

### Training

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```bash linenums="1"
# Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_d28_can.yml

# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_d28_can.yml
```

### Evaluation

```bash linenums="1"
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_d28_can.yml -o Global.pretrained_model=./rec_d28_can_train/best_accuracy.pdparams
```

Prediction:

```bash linenums="1"
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_d28_can.yml -o Architecture.Head.attdecoder.is_train=False Global.infer_img='./doc/crohme_demo/hme_00.jpg' Global.pretrained_model=./rec_d28_can_train/best_accuracy.pdparams
```

## 4. Inference and Deployment

### 4.1 Python Inference

First, the model saved during the CAN handwritten mathematical expression recognition training process is converted into an inference model. you can use the following command to convert:

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_d28_can.yml -o Global.pretrained_model=./rec_d28_can_train/best_accuracy.pdparams Global.save_inference_dir=./inference/rec_d28_can/ Architecture.Head.attdecoder.is_train=False

# The default output max length of the model is 36. If you need to predict a longer sequence, please specify its output sequence as an appropriate value when exporting the model, as: Architecture.Head.max_ text_ length=72
```

For CAN handwritten mathematical expression recognition model inference, the following commands can be executed:

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="./doc/datasets/crohme_demo/hme_00.jpg" --rec_algorithm="CAN" --rec_batch_num=1 --rec_model_dir="./inference/rec_d28_can/" --rec_char_dict_path="./ppocr/utils/dict/latex_symbol_dict.txt"

# If you need to predict on a picture with black characters on a white background, please set: -- rec_ image_ inverse=False
```

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## 5. FAQ

## Citation

```bibtex
@misc{https://doi.org/10.48550/arxiv.2207.11463,
  doi = {10.48550/ARXIV.2207.11463},
  url = {https://arxiv.org/abs/2207.11463},
  author = {Li, Bohan and Yuan, Ye and Liang, Dingkang and Liu, Xiao and Ji, Zhilong and Bai, Jinfeng and Liu, Wenyu and Bai, Xiang},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
