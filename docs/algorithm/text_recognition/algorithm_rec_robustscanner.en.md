---
comments: true
---

# RobustScanner

## 1. Introduction

Paper:
> [RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition](https://arxiv.org/pdf/2007.07542.pdf)
> Xiaoyu Yue, Zhanghui Kuang, Chenhao Lin, Hongbin Sun, Wayne
Zhang
> ECCV, 2020

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Model|Backbone|config|Acc|Download link|
| --- | --- | --- | --- | --- |
|RobustScanner|ResNet31|[rec_r31_robustscanner.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_r31_robustscanner.yml)|87.77%|[trained model](https://paddleocr.bj.bcebos.com/contribution/rec_r31_robustscanner.tar)|

Note:In addition to using the two text recognition datasets MJSynth and SynthText, [SynthAdd](https://pan.baidu.com/share/init?surl=uV0LtoNmcxbO-0YA7Ch4dg) data (extraction code: 627x), and some real data are used in training, the specific data details can refer to the paper.

## 2. Environment

Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md)to clone the project code.

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

### Training

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```bash linenums="1"
# Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_r31_robustscanner.yml

# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r31_robustscanner.yml
```

### Evaluation

```bash linenums="1"
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r31_robustscanner.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### Prediction

```bash linenums="1"
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_r31_robustscanner.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

## 4. Inference and Deployment

### 4.1 Python Inference

First, the model saved during the RobustScanner text recognition training process is converted into an inference model. you can use the following command to convert:

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_r31_robustscanner.yml -o Global.pretrained_model={path/to/weights}/best_accuracy  Global.save_inference_dir=./inference/rec_r31_robustscanner
```

For RobustScanner text recognition model inference, the following commands can be executed:

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_r31_robustscanner/" --rec_image_shape="3, 48, 48, 160" --rec_algorithm="RobustScanner" --rec_char_dict_path="ppocr/utils/dict90.txt" --use_space_char=False
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
@article{2020RobustScanner,
  title={RobustScanner: Dynamically Enhancing Positional Clues for Robust Text Recognition},
  author={Xiaoyu Yue and Zhanghui Kuang and Chenhao Lin and Hongbin Sun and Wayne Zhang},
  journal={ECCV2020},
  year={2020},
}
```
