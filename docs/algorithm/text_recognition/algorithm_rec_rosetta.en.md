---
comments: true
---

# Rosetta

## 1. Introduction

Paper information:
> [Rosetta: Large Scale System for Text Detection and Recognition in Images](https://arxiv.org/abs/1910.05085)
> Borisyuk F , Gordo A , V Sivakumar
> KDD, 2018

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Models|Backbone Networks|Configuration Files|Avg Accuracy|Download Links|
| --- | --- | --- | --- | --- |
|Rosetta|Resnet34_vd|[configs/rec/rec_r34_vd_none_none_ctc.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_r34_vd_none_none_ctc.yml)|79.11%|[training model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|[configs/rec/rec_mv3_none_none_ctc.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_mv3_none_none_ctc.yml)|75.80%|[training model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|

## 2. Environment

Please refer to [Operating Environment Preparation](../../ppocr/environment.en.md) to configure the PaddleOCR operating environment, and refer to [Project Clone](../../ppocr/blog/clone.en.md)to clone the project code.

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Training Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**. Take the backbone network based on Resnet34_vd as an example:

### 3.1 Training

```bash linenums="1"
# Single card training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_r34_vd_none_none_ctc.yml
# Multi-card training, specify the card number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/rec/rec_r34_vd_none_none_ctc.yml
```

### 3.2 Evaluation

```bash linenums="1"
# GPU evaluation, Global.pretrained_model is the model to be evaluated
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r34_vd_none_none_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### 3.3 Prediction

```bash linenums="1"
python3 tools/infer_rec.py -c configs/rec/rec_r34_vd_none_none_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

## 4. Inference and Deployment

### 4.1 Python Inference

First, convert the model saved during the Rosetta text recognition training process into an inference model. Take the model trained on the MJSynth and SynthText text recognition datasets based on the Resnet34_vd backbone network as an example ( [Model download address](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar) ), which can be converted using the following command:

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_r34_vd_none_none_ctc.yml -o Global.pretrained_model=./rec_r34_vd_none_none_ctc_v2.0_train/best_accuracy Global.save_inference_dir=./inference/rec_rosetta
```

Rosetta text recognition model inference, you can execute the following commands:

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_rosetta/" --rec_image_shape="3, 32, 100" --rec_char_dict_path= "./ppocr/utils/ic15_dict.txt"
```

The inference results are as follows:

![img](./images/word_1-20240704183926496.png)

```bash linenums="1"
Predicts of doc/imgs_words/en/word_1.png:('joint', 0.9999982714653015)
```

### 4.2 C++ Inference

Not currently supported

### 4.3 Serving

Not currently supported

### 4.4 More

The Rosetta model also supports the following inference deployment methods:

- Paddle2ONNX Inference: After preparing the inference model, refer to the [paddle2onnx](../../ppocr/infer_deploy/paddle2onnx.en.md) tutorial.

## 5. FAQ

## Citation

```bibtex
@inproceedings{2018Rosetta,
  title={Rosetta: Large Scale System for Text Detection and Recognition in Images},
  author={ Borisyuk, Fedor and Gordo, Albert and Sivakumar, Viswanath },
  booktitle={the 24th ACM SIGKDD International Conference},
  year={2018},
}
```
