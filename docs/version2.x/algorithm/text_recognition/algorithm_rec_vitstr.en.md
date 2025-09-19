---
comments: true
---

# ViTSTR

## 1. Introduction

Paper:
> [Vision Transformer for Fast and Efficient Scene Text Recognition](https://arxiv.org/abs/2105.08582)
> Rowel Atienza
> ICDAR, 2021

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Model|Backbone|config|Acc|Download link|
| --- | --- | --- | --- | --- |
|ViTSTR|ViTSTR|[rec_vitstr_none_ce.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_vitstr_none_ce.yml)|79.82%|[trained model](https://paddleocr.bj.bcebos.com/rec_vitstr_none_ce_train.tar)|

## 2. Environment

Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md)to clone the project code.

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

### Training

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```bash linenums="1"
# Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_vitstr_none_ce.yml

# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_vitstr_none_ce.yml
```

### Evaluation

```bash linenums="1"
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_vitstr_none_ce.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### Prediction

```bash linenums="1"
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_vitstr_none_ce.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_vitstr_none_ce_train/best_accuracy
```

## 4. Inference and Deployment

### 4.1 Python Inference

First, the model saved during the ViTSTR text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/rec_vitstr_none_none_train.tar)) ), you can use the following command to convert:

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_vitstr_none_ce.yml -o Global.pretrained_model=./rec_vitstr_none_ce_train/best_accuracy  Global.save_inference_dir=./inference/rec_vitstr
```

**Note:**

- If you are training the model on your own dataset and have modified the dictionary file, please pay attention to modify the `character_dict_path` in the configuration file to the modified dictionary file.
- If you modified the input size during training, please modify the `infer_shape` corresponding to ViTSTR in the `tools/export_model.py` file.

After the conversion is successful, there are three files in the directory:

```text linenums="1"
/inference/rec_vitstr/
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

For ViTSTR text recognition model inference, the following commands can be executed:

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_vitstr/' --rec_algorithm='ViTSTR' --rec_image_shape='1,224,224' --rec_char_dict_path='./ppocr/utils/EN_symbol_dict.txt'
```

![img](./images/word_10.png)

After executing the command, the prediction result (recognized text and score) of the image above is printed to the screen, an example is as follows:
The result is as follows:

```bash linenums="1"
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9998350143432617)
```

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## 5. FAQ

1. In the `ViTSTR` paper, using pre-trained weights on ImageNet1k for initial training, we did not use pre-trained weights in training, and the final accuracy did not change or even improved.

## Citation

```bibtex
@article{Atienza2021ViTSTR,
  title     = {Vision Transformer for Fast and Efficient Scene Text Recognition},
  author    = {Rowel Atienza},
  booktitle = {ICDAR},
  year      = {2021},
  url       = {https://arxiv.org/abs/2105.08582}
}
```
