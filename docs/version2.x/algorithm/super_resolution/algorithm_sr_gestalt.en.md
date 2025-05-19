---
comments: true
---

# Text Gestalt

## 1. Introduction

Paper:
> [Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution](https://arxiv.org/pdf/2112.08171.pdf)
> Chen, Jingye and Yu, Haiyang and Ma, Jianqi and Li, Bin and Xue, Xiangyang
> AAAI, 2022

Referring to the [FudanOCR](https://github.com/FudanVI/FudanOCR/tree/main/text-gestalt) data download instructions, the effect of the super-score algorithm on the TextZoom test set is as follows:

|Model | Backbone|config|Acc|Download link|
|---|---|---|---|---|
|Text Gestalt|tsrn|19.28|0.6560| [configs/sr/sr_tsrn_transformer_strock.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/sr/sr_tsrn_transformer_strock.yml)|[train model](https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar)|

## 2. Environment

Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md)to clone the project code.

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different models only requires **changing the configuration file**.

### Training

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```bash linenums="1"
# Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/sr/sr_tsrn_transformer_strock.yml

# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/sr/sr_tsrn_transformer_strock.yml
```

### Evaluation

```bash linenums="1"
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/sr/sr_tsrn_transformer_strock.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### Prediction

```bash linenums="1"
# The configuration file used for prediction must match the training

python3 tools/infer_sr.py -c configs/sr/sr_tsrn_transformer_strock.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words_en/word_52.png
```

![img](./images/word_52.png)

After executing the command, the super-resolution result of the above image is as follows:

![img](./images/sr_word_52.png)

## 4. Inference and Deployment

### 4.1 Python Inference

First, the model saved during the training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar) ), you can use the following command to convert:

```bash linenums="1"
python3 tools/export_model.py -c configs/sr/sr_tsrn_transformer_strock.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.save_inference_dir=./inference/sr_out
```

For Text-Gestalt super-resolution model inference, the following commands can be executed:

```bash linenums="1"
python3 tools/infer/predict_sr.py --sr_model_dir=./inference/sr_out --image_dir=doc/imgs_words_en/word_52.png --sr_image_shape=3,32,128
```

After executing the command, the super-resolution result of the above image is as follows:

![img](./images/sr_word_52-20240704093810101.png)

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## 5. FAQ

## Citation

```bibtex
@inproceedings{chen2022text,
  title={Text gestalt: Stroke-aware scene text image super-resolution},
  author={Chen, Jingye and Yu, Haiyang and Ma, Jianqi and Li, Bin and Xue, Xiangyang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={1},
  pages={285--293},
  year={2022}
}
```
