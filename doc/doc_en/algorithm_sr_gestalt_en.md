# Text Gestalt

- [1. Introduction](#1)
- [2. Environment](#2)
- [3. Model Training / Evaluation / Prediction](#3)
    - [3.1 Training](#3-1)
    - [3.2 Evaluation](#3-2)
    - [3.3 Prediction](#3-3)
- [4. Inference and Deployment](#4)
    - [4.1 Python Inference](#4-1)
    - [4.2 C++ Inference](#4-2)
    - [4.3 Serving](#4-3)
    - [4.4 More](#4-4)
- [5. FAQ](#5)


<a name="1"></a>
## 1. Introduction

Paper:
> [Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution](https://arxiv.org/pdf/2112.08171.pdf)

> Chen, Jingye and Yu, Haiyang and Ma, Jianqi and Li, Bin and Xue, Xiangyang

> AAAI, 2022

Referring to the [FudanOCR](https://github.com/FudanVI/FudanOCR/tree/main/text-gestalt) data download instructions, the effect of the super-score algorithm on the TextZoom test set is as follows:

|Model|Backbone|config|Acc|Download link|
|---|---|---|---|---|---|
|Text Gestalt|tsrn|19.28|0.6560| [configs/sr/sr_tsrn_transformer_strock.yml](../../configs/sr/sr_tsrn_transformer_strock.yml)|[train model](https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar)|


<a name="2"></a>
## 2. Environment
Please refer to ["Environment Preparation"](./environment_en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](./clone_en.md) to clone the project code.


<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](./recognition_en.md). PaddleOCR modularizes the code, and training different models only requires **changing the configuration file**.

Training:

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```
#Single GPU training (long training period, not recommended)

python3 tools/train.py -c configs/sr/sr_tsrn_transformer_strock.yml

#Multi GPU training, specify the gpu number through the --gpus parameter

python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/sr/sr_tsrn_transformer_strock.yml

```


Evaluation:

```
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/sr/sr_tsrn_transformer_strock.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

Prediction:

```
# The configuration file used for prediction must match the training

python3 tools/infer_sr.py -c configs/sr/sr_tsrn_transformer_strock.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words_en/word_52.png
```

![](../imgs_words_en/word_52.png)

After executing the command, the super-resolution result of the above image is as follows:

![](../imgs_results/sr_word_52.png)

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference

First, the model saved during the training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar) ), you can use the following command to convert:

```shell
python3 tools/export_model.py -c configs/sr/sr_tsrn_transformer_strock.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.save_inference_dir=./inference/sr_out
```

For Text-Gestalt super-resolution model inference, the following commands can be executed:

```
python3 tools/infer/predict_sr.py --sr_model_dir=./inference/sr_out --image_dir=doc/imgs_words_en/word_52.png --sr_image_shape=3,32,128

```

After executing the command, the super-resolution result of the above image is as follows:

![](../imgs_results/sr_word_52.png)


<a name="4-2"></a>
### 4.2 C++ Inference

Not supported

<a name="4-3"></a>
### 4.3 Serving

Not supported

<a name="4-4"></a>
### 4.4 More

Not supported

<a name="5"></a>
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
