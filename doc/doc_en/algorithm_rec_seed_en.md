# SEED

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
> [SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition](https://arxiv.org/pdf/2005.10977.pdf)

> Qiao, Zhi and Zhou, Yu and Yang, Dongbao and Zhou, Yucan and Wang, Weiping

> CVPR, 2020

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Model|Backbone|ACC|config|Download link|
| --- | --- | --- | --- | --- |
|SEED|Aster_Resnet| 85.2% | [configs/rec/rec_resnet_stn_bilstm_att.yml](../../configs/rec/rec_resnet_stn_bilstm_att.yml) | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_resnet_stn_bilstm_att.tar) |

<a name="2"></a>
## 2. Environment
Please refer to ["Environment Preparation"](./environment.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](./clone.md) to clone the project code.


<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](./recognition.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

Training:

The SEED model needs to additionally load the [language model](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) trained by FastText, and install the fasttext dependencies:

```
python3 -m pip install fasttext==0.9.1
```

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```
#Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_resnet_stn_bilstm_att.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c rec_resnet_stn_bilstm_att.yml
```

Evaluation:

```
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_resnet_stn_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_resnet_stn_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference

Not support

<a name="4-2"></a>
### 4.2 C++ Inference

Not support

<a name="4-3"></a>
### 4.3 Serving

Not support

<a name="4-4"></a>
### 4.4 More

Not support

<a name="5"></a>
## 5. FAQ


## Citation

```bibtex
@inproceedings{qiao2020seed,
  title={Seed: Semantics enhanced encoder-decoder framework for scene text recognition},
  author={Qiao, Zhi and Zhou, Yu and Yang, Dongbao and Zhou, Yucan and Wang, Weiping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13528--13537},
  year={2020}
}
```
