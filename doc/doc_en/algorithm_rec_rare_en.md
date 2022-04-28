# RARE

- [1. Introduction to Algorithms](#1)
- [2. Environment Configuration](#2)
- [3. Model training, evaluation, prediction](#3)
    - [3.1 Training](#3-1)
    - [3.2 Evaluation](#3-2)
    - [3.3 Forecast](#3-3)
- [4. Inference Deployment](#4)
    - [4.1 Python Reasoning](#4-1)
    - [4.2 C++ Reasoning] (#4-2)
    - [4.3 Serving service deployment](#4-3)
    - [4.4 More inference deployments](#4-4)
- [5. FAQ](#5)

<a name="1"></a>
## 1. Introduction to the algorithm

Paper information:
> [Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/abs/1603.03915v2)
> Baoguang Shi, Xinggang Wang, Pengyuan Lyu, Cong Yao, Xiang Bai∗
> CVPR, 2016

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Models|Backbone Networks|Configuration Files|Avg Accuracy|Download Links|
| --- | --- | --- | --- | --- |
|RARE|Resnet34_vd|[configs/rec/rec_r34_vd_tps_bilstm_att.yml](../../configs/rec/rec_r34_vd_tps_bilstm_att.yml)|83.6%|[training model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar)|
|RARE|MobileNetV3|[configs/rec/rec_mv3_tps_bilstm_att.yml](../../configs/rec/rec_mv3_tps_bilstm_att.yml)|82.5%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar)|


<a name="2"></a>
## 2. Environment configuration
Please refer to ["Operating Environment Preparation"](./environment.md) to configure the PaddleOCR operating environment, and refer to ["Project Clone"](./clone.md) to clone the project code.

<a name="3"></a>
## 3. Model training, evaluation, prediction

Please refer to [Text Recognition Training Tutorial](./recognition.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**. Take the backbone network based on Resnet34_vd as an example:

<a name="3-1"></a>
### 3.1 Training

````
#Single card training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_r34_vd_tps_bilstm_att.yml
#Multi-card training, specify the card number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/rec/rec_r34_vd_tps_bilstm_att.yml
````

<a name="3.2"></a>
### 3.2 Evaluation

````
# GPU evaluation, Global.pretrained_model is the model to be evaluated
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r34_vd_tps_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
````

<a name="3.3"></a>
### 3.3 Prediction

````
python3 tools/infer_rec.py -c configs/rec/rec_r34_vd_tps_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
````

<a name="4"></a>
## 4. Inference Deployment

<a name="4-1"></a>
### 4.1 Python Reasoning
First, convert the model saved during the RARE text recognition training process into an inference model. Take the model trained on the MJSynth and SynthText text recognition datasets based on the Resnet34_vd backbone network as an example ([Model download address](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar) ), which can be converted using the following command:

```shell
python3 tools/export_model.py -c configs/rec/rec_r34_vd_tps_bilstm_att.yml -o Global.pretrained_model=./rec_r34_vd_tps_bilstm_att_v2.0_train/best_accuracy Global.save_inference_dir=./inference/rec_rare
````

RARE text recognition model inference, you can execute the following commands:

```shell
python3 tools/infer/predict_rec.py --image_dir="doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_rare/"
````

<a name="4-2"></a>
### 4.2 C++ Reasoning

Not currently supported

<a name="4-3"></a>
### 4.3 Serving service deployment

Not currently supported

<a name="4-4"></a>
### 4.4 More inference deployment

The RARE model also supports the following inference deployment methods:

- Paddle2ONNX Inference: After preparing the inference model, refer to the [paddle2onnx](../../deploy/paddle2onnx/) tutorial.

<a name="5"></a>
## 5. FAQ


## Quote

````bibtex
@inproceedings{2016Robust,
  title={Robust Scene Text Recognition with Automatic Rectification},
  author={ Shi, B. and Wang, X. and Lyu, P. and Cong, Y. and Xiang, B. },
  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016},
}
````
