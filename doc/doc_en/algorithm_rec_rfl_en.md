# RFL

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
> [Reciprocal Feature Learning via Explicit and Implicit Tasks in Scene Text Recognition](https://arxiv.org/abs/2105.06229.pdf)
> Hui Jiang, Yunlu Xu, Zhanzhan Cheng, Shiliang Pu, Yi Niu, Wenqi Ren, Fei Wu, and Wenming Tan
> ICDAR, 2021

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Model|Backbone|config|Acc|Download link|
| --- | --- | --- | --- | --- |
|RFL-CNT|ResNetRFL|[rec_resnet_rfl_visual.yml](../../configs/rec/rec_resnet_rfl_visual.yml)|93.40%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_resnet_rfl_visual_train.tar)|
|RFL-Att|ResNetRFL|[rec_resnet_rfl_att.yml](../../configs/rec/rec_resnet_rfl_att.yml)|88.63%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_resnet_rfl_att_train.tar)|

<a name="2"></a>
## 2. Environment
Please refer to ["Environment Preparation"](./environment_en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](./clone_en.md) to clone the project code.


<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

Training:

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```
#step1:train the CNT branch
#Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_resnet_rfl_visual.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_resnet_rfl_visual.yml

#step2:joint training of CNT and Att branches
#Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_resnet_rfl_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_resnet_rfl_att.yml  -o Global.pretrained_model={path/to/weights}/best_accuracy


```

Evaluation:

```
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_resnet_rfl_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_resnet_rfl_att.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model={path/to/weights}/best_accuracy
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, the model saved during the RFL text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/contribution/rec_resnet_rfl.tar)) ), you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_resnet_rfl_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy  Global.save_inference_dir=./inference/rec_resnet_rfl_att
```

**Note:**
- If you are training the model on your own dataset and have modified the dictionary file, please pay attention to modify the `character_dict_path` in the configuration file to the modified dictionary file.
- If you modified the input size during training, please modify the `infer_shape` corresponding to NRTR in the `tools/export_model.py` file.

After the conversion is successful, there are three files in the directory:
```
/inference/rec_resnet_rfl_att/
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```


For RFL text recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_resnet_rfl_att/' --rec_algorithm='RFL' --rec_image_shape='1,32,100'
```

![](../imgs_words_en/word_10.png)

After executing the command, the prediction result (recognized text and score) of the image above is printed to the screen, an example is as follows:
The result is as follows:
```shell
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9999927282333374)
```

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
@article{2021Reciprocal,
  title     = {Reciprocal Feature Learning via Explicit and Implicit Tasks in Scene Text Recognition},
  author    = {Jiang, H.  and  Xu, Y.  and  Cheng, Z.  and  Pu, S.  and  Niu, Y.  and  Ren, W.  and  Wu, F.  and  Tan, W. },
  booktitle = {ICDAR},
  year      = {2021},
  url       = {https://arxiv.org/abs/2105.06229}
}
```
