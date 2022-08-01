# SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition

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
> [SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition](https://arxiv.org/abs/2005.13117)
> Chengwei Zhang, Yunlu Xu, Zhanzhan Cheng, Shiliang Pu, Yi Niu, Fei Wu, Futai Zou
> AAAI, 2020

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets. The algorithm reproduction effect is as follows:

|Model|Backbone|config|Acc|Download link|
| --- | --- | --- | --- | --- |
|SPIN|ResNet32|[rec_r32_gaspin_bilstm_att.yml](../../configs/rec/rec_r32_gaspin_bilstm_att.yml)|90.0%|coming soon|


<a name="2"></a>
## 2. Environment
Please refer to ["Environment Preparation"](./environment_en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](./clone_en.md) to clone the project code.


<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](./recognition_en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

Training:

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```
#Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml
```

Evaluation:

```
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, the model saved during the SPIN text recognition training process is converted into an inference model. you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy  Global.save_inference_dir=./inference/rec_r32_gaspin_bilstm_att
```

For SPIN text recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_r32_gaspin_bilstm_att/" --rec_image_shape="3, 32, 100" --rec_algorithm="SPIN" --rec_char_dict_path="/ppocr/utils/dict/spin_dict.txt" --use_space_char=False
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
@article{2020SPIN,
  title={SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition},
  author={Chengwei Zhang and Yunlu Xu and Zhanzhan Cheng and Shiliang Pu and Yi Niu and Fei Wu and Futai Zou},
  journal={AAAI2020},
  year={2020},
}
```
