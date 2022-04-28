# STAR-Net

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
> [STAR-Net: a spatial attention residue network for scene text recognition.](http://www.bmva.org/bmvc/2016/papers/paper043/paper043.pdf)

> Wei Liu, Chaofeng Chen, Kwan-Yee K. Wong, Zhizhong Su and Junyu Han.

> BMVC, pages 43.1-43.13, 2016

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Model|Backbone|ACC|config|Download link|
| --- | --- | --- | --- | --- |
|---|---|---|---|---|
|StarNet|Resnet34_vd|84.44%|[configs/rec/rec_r34_vd_tps_bilstm_ctc.yml](../../configs/rec/rec_r34_vd_tps_bilstm_ctc.yml)|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|81.42%|[configs/rec/rec_mv3_tps_bilstm_ctc.yml](../../configs/rec/rec_mv3_tps_bilstm_ctc.yml)|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|

<a name="2"></a>
## 2. Environment
Please refer to ["Environment Preparation"](./environment.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](./clone.md) to clone the project code.


<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](./recognition.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

Training:

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```
#Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c rec_r34_vd_tps_bilstm_ctc.yml
```

Evaluation:

```
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, the model saved during the STAR-Net text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_STAR-Net_train.tar) ), you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml -o Global.pretrained_model=./rec_r34_vd_tps_bilstm_ctc_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/rec_starnet
```

For STAR-Net text recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./inference/rec_starnet/" --rec_image_shape="3, 32, 100" --rec_char_dict_path="./ppocr/utils/ic15_dict.txt"
```

<a name="4-2"></a>
### 4.2 C++ Inference

With the inference model prepared, refer to the [cpp infer](../../deploy/cpp_infer/) tutorial for C++ inference.


<a name="4-3"></a>
### 4.3 Serving

With the inference model prepared, refer to the [pdserving](../../deploy/pdserving/) tutorial for service deployment by Paddle Serving.


<a name="4-4"></a>
### 4.4 More

More deployment schemes supported for STAR-Net:

- Paddle2ONNX: with the inference model prepared, please refer to the [paddle2onnx](../../deploy/paddle2onnx/) tutorial.


<a name="5"></a>
## 5. FAQ


## Citation

```bibtex
@inproceedings{liu2016star,
  title={STAR-Net: a spatial attention residue network for scene text recognition.},
  author={Liu, Wei and Chen, Chaofeng and Wong, Kwan-Yee K and Su, Zhizhong and Han, Junyu},
  booktitle={BMVC},
  volume={2},
  pages={7},
  year={2016}
}
```
