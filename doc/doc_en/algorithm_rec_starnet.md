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

Paper information:
> [STAR-Net: a spatial attention residue network for scene text recognition.](http://www.bmva.org/bmvc/2016/papers/paper043/paper043.pdf)
> Wei Liu, Chaofeng Chen, Kwan-Yee K. Wong, Zhizhong Su and Junyu Han.
> BMVC, pages 43.1-43.13, 2016

Refer to [DTRB](https://arxiv.org/abs/1904.01906) text Recognition Training and Evaluation Process . Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Models|Backbone Networks|Avg Accuracy|Configuration Files|Download Links|
| --- | --- | --- | --- | --- |
|StarNet|Resnet34_vd|84.44%|[configs/rec/rec_r34_vd_tps_bilstm_ctc.yml](../../configs/rec/rec_r34_vd_tps_bilstm_ctc.yml)|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|81.42%|[configs/rec/rec_mv3_tps_bilstm_ctc.yml](../../configs/rec/rec_mv3_tps_bilstm_ctc.yml)|[ trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|


<a name="2"></a>
## 2. Environment
Please refer to [Operating Environment Preparation](./environment_en.md) to configure the PaddleOCR operating environment, and refer to [Project Clone](./clone_en.md) to clone the project code.

<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Training Tutorial](./recognition_en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**. Take the backbone network based on Resnet34_vd as an example:

<a name="3-1"></a>
### 3.1 Training
After the data preparation is complete, the training can be started. The training command is as follows:

````
#Single card training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml #Multi-card training, specify the card number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c rec_r34_vd_tps_bilstm_ctc.yml
 ````

<a name="3-2"></a>
### 3.2 Evaluation

````
# GPU evaluation, Global.pretrained_model is the model to be evaluated
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
 ````

<a name="3-3"></a>
### 3.3 Prediction

````
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
 ````

<a name="4"></a>
## 4. Inference

<a name="4-1"></a>
### 4.1 Python Inference
First, convert the model saved during the STAR-Net text recognition training process into an inference model. Take the model trained on the MJSynth and SynthText text recognition datasets based on the Resnet34_vd backbone network as an example [Model download address]( https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar) , which can be converted using the following command:

```shell
python3 tools/export_model.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml -o Global.pretrained_model=./rec_r34_vd_tps_bilstm_ctc_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/rec_starnet
 ````

STAR-Net text recognition model inference, you can execute the following commands:

```shell
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./inference/rec_starnet/" --rec_image_shape="3, 32, 100" --rec_char_dict_path="./ppocr/utils/ic15_dict.txt"
 ````

![](../imgs_words_en/word_336.png)

The inference results are as follows:


```bash
Predicts of ./doc/imgs_words_en/word_336.png:('super', 0.9999073)
```

**Attention** Since the above model refers to the [DTRB](https://arxiv.org/abs/1904.01906) text recognition training and evaluation process, it is different from the ultra-lightweight Chinese recognition model training in two aspects:

- The image resolutions used during training are different. The image resolutions used for training the above models are [3, 32, 100], while for Chinese model training, in order to ensure the recognition effect of long texts, the image resolutions used during training are [ 3, 32, 320]. The default shape parameter of the predictive inference program is the image resolution used for training Chinese, i.e. [3, 32, 320]. Therefore, when inferring the above English model here, it is necessary to set the shape of the recognized image through the parameter rec_image_shape.

- Character list, the experiment in the DTRB paper is only for 26 lowercase English letters and 10 numbers, a total of 36 characters. All uppercase and lowercase characters are converted to lowercase characters, and characters not listed above are ignored and considered spaces. Therefore, there is no input character dictionary here, but a dictionary is generated by the following command. Therefore, the parameter rec_char_dict_path needs to be set during inference, which is specified as an English dictionary "./ppocr/utils/ic15_dict.txt".

```
self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
dict_character = list(self.character_str)


 ```

<a name="4-2"></a>
### 4.2 C++ Inference

After preparing the inference model, refer to the [cpp infer](../../deploy/cpp_infer/) tutorial to operate.

<a name="4-3"></a>
### 4.3 Serving

After preparing the inference model, refer to the [pdserving](../../deploy/pdserving/) tutorial for Serving deployment, including two modes: Python Serving and C++ Serving.

<a name="4-4"></a>
### 4.4 More

The STAR-Net model also supports the following inference deployment methods:

- Paddle2ONNX Inference: After preparing the inference model, refer to the [paddle2onnx](../../deploy/paddle2onnx/) tutorial.

<a name="5"></a>
## 5. FAQ

## Quote

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
