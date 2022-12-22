# VisionLAN

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
> [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://arxiv.org/abs/2108.09661)
> Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, Yongdong Zhang
> ICCV, 2021

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Model|Backbone|config|Acc|Download link|
| --- | --- | --- | --- | --- |
|VisionLAN|ResNet45|[rec_r45_visionlan.yml](../../configs/rec/rec_r45_visionlan.yml)|90.30%|[预训练、训练模型](https://paddleocr.bj.bcebos.com/VisionLAN/rec_r45_visionlan_train.tar)|

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
python3 tools/train.py -c configs/rec/rec_r45_visionlan.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r45_visionlan.yml
```

Evaluation:

```
# GPU evaluation
python3 tools/eval.py -c configs/rec/rec_r45_visionlan.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_r45_visionlan.yml -o Global.infer_img='./doc/imgs_words/en/word_2.png' Global.pretrained_model=./rec_r45_visionlan_train/best_accuracy
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, the model saved during the VisionLAN text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/VisionLAN/rec_r45_visionlan_train.tar)) ), you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_r45_visionlan.yml -o Global.pretrained_model=./rec_r45_visionlan_train/best_accuracy Global.save_inference_dir=./inference/rec_r45_visionlan/
```

**Note:**
- If you are training the model on your own dataset and have modified the dictionary file, please pay attention to modify the `character_dict_path` in the configuration file to the modified dictionary file.
- If you modified the input size during training, please modify the `infer_shape` corresponding to VisionLAN in the `tools/export_model.py` file.

After the conversion is successful, there are three files in the directory:
```
./inference/rec_r45_visionlan/
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```


For VisionLAN text recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words/en/word_2.png' --rec_model_dir='./inference/rec_r45_visionlan/' --rec_algorithm='VisionLAN' --rec_image_shape='3,64,256' --rec_char_dict_path='./ppocr/utils/ic15_dict.txt' --use_space_char=False
```

![](../imgs_words/en/word_2.png)

After executing the command, the prediction result (recognized text and score) of the image above is printed to the screen, an example is as follows:
The result is as follows:
```shell
Predicts of ./doc/imgs_words/en/word_2.png:('yourself', 0.9999493)
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

1. Note that the MJSynth and SynthText datasets come from [VisionLAN repo](https://github.com/wangyuxin87/VisionLAN).
2. We use the pre-trained model provided by the VisionLAN authors for finetune training. The dictionary for the pre-trained model is 'ppocr/utils/ic15_dict.txt'.

## Citation

```bibtex
@inproceedings{wang2021two,
  title={From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network},
  author={Wang, Yuxin and Xie, Hongtao and Fang, Shancheng and Wang, Jing and Zhu, Shenggao and Zhang, Yongdong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14194--14203},
  year={2021}
}
```
