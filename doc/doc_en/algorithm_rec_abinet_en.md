# ABINet

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
> [ABINet: Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.pdf)
> Shancheng Fang and Hongtao Xie and Yuxin Wang and Zhendong Mao and Yongdong Zhang
> CVPR, 2021

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Model|Backbone|config|Acc|Download link|
| --- | --- | --- | --- | --- |
|ABINet|ResNet45|[rec_r45_abinet.yml](../../configs/rec/rec_r45_abinet.yml)|90.75%|[pretrained & trained model](https://paddleocr.bj.bcebos.com/rec_r45_abinet_train.tar)|

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
python3 tools/train.py -c configs/rec/rec_r45_abinet.yml

#Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r45_abinet.yml
```

Evaluation:

```
# GPU evaluation
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r45_abinet.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

Prediction:

```
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_r45_abinet.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_r45_abinet_train/best_accuracy
```

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, the model saved during the ABINet text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/rec_r45_abinet_train.tar)) ), you can use the following command to convert:

```
python3 tools/export_model.py -c configs/rec/rec_r45_abinet.yml -o Global.pretrained_model=./rec_r45_abinet_train/best_accuracy  Global.save_inference_dir=./inference/rec_r45_abinet
```

**Note:**
- If you are training the model on your own dataset and have modified the dictionary file, please pay attention to modify the `character_dict_path` in the configuration file to the modified dictionary file.
- If you modified the input size during training, please modify the `infer_shape` corresponding to ABINet in the `tools/export_model.py` file.

After the conversion is successful, there are three files in the directory:
```
/inference/rec_r45_abinet/
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```


For ABINet text recognition model inference, the following commands can be executed:

```
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_r45_abinet/' --rec_algorithm='ABINet' --rec_image_shape='3,32,128' --rec_char_dict_path='./ppocr/utils/ic15_dict.txt'
```

![](../imgs_words_en/word_10.png)

After executing the command, the prediction result (recognized text and score) of the image above is printed to the screen, an example is as follows:
The result is as follows:
```shell
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9999995231628418)
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

1. Note that the MJSynth and SynthText datasets come from [ABINet repo](https://github.com/FangShancheng/ABINet).
2. We use the pre-trained model provided by the ABINet authors for finetune training.

## Citation

```bibtex
@article{Fang2021ABINet,
  title     = {ABINet: Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition},
  author    = {Shancheng Fang and Hongtao Xie and Yuxin Wang and Zhendong Mao and Yongdong Zhang},
  booktitle = {CVPR},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.06495},
  pages     = {7098-7107}
}
```
