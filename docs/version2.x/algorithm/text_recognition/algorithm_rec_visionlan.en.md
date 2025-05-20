---
comments: true
---

# VisionLAN

## 1. Introduction

Paper:
> [From Two to One: A New Scene Text Recognizer with Visual Language Modeling Network](https://arxiv.org/abs/2108.09661)
> Yuxin Wang, Hongtao Xie, Shancheng Fang, Jing Wang, Shenggao Zhu, Yongdong Zhang
> ICCV, 2021

Using MJSynth and SynthText two text recognition datasets for training, and evaluating on IIIT, SVT, IC13, IC15, SVTP, CUTE datasets, the algorithm reproduction effect is as follows:

|Model|Backbone|config|Acc|Download link|
| --- | --- | --- | --- | --- |
|VisionLAN|ResNet45|[rec_r45_visionlan.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_r45_visionlan.yml)|90.30%|[model link](https://paddleocr.bj.bcebos.com/VisionLAN/rec_r45_visionlan_train.tar)|

## 2. Environment

Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md)to clone the project code.

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

### Training

Specifically, after the data preparation is completed, the training can be started. The training command is as follows:

```bash linenums="1"
# Single GPU training (long training period, not recommended)
python3 tools/train.py -c configs/rec/rec_r45_visionlan.yml

# Multi GPU training, specify the gpu number through the --gpus parameter
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r45_visionlan.yml
```

### Evaluation

```bash linenums="1"
# GPU evaluation
python3 tools/eval.py -c configs/rec/rec_r45_visionlan.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### Prediction

```bash linenums="1"
# The configuration file used for prediction must match the training
python3 tools/infer_rec.py -c configs/rec/rec_r45_visionlan.yml -o Global.infer_img='./doc/imgs_words/en/word_2.png' Global.pretrained_model=./rec_r45_visionlan_train/best_accuracy
```

## 4. Inference and Deployment

### 4.1 Python Inference

First, the model saved during the VisionLAN text recognition training process is converted into an inference model. ( [Model download link](https://paddleocr.bj.bcebos.com/VisionLAN/rec_r45_visionlan_train.tar)) ), you can use the following command to convert:

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_r45_visionlan.yml -o Global.pretrained_model=./rec_r45_visionlan_train/best_accuracy Global.save_inference_dir=./inference/rec_r45_visionlan/
```

**Note:**

- If you are training the model on your own dataset and have modified the dictionary file, please pay attention to modify the `character_dict_path` in the configuration file to the modified dictionary file.
- If you modified the input size during training, please modify the `infer_shape` corresponding to VisionLAN in the `tools/export_model.py` file.

After the conversion is successful, there are three files in the directory:

```text linenums="1"
./inference/rec_r45_visionlan/
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

For VisionLAN text recognition model inference, the following commands can be executed:

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words/en/word_2.png' --rec_model_dir='./inference/rec_r45_visionlan/' --rec_algorithm='VisionLAN' --rec_image_shape='3,64,256' --rec_char_dict_path='./ppocr/utils/ic15_dict.txt' --use_space_char=False
```

![img](./images/word_10.png)

After executing the command, the prediction result (recognized text and score) of the image above is printed to the screen, an example is as follows:
The result is as follows:

```bash linenums="1"
Predicts of ./doc/imgs_words/en/word_2.png:('yourself', 0.9999493)
```

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

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
