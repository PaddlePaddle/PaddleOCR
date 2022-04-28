# SAST

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
- [5. FAQ](#5)

<a name="1"></a>
## 1. Introduction

Paper:
> [A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning](https://arxiv.org/abs/1908.05498)
> Wang, Pengfei and Zhang, Chengquan and Qi, Fei and Huang, Zuming and En, Mengyi and Han, Junyu and Liu, Jingtuo and Ding, Errui and Shi, Guangming
> ACM MM, 2019

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|Configuration|Precision|Recall|Hmean|Download|
| --- | --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|[configs/det/det_r50_vd_sast_icdar15.yml](../../configs/det/det_r50_vd_sast_icdar15.yml)|91.39%|83.77%|87.42%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|


On the Total-text dataset, the text detection result is as follows:

|Model|Backbone|Configuration|Precision|Recall|Hmean|Download|
| --- | --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|[configs/det/det_r50_vd_sast_totaltext.yml](../../configs/det/det_r50_vd_sast_totaltext.yml)|89.63%|78.44%|83.66%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|


<a name="2"></a>
## 2. Environment
Please prepare your environment referring to [prepare the environment](./environment_en.md) and [clone the repo](./clone_en.md).


<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

Please refer to [text detection training tutorial](./detection_en.md). PaddleOCR has modularized the code structure, so that you only need to **replace the configuration file** to train different detection models.

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, convert the model saved in the SAST text detection training process into an inference model. Taking the model based on the Resnet50_vd backbone network and trained on the ICDAR2015 English dataset as example ([model download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)), you can use the following command to convert:

```shell
python3 tools/export_model.py -c configs/det/det_r50_vd_sast_icdar15.yml -o Global.pretrained_model=./det_r50_vd_sast_icdar15_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_sast
```

SAST text detection model inference, you can execute the following command:

```shell
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_sast/"
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![](../imgs_results/det_res_img_10_sast.jpg)

**Note**: Since the ICDAR2015 dataset has only 1,000 training images, mainly for English scenes, the above model has very poor detection result on Chinese text images.


<a name="5"></a>
## 5. FAQ


## Citation

```bibtex
@inproceedings{wang2019single,
  title={A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning},
  author={Wang, Pengfei and Zhang, Chengquan and Qi, Fei and Huang, Zuming and En, Mengyi and Han, Junyu and Liu, Jingtuo and Ding, Errui and Shi, Guangming},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={1277--1285},
  year={2019}
}
```
