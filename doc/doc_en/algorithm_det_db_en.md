# DB && DB++

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
> [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
> Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang
> AAAI, 2020

> [Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion](https://arxiv.org/abs/2202.10304)
> Liao, Minghui and Zou, Zhisheng and Wan, Zhaoyi and Yao, Cong and Bai, Xiang
> TPAMI, 2022

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|Configuration|Precision|Recall|Hmean|Download|
| --- | --- | --- | --- | --- | --- | --- |
|DB|ResNet50_vd|[configs/det/det_r50_vd_db.yml](../../configs/det/det_r50_vd_db.yml)|86.41%|78.72%|82.38%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|[configs/det/det_mv3_db.yml](../../configs/det/det_mv3_db.yml)|77.29%|73.08%|75.12%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|DB++|ResNet50|[configs/det/det_r50_db++_icdar15.yml](../../configs/det/det_r50_db++_icdar15.yml)|90.89%|82.66%|86.58%|[pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams)/[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_db%2B%2B_icdar15_train.tar)|

On the TD_TR dataset, the text detection result is as follows:

|Model|Backbone|Configuration|Precision|Recall|Hmean|Download|
| --- | --- | --- | --- | --- | --- | --- |
|DB++|ResNet50|[configs/det/det_r50_db++_td_tr.yml](../../configs/det/det_r50_db++_td_tr.yml)|92.92%|86.48%|89.58%|[pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/ResNet50_dcn_asf_synthtext_pretrained.pdparams)/[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_db%2B%2B_td_tr_train.tar)|

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
First, convert the model saved in the DB text detection training process into an inference model. Taking the model based on the Resnet50_vd backbone network and trained on the ICDAR2015 English dataset as example ([model download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)), you can use the following command to convert:

```shell
python3 tools/export_model.py -c configs/det/det_r50_vd_db.yml -o Global.pretrained_model=./det_r50_vd_db_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_db
```

DB text detection model inference, you can execute the following command:

```shell
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_db/"
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![](../imgs_results/det_res_img_10_db.jpg)

**Note**: Since the ICDAR2015 dataset has only 1,000 training images, mainly for English scenes, the above model has very poor detection result on Chinese text images.


<a name="4-2"></a>
### 4.2 C++ Inference

With the inference model prepared, refer to the [cpp infer](../../deploy/cpp_infer/) tutorial for C++ inference.

<a name="4-3"></a>
### 4.3 Serving

With the inference model prepared, refer to the [pdserving](../../deploy/pdserving/) tutorial for service deployment by Paddle Serving.

<a name="4-4"></a>
### 4.4 More

More deployment schemes supported for DB:

- Paddle2ONNX: with the inference model prepared, please refer to the [paddle2onnx](../../deploy/paddle2onnx/) tutorial.

<a name="5"></a>
## 5. FAQ


## Citation

```bibtex
@inproceedings{liao2020real,
  title={Real-time scene text detection with differentiable binarization},
  author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={11474--11481},
  year={2020}
}

@article{liao2022real,
  title={Real-Time Scene Text Detection with Differentiable Binarization and Adaptive Scale Fusion},
  author={Liao, Minghui and Zou, Zhisheng and Wan, Zhaoyi and Yao, Cong and Bai, Xiang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```
