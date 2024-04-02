# PSENet

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
> [Shape robust text detection with progressive scale expansion network](https://arxiv.org/abs/1903.12473)
> Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai
> CVPR, 2019

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|Configuration|Precision|Recall|Hmean|Download|
| --- | --- | --- | --- | --- | --- | --- |
|PSE| ResNet50_vd | [configs/det/det_r50_vd_pse.yml](../../configs/det/det_r50_vd_pse.yml)| 85.81%    |79.53%|82.55%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar)|
|PSE| MobileNetV3| [configs/det/det_mv3_pse.yml](../../configs/det/det_mv3_pse.yml) | 82.20%    |70.48%|75.89%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar)|


<a name="2"></a>
## 2. Environment
Please prepare your environment referring to [prepare the environment](./environment_en.md) and [clone the repo](./clone_en.md).


<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

The above PSE model is trained using the ICDAR2015 text detection public dataset. For the download of the dataset, please refer to [ocr_datasets](./dataset/ocr_datasets_en.md).

After the data download is complete, please refer to [Text Detection Training Tutorial](./detection_en.md) for training. PaddleOCR has modularized the code structure, so that you only need to **replace the configuration file** to train different detection models.

<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference
First, convert the model saved in the PSE text detection training process into an inference model. Taking the model based on the Resnet50_vd backbone network and trained on the ICDAR2015 English dataset as example ([model download link](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar)), you can use the following command to convert:

```shell
python3 tools/export_model.py -c configs/det/det_r50_vd_pse.yml -o Global.pretrained_model=./det_r50_vd_pse_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_pse
```

PSE text detection model inference, to perform non-curved text detection, you can run the following commands:

```shell
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_pse/" --det_algorithm="PSE" --det_pse_box_type=quad
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![](../imgs_results/det_res_img_10_pse.jpg)

If you want to perform curved text detection, you can execute the following command:

```shell
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_pse/" --det_algorithm="PSE" --det_pse_box_type=poly
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![](../imgs_results/det_res_img_10_pse_poly.jpg)

**Note**: Since the ICDAR2015 dataset has only 1,000 training images, mainly for English scenes, the above model has very poor detection result on Chinese or curved text images.


<a name="4-2"></a>
### 4.2 C++ Inference

Since the post-processing is not written in CPP, the PSE text detection model does not support CPP inference.

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
@inproceedings{wang2019shape,
  title={Shape robust text detection with progressive scale expansion network},
  author={Wang, Wenhai and Xie, Enze and Li, Xiang and Hou, Wenbo and Lu, Tong and Yu, Gang and Shao, Shuai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9336--9345},
  year={2019}
}
```
