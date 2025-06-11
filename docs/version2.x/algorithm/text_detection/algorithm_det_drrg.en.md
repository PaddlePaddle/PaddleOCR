---
typora-copy-images-to: images
comments: true
---

# DRRG

## 1. Introduction

Paper:
> [Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection](https://arxiv.org/abs/2003.07493)
> Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Liu, Chang and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng
> CVPR, 2020

On the CTW1500 dataset, the text detection result is as follows:

|Model|Backbone|Configuration|Precision|Recall|Hmean|Download|
| --- | --- | --- | --- | --- | --- | --- |
| DRRG | ResNet50_vd | [configs/det/det_r50_drrg_ctw.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_r50_drrg_ctw.yml)| 89.92%|80.91%|85.18%|[trained model](https://paddleocr.bj.bcebos.com/contribution/det_r50_drrg_ctw_train.tar)|

## 2. Environment

Please prepare your environment referring to [prepare the environment](../../ppocr/environment.en.md) and [clone the repo](../../ppocr/blog/clone.en.md).

## 3. Model Training / Evaluation / Prediction

The above DRRG model is trained using the CTW1500 text detection public dataset. For the download of the dataset, please refer to [ocr_datasets](./dataset/ocr_datasets_en.md).

After the data download is complete, please refer to [Text Detection Training Tutorial](../../ppocr/model_train/detection.en.md) for training. PaddleOCR has modularized the code structure, so that you only need to **replace the configuration file** to train different detection models.

## 4. Inference and Deployment

### 4.1 Python Inference

Since the model needs to be converted to Numpy data for many times in the forward, DRRG dynamic graph to static graph is not supported.

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## 5. FAQ

## Citation

```bibtex
@inproceedings{zhang2020deep,
  title={Deep relational reasoning graph network for arbitrary shape text detection},
  author={Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Liu, Chang and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9699--9708},
  year={2020}
}
```
