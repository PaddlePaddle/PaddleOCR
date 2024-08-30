---
typora-copy-images-to: images
comments: true
---

# FCENet

## 1. Introduction

Paper:
> [Fourier Contour Embedding for Arbitrary-Shaped Text Detection](https://arxiv.org/abs/2104.10442)
> Yiqin Zhu and Jianyong Chen and Lingyu Liang and Zhanghui Kuang and Lianwen Jin and Wayne Zhang
> CVPR, 2021

On the CTW1500 dataset, the text detection result is as follows:

|Model|Backbone|Configuration|Precision|Recall|Hmean|Download|
| --- | --- | --- | --- | --- | --- | --- |
| FCE | ResNet50_dcn | [configs/det/det_r50_vd_dcn_fce_ctw.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_r50_vd_dcn_fce_ctw.yml)| 88.39%|82.18%|85.27%|[trained model](https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar)|

## 2. Environment

Please prepare your environment referring to [prepare the environment](../../ppocr/environment.en.md) and [clone the repo](../../ppocr/blog/clone.en.md).

## 3. Model Training / Evaluation / Prediction

The above FCE model is trained using the CTW1500 text detection public dataset. For the download of the dataset, please refer to [ocr_datasets](./dataset/ocr_datasets_en.md).

After the data download is complete, please refer to [Text Detection Training Tutorial](../../ppocr/model_train/detection.en.md) for training. PaddleOCR has modularized the code structure, so that you only need to **replace the configuration file** to train different detection models.

## 4. Inference and Deployment

### 4.1 Python Inference

First, convert the model saved in the FCE text detection training process into an inference model. Taking the model based on the Resnet50_vd_dcn backbone network and trained on the CTW1500 English dataset as example ([model download link](https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar)), you can use the following command to convert:

```bash linenums="1"
python3 tools/export_model.py -c configs/det/det_r50_vd_dcn_fce_ctw.yml -o Global.pretrained_model=./det_r50_dcn_fce_ctw_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_fce
```

FCE text detection model inference, to perform non-curved text detection, you can run the following commands:

```bash linenums="1"
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_fce/" --det_algorithm="FCE" --det_fce_box_type=quad
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![img](./images/det_res_img_10_fce.jpg)

If you want to perform curved text detection, you can execute the following command:

```bash linenums="1"
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img623.jpg" --det_model_dir="./inference/det_fce/" --det_algorithm="FCE" --det_fce_box_type=poly
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![img](./images/det_res_img623_fce.jpg)

**Note**: Since the CTW1500 dataset has only 1,000 training images, mainly for English scenes, the above model has very poor detection result on Chinese or curved text images.

### 4.2 C++ Inference

Since the post-processing is not written in CPP, the FCE text detection model does not support CPP inference.

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## 5. FAQ

## Citation

```bibtex
@InProceedings{zhu2021fourier,
  title={Fourier Contour Embedding for Arbitrary-Shaped Text Detection},
  author={Yiqin Zhu and Jianyong Chen and Lingyu Liang and Zhanghui Kuang and Lianwen Jin and Wayne Zhang},
  year={2021},
  booktitle = {CVPR}
}
```
