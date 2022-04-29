English | [简体中文](../doc_ch/ppocr_introduction.md)

# PP-OCR

- [1. Introduction](#1)
- [2. Features](#2)
- [3. Benchmark](#3)
- [4. Visualization](#4)
- [5. Tutorial](#5)
    - [5.1 Quick start](#51)
    - [5.2 Model training / compression / deployment](#52)
- [6. Model zoo](#6)


<a name="1"></a>
## 1. Introduction

PP-OCR is a self-developed practical ultra-lightweight OCR system, which is slimed and optimized based on the reimplemented [academic algorithms](algorithm_en.md), considering the balance between **accuracy** and **speed**.

PP-OCR is a two-stage OCR system, in which the text detection algorithm is [DB](algorithm_det_db_en.md), and the text recognition algorithm is [CRNN](algorithm_rec_crnn_en.md). Besides, a [text direction classifier](angle_class_en.md) is added between the detection and recognition modules to deal with text in different directions.

PP-OCR pipeline is as follows:

<div align="center">
    <img src="../ppocrv2_framework.jpg" width="800">
</div>


PP-OCR system is in continuous optimization. At present, PP-OCR and PP-OCRv2 have been released:

[1] PP-OCR adopts 19 effective strategies from 8 aspects including backbone network selection and adjustment, prediction head design, data augmentation, learning rate transformation strategy, regularization parameter selection, pre-training model use, and automatic model tailoring and quantization to optimize and slim down the models of each module (as shown in the green box above). The final results are an ultra-lightweight Chinese and English OCR model with an overall size of 3.5M and a 2.8M English digital OCR model. For more details, please refer to the PP-OCR technical article (https://arxiv.org/abs/2009.09941).

[2] On the basis of PP-OCR, PP-OCRv2 is further optimized in five aspects. The detection model adopts CML(Collaborative Mutual Learning) knowledge distillation strategy and CopyPaste data expansion strategy. The recognition model adopts LCNet lightweight backbone network, U-DML knowledge distillation strategy and enhanced CTC loss function improvement (as shown in the red box above), which further improves the inference speed and prediction effect. For more details, please refer to the technical report of PP-OCRv2 (https://arxiv.org/abs/2109.03144).

<a name="2"></a>
## 2. Features

- Ultra lightweight PP-OCRv2 series models: detection (3.1M) + direction classifier (1.4M) + recognition 8.5M) = 13.0M
- Ultra lightweight PP-OCR mobile series models: detection (3.0M) + direction classifier (1.4M) + recognition (5.0M) = 9.4M
- General PP-OCR server series models: detection (47.1M) + direction classifier (1.4M) + recognition (94.9M) = 143.4M
- Support Chinese, English, and digit recognition, vertical text recognition, and long text recognition
- Support multi-lingual recognition: about 80 languages like Korean, Japanese, German, French, etc

<a name="3"></a>
## 3. benchmark

For the performance comparison between PP-OCR series models, please check the [benchmark](./benchmark_en.md) documentation.

<a name="4"></a>
## 4. Visualization [more](./visualization.md)

<details open>
<summary>PP-OCRv2 English model</summary>

<div align="center">
    <img src="../imgs_results/ch_ppocr_mobile_v2.0/img_12.jpg" width="800">
</div>

</details>

<details open>
<summary>PP-OCRv2 Chinese model</summary>

<div align="center">
      <img src="../imgs_results/ch_ppocr_mobile_v2.0/test_add_91.jpg" width="800">
      <img src="../imgs_results/ch_ppocr_mobile_v2.0/00018069.jpg" width="800">
</div>
<div align="center">
    <img src="../imgs_results/ch_ppocr_mobile_v2.0/00056221.jpg" width="800">
    <img src="../imgs_results/ch_ppocr_mobile_v2.0/rotate_00052204.jpg" width="800">
</div>

</details>

<details open>
<summary>PP-OCRv2 Multilingual model</summary>

<div align="center">
    <img src="../imgs_results/french_0.jpg" width="800">
    <img src="../imgs_results/korean.jpg" width="800">
</div>

</details>


<a name="5"></a>
## 5. Tutorial

<a name="51"></a>
### 5.1 Quick start

- You can also quickly experience the ultra-lightweight OCR : [Online Experience](https://www.paddlepaddle.org.cn/hub/scene/ocr)
- Mobile DEMO experience (based on EasyEdge and Paddle-Lite, supports iOS and Android systems): [Sign in to the website to obtain the QR code for  installing the App](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)
- One line of code quick use: [Quick Start](./quickstart_en.md)

<a name="52"></a>
### 5.2 Model training / compression / deployment

For more tutorials, including model training, model compression, deployment, etc., please refer to [tutorials](../../README.md#Tutorials)。

<a name="6"></a>
## 6. Model zoo

## PP-OCR Series Model List（Update on 2022.04.28）

| Model introduction                                           | Model name                   | Recommended scene | Detection model                                              | Direction classifier                                         | Recognition model                                            |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Chinese and English ultra-lightweight PP-OCRv3 model（16.2M）     | ch_PP-OCRv3_xx          | Mobile & Server | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
| English ultra-lightweight PP-OCRv3 model（13.4M）     | en_PP-OCRv3_xx          | Mobile & Server | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/ch_ppocr_mobile_v2.0_cls_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
| Chinese and English ultra-lightweight PP-OCRv2 model（11.6M） |  ch_PP-OCRv2_xx |Mobile & Server|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar)| [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar)|
| Chinese and English ultra-lightweight PP-OCR model (9.4M)       | ch_ppocr_mobile_v2.0_xx      | Mobile & server   |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar)      |
| Chinese and English general PP-OCR model (143.4M)               | ch_ppocr_server_v2.0_xx      | Server            |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)    |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar)    |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar)  |


For more model downloads (including multiple languages), please refer to [PP-OCR series model downloads](./models_list_en.md).

For a new language request, please refer to [Guideline for new language_requests](../../README.md#language_requests).
