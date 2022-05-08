English | [简体中文](../doc_ch/PP-OCRv3_introduction.md)


# PP-OCRv3

- [1. Introduction](#1)
- [2. Optimization for Text Detection Model](#2)
- [3. Optimization for Text Recognition Model](#3)
- [4. End-to-end Evaluation](#4)


<a name="1"></a>

## 1. Introduction


PP-OCRv3 is further upgraded on the basis of PP-OCRv2. The overall framework of PP-OCRv3 is same as that of PP-OCRv2. The text detection model and text recognition model are further optimized, respectively. Specifically, the detection network is still optimized based on DBNet, and base model of recognition network is replaced from CRNN to [SVTR](https://arxiv.org/abs/2205.00159, which is recorded in IJCAI 2022. The block diagram of the PP-OCRv3 system is as follows (the pink box is the new policies of PP-OCRv3):

<div align="center">
    <img src="../ppocrv3_framework.png" width="800">
</div>

There are 9 optimization tricks for text detection and recognition models in PP-OCRv3, which is as follows.

- Text detection:
    - LK-PAN: PAN structure with large receptive field;
    - DML: mutual learning strategy for teacher model;
    - RSE-FPN: FPN structure of residual attention mechanism;

- Text recognition:
    - SVTR_LCNet: Light-weight text recognition network;
    - GTC: training strategy using Attention to guide CTC;
    - TextConAug: A data augmentation strategy for mining textual context information;
    - TextRotNet: self-supervised strategy to optimize the pretrained model;
    - UDML: unified deep mutual learning strategy;
    - UIM: unlabeled data mining strategy.

From the effect point of view, when the speed is comparable, the accuracy of various scenes has been greatly improved:

Finally, in the case of comparable inference speed, PP-OCRv3 significantly outperforms PP-OCRv2 in terms of accuracy in multiple scenarios.

- In Chinese scenarios, PP-OCRv3 outperforms PP-OCRv2 by more than 5%.
- In English scenarios, PP-OCRv3 outperforms PP-OCRv2 by more than 11%.
- In multi-language scenarios, more than 80 languages and corresponding models are optimized, the average accuracy is increased by over 5%.


<a name="2"></a>

## 2. Optimization for Text Detection Model



<a name="3"></a>

## 3. Optimization for Text Recognition Model


<a name="4"></a>

## 4. End-to-end Evaluation

After the optimization strategies mentioned above, under he condition of comparable speed, PP-OCRv3 outperforms PP-OCRv2 by 5% in terms of end-to-end Hmean for Chinese scenarios. The specific metrics are shown as follows.

| Model | Hmean |  Model Size (M) | Time Cost (CPU, ms) | Time Cost (T4 GPU, ms) |
|-----|-----|--------|----| --- |
| PP-OCR mobile | 50.3% | 8.1 | 356  | 116 |
| PP-OCR server | 57.0% | 155.1 | 1056 | 200 |
| PP-OCRv2 | 57.6% | 11.6 | 330 | 111 |
| PP-OCRv3 | 62.9% | 15.6 | 331 | 86.64 |


Test environment:
- CPU: Intel Gold 6148, and MKLDNN acceleration is enabled during CPU inference.


In addition to updating the recognition model for Chinese, the recognition model for English is also optimized with an increasement of 11% for end-to-end Hmean, which is shown as follows.

| Model | Recall |  Precision | Hmean |
|-----|-----|--------|----|
| PP-OCR_en | 38.99% | 45.91% | 42.17%  |
| PP-OCRv3_en | 50.95% | 55.53% | 53.14% |

At the same time, more than 80 language recognition models that have been supported have been upgraded this time, and the recognition accuracy of the four language families with evaluation sets has increased by more than 5% on average, which is shown as follows.

| Model | Latin | Arabic | Japanese | Korean |
|-----|-----|--------|----| --- |
| PP-OCR_mul | 69.6% | 40.5% | 38.5% | 55.4% |
| PP-OCRv3_mul | 75.2% | 45.37% | 45.8% | 60.1% |
