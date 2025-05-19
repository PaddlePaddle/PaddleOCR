---
comments: true
---


# Table Recognition Datasets

Here are the commonly used table recognition datasets, which are being updated continuously. Welcome to contribute datasets~

## Dataset Summary

| dataset | Image download link | PPOCR format annotation download link |
|---|---|---|
| PubTabNet |<https://github.com/ibm-aur-nlp/PubTabNet>| jsonl format, which can be loaded directly with [pubtab_dataset.py](../../../ppocr/data/pubtab_dataset.py) |
| TAL Table Recognition Competition Dataset |<https://ai.100tal.com/dataset>| jsonl format, which can be loaded directly with [pubtab_dataset.py](../../../ppocr/data/pubtab_dataset.py) |
| WTW Chinese scene table dataset |<https://github.com/wangwen-whu/WTW-Dataset>| Conversion is required to load with [pubtab_dataset.py](../../../ppocr/data/pubtab_dataset.py)|

## 1. PubTabNet

- **Data Introduction**：The training set of the PubTabNet dataset contains 500,000 images and the validation set contains 9000 images. Part of the image visualization is shown below.

    ![](./images/table_PubTabNet_demo/PMC524509_007_00.png)

    ![](./images/table_PubTabNet_demo/PMC535543_007_01.png)

- **illustrate**：When using this dataset, the [CDLA-Permissive](https://cdla.io/permissive-1-0/) protocol is required.

## 2. TAL Table Recognition Competition Dataset

- **Data Introduction**：The training set of the TAL table recognition competition dataset contains 16,000 images. The validation set does not give trainable annotations.

    ![](./images/table_tal_demo/1.jpg)

    ![](./images/table_tal_demo/2.jpg)

## 3. WTW Chinese scene table dataset

- **Data Introduction**：The WTW Chinese scene table dataset consists of two parts: table detection and table data. The dataset contains images of two scenes, scanned and photographed.

    ![img](./images/20210816_210413.gif)
