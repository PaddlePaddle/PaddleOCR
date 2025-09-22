---
comments: true
---

# KIE Algorithm - SDMGR

## 1. Introduction

Paper:

> [Spatial Dual-Modality Graph Reasoning for Key Information Extraction](https://arxiv.org/abs/2103.14470)
>
> Hongbin Sun and Zhanghui Kuang and Xiaoyu Yue and Chenhao Lin and Wayne Zhang
>
> 2021

On wildreceipt dataset, the algorithm reproduction Hmean is as follows.

|Model|Backbone |Cnnfig|Hmean|Download link|
| --- | --- | --- | --- | --- |
|SDMGR|VGG6|[configs/kie/sdmgr/kie_unet_sdmgr.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/kie/sdmgr/kie_unet_sdmgr.yml)|86.70%|[trained model]( https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar)/[inference model(coming soon)]()|

## 2. 环境配置

Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md)to clone the project code.

## 3. Model Training / Evaluation / Prediction

SDMGR is a key information extraction algorithm that classifies each detected textline into predefined categories, such as order ID, invoice number, amount, etc.

The training and test data are collected in the wildreceipt dataset, use following command to downloaded the dataset.

```bash linenums="1"
wget https://paddleocr.bj.bcebos.com/ppstructure/dataset/wildreceipt.tar && tar xf wildreceipt.tar
```

Create dataset soft link to `PaddleOCR/train_data` directory.

```bash linenums="1"
cd PaddleOCR/ && mkdir train_data && cd train_data
ln -s ../../wildreceipt ./
```

### 3.1 Model training

The config file is `configs/kie/sdmgr/kie_unet_sdmgr.yml`， the default dataset path is `train_data/wildreceipt`.

Use the following command to train the model.

```bash linenums="1"
python3 tools/train.py -c configs/kie/sdmgr/kie_unet_sdmgr.yml -o Global.save_model_dir=./output/kie/
```

### 3.2 Model evaluation

Use the following command to evaluate the model:

```bash linenums="1"
python3 tools/eval.py -c configs/kie/sdmgr/kie_unet_sdmgr.yml -o Global.checkpoints=./output/kie/best_accuracy
```

An example of output information is shown below.

```bash linenums="1"
[2022/08/10 05:22:23] ppocr INFO: metric eval ***************
[2022/08/10 05:22:23] ppocr INFO: hmean:0.8670120239257812
[2022/08/10 05:22:23] ppocr INFO: fps:10.18816520530961
```

### 3.3 Model prediction

Use the following command to load the model and predict. During the prediction, the text file storing the image path and OCR information needs to be loaded in advance. Use `Global.infer_img` to assign.

```bash linenums="1"
python3 tools/infer_kie.py -c configs/kie/kie_unet_sdmgr.yml -o Global.checkpoints=kie_vgg16/best_accuracy  Global.infer_img=./train_data/wildreceipt/1.txt
```

The visualization results and texts are saved in the `./output/sdmgr_kie/` directory by default. The results are as follows.

![img](./images/sdmgr_result.png)

## 4. Inference and Deployment

### 4.1 Python Inference

Not supported

### 4.2 C++ Inference

Not supported

### 4.3 Serving

Not supported

### 4.4 More

Not supported

## 5. FAQ

## Citation

```bibtex
@misc{sun2021spatial,
      title={Spatial Dual-Modality Graph Reasoning for Key Information Extraction},
      author={Hongbin Sun and Zhanghui Kuang and Xiaoyu Yue and Chenhao Lin and Wayne Zhang},
      year={2021},
      eprint={2103.14470},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
