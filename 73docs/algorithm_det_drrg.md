# DRRG

- [1. 算法简介](#1-算法简介)
- [2. 环境配置](#2-环境配置)
- [3. 模型训练、评估、预测](#3-模型训练评估预测)
- [4. 推理部署](#4-推理部署)
  - [4.1 Python推理](#41-python推理)
  - [4.2 C++推理](#42-c推理)
  - [4.3 Serving服务化部署](#43-serving服务化部署)
  - [4.4 更多推理部署](#44-更多推理部署)
- [5. FAQ](#5-faq)
- [引用](#引用)

<a name="1"></a>
## 1. 算法简介

论文信息：
> [Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection](https://arxiv.org/abs/2003.07493)
> Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Liu, Chang and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng
> CVPR, 2020

一种新颖的统一关系推理图网络用于任意形状的文本检测，一个独创的局部图构建了文本建议模型，通过卷积神经网络（CNN）和基于图关系卷积网络的深度关系推理网络（GCN），使的网络达到端到端训练。

在CTW1500文本检测公开数据集上，算法复现效果如下：

| 模型  |骨干网络|配置文件|precision|recall|Hmean|下载链接|
|-----| --- | --- | --- | --- | --- | --- |
| DRRG | ResNet50_vd | [configs/det/det_r50_drrg_ctw.yml](../../configs/det/det_r50_drrg_ctw.yml)| 89.92%|80.91%|85.18%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/det_r50_drrg_ctw_train.tar)|

<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。

```python
# 首先git官方的PaddleOCR项目，安装需要的依赖
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
```


<a name="3"></a>
## 3. 模型训练、评估、预测

上述DRRG模型使用CTW1500文本检测公开数据集训练得到，数据集下载可参考 [ocr_datasets](./dataset/ocr_datasets.md)。

PaddleOCR训练数据的默认存储路径是 PaddleOCR/train_data,如果您的磁盘上已有数据集，只需创建软链接至数据集目录：

```
# linux and mac os
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/dataset
# windows
mklink /d <path/to/paddle_ocr>/train_data/dataset <path/to/dataset>
```

数据下载完成后，请参考[文本检测训练教程](./detection.md)进行训练。PaddleOCR对代码进行了模块化，训练不同的检测模型只需要**更换配置文件**即可。以下只提供部分训练方式。

<a name="3-1"></a>
### 3.1 训练
首先下载与训练模型放到./pretrain 目录下

*如果您安装的是cpu版本，请将配置文件中的 `use_gpu` 字段修改为false*

```shell
# 单机单卡训练
python3 tools/train.py -c configs/det/det_r50_drrg_ctw.yml \
     -o Global.pretrained_model=./pretrain_models/det_r50_drrg_ctw_train

# 单机多卡训练，通过 --gpus 参数设置使用的GPU ID
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/det_r50_drrg_ctw.yml \
     -o Global.pretrained_model=./pretrain_models/det_r50_drrg_ctw_train

```

<a name="3-2"></a>
### 3.2 评估

PaddleOCR计算三个OCR检测相关的指标，分别是：Precision、Recall、Hmean（F-Score）。

训练中模型参数默认保存在`Global.save_model_dir`目录下。在评估指标时，需要设置`Global.checkpoints`指向保存的参数文件。

```shell
python3 tools/eval.py -c configs/det/det_r50_drrg_ctw.yml  -o Global.checkpoints="{path/to/weights}/best_accuracy"
```

<a name="3-3"></a>
### 3.3 预测

测试单张图像的检测效果：

```shell
python3 tools/infer_det.py -c configs/det/det_r50_drrg_ctw.yml  -o Global.infer_img="./doc/imgs_en/img_10.jpg" Global.pretrained_model={path/to/weights}/best_accuracy"
```









<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理

由于模型前向运行时需要多次转换为Numpy数据进行运算，因此DRRG的动态图转静态图暂未支持。

<a name="4-2"></a>
### 4.2 C++推理

暂未支持

<a name="4-3"></a>
### 4.3 Serving服务化部署

暂未支持

<a name="4-4"></a>
### 4.4 更多推理部署

暂未支持

<a name="5"></a>
## 5. FAQ


## 引用

```bibtex
@inproceedings{zhang2020deep,
  title={Deep relational reasoning graph network for arbitrary shape text detection},
  author={Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Liu, Chang and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9699--9708},
  year={2020}
}
```
