---
typora-copy-images-to: images
comments: true
---

# SAST

## 1. 算法简介

论文信息：
> [A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning](https://arxiv.org/abs/1908.05498)
> Wang, Pengfei and Zhang, Chengquan and Qi, Fei and Huang, Zuming and En, Mengyi and Han, Junyu and Liu, Jingtuo and Ding, Errui and Shi, Guangming
> ACM MM, 2019

在ICDAR2015文本检测公开数据集上，算法复现效果如下：

|模型|骨干网络|配置文件|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|[configs/det/det_r50_vd_sast_icdar15.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_r50_vd_sast_icdar15.yml)|91.39%|83.77%|87.42%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|

在Total-text文本检测公开数据集上，算法复现效果如下：

|模型|骨干网络|配置文件|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|[configs/det/det_r50_vd_sast_totaltext.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_r50_vd_sast_totaltext.yml)|89.63%|78.44%|83.66%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

请参考[文本检测训练教程](../../ppocr/model_train/detection.md)。PaddleOCR对代码进行了模块化，训练不同的检测模型只需要**更换配置文件**即可。

## 4. 推理部署

### 4.1 Python推理

#### (1). 四边形文本检测模型（ICDAR2015）

首先将SAST文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在ICDAR2015英文数据集训练的模型为例([模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar))，可以使用如下命令进行转换：

```bash linenums="1"
python3 tools/export_model.py -c configs/det/det_r50_vd_sast_icdar15.yml -o Global.pretrained_model=./det_r50_vd_sast_icdar15_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_sast_ic15
```

**SAST文本检测模型推理，需要设置参数`--det_algorithm="SAST"`**，可以执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_det.py --det_algorithm="SAST" --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_sast_ic15/"
```

可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![img](./images/det_res_img_10_sast.jpg)

#### (2). 弯曲文本检测模型（Total-Text）

首先将SAST文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在Total-Text英文数据集训练的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar))，可以使用如下命令进行转换：

```bash linenums="1"
python3 tools/export_model.py -c configs/det/det_r50_vd_sast_totaltext.yml -o Global.pretrained_model=./det_r50_vd_sast_totaltext_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_sast_tt
```

SAST文本检测模型推理，需要设置参数`--det_algorithm="SAST"`，同时，还需要增加参数`--det_box_type=poly`，可以执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_det.py --det_algorithm="SAST" --image_dir="./doc/imgs_en/img623.jpg" --det_model_dir="./inference/det_sast_tt/" --det_box_type='poly'
```

可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![img](./images/det_res_img623_sast.jpg)

**注意**：本代码库中，SAST后处理Locality-Aware NMS有python和c++两种版本，c++版速度明显快于python版。由于c++版本nms编译版本问题，只有python3.5环境下会调用c++版nms，其他情况将调用python版nms。

### 4.2 C++推理

暂未支持

### 4.3 Serving服务化部署

暂未支持

### 4.4 更多推理部署

暂未支持

## 5. FAQ

## 引用

```bibtex
@inproceedings{wang2019single,
  title={A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning},
  author={Wang, Pengfei and Zhang, Chengquan and Qi, Fei and Huang, Zuming and En, Mengyi and Han, Junyu and Liu, Jingtuo and Ding, Errui and Shi, Guangming},
  booktitle={Proceedings of the 27th ACM International Conference on Multimedia},
  pages={1277--1285},
  year={2019}
}
```
