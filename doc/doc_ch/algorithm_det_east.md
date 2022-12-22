# EAST

- [1. 算法简介](#1)
- [2. 环境配置](#2)
- [3. 模型训练、评估、预测](#3)
    - [3.1 训练](#3-1)
    - [3.2 评估](#3-2)
    - [3.3 预测](#3-3)
- [4. 推理部署](#4)
    - [4.1 Python推理](#4-1)
    - [4.2 C++推理](#4-2)
    - [4.3 Serving服务化部署](#4-3)
    - [4.4 更多推理部署](#4-4)
- [5. FAQ](#5)

<a name="1"></a>
## 1. 算法简介

论文信息：
> [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155)
> Xinyu Zhou, Cong Yao, He Wen, Yuzhi Wang, Shuchang Zhou, Weiran He, Jiajun Liang
> CVPR, 2017


在ICDAR2015文本检测公开数据集上，算法复现效果如下：

|模型|骨干网络|配置文件|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- | --- |
|EAST|ResNet50_vd| [det_r50_vd_east.yml](../../configs/det/det_r50_vd_east.yml)|88.71%|    81.36%|    84.88%|    [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)|
|EAST|MobileNetV3|[det_mv3_east.yml](../../configs/det/det_mv3_east.yml) | 78.20%|    79.10%|    78.65%|    [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar)|




<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

上表中的EAST训练模型使用ICDAR2015文本检测公开数据集训练得到，数据集下载可参考 [ocr_datasets](./dataset/ocr_datasets.md)。

数据下载完成后，请参考[文本检测训练教程](./detection.md)进行训练。PaddleOCR对代码进行了模块化，训练不同的检测模型只需要**更换配置文件**即可。


<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理

首先将EAST文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在ICDAR2015英文数据集训练的模型为例（[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar))，可以使用如下命令进行转换：

```shell
python3 tools/export_model.py -c configs/det/det_r50_vd_east.yml -o Global.pretrained_model=./det_r50_vd_east_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_r50_east/
```

EAST文本检测模型推理，需要设置参数--det_algorithm="EAST"，执行预测：
```shell
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_r50_east/" --det_algorithm="EAST"
```

可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。

![](../imgs_results/det_res_img_10_east.jpg)

<a name="4-2"></a>
### 4.2 C++推理

由于后处理暂未使用CPP编写，EAST文本检测模型暂不支持CPP推理。

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
@inproceedings{zhou2017east,
  title={East: an efficient and accurate scene text detector},
  author={Zhou, Xinyu and Yao, Cong and Wen, He and Wang, Yuzhi and Zhou, Shuchang and He, Weiran and Liang, Jiajun},
  booktitle={Proceedings of the IEEE conference on Computer Vision and Pattern Recognition},
  pages={5551--5560},
  year={2017}
}
```
