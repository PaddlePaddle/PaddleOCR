---
typora-copy-images-to: images
comments: true
---

# DRRG

## 1. 算法简介

论文信息：
> [Deep Relational Reasoning Graph Network for Arbitrary Shape Text Detection](https://arxiv.org/abs/2003.07493)
> Zhang, Shi-Xue and Zhu, Xiaobin and Hou, Jie-Bo and Liu, Chang and Yang, Chun and Wang, Hongfa and Yin, Xu-Cheng
> CVPR, 2020

在CTW1500文本检测公开数据集上，算法复现效果如下：

| 模型  |骨干网络|配置文件|precision|recall|Hmean|下载链接|
|-----| --- | --- | --- | --- | --- | --- |
| DRRG | ResNet50_vd | [configs/det/det_r50_drrg_ctw.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/det/det_r50_drrg_ctw.yml)| 89.92%|80.91%|85.18%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/det_r50_drrg_ctw_train.tar)|

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

上述DRRG模型使用CTW1500文本检测公开数据集训练得到，数据集下载可参考 [ocr_datasets](../../datasets/ocr_datasets.md)。

数据下载完成后，请参考[文本检测训练教程](../../ppocr/model_train/detection.md)进行训练。PaddleOCR对代码进行了模块化，训练不同的检测模型只需要**更换配置文件**即可。

## 4. 推理部署

### 4.1 Python推理

由于模型前向运行时需要多次转换为Numpy数据进行运算，因此DRRG的动态图转静态图暂未支持。

### 4.2 C++推理

暂未支持

### 4.3 Serving服务化部署

暂未支持

### 4.4 更多推理部署

暂未支持

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
