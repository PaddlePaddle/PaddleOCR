  [原项目说明文档](README_ch.md)

<p align="center">
 <img src="./doc/PaddleOCR_log.png" align="middle" width = "600"/>
<p align="center">
<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/pypi/format/PaddleOCR?color=c77"></a>
    <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
</p>
# 简介

本项目主要继承自[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)项目。

# 提升

- [x] 多语言文字识别模型优化训练（PP-OCRv30）
- [x] 基于文本行的多语言文字检测模型训练（PP-OCRv30）

# 模型

## **文字检测**

|     模型名称     |                   模型简介                   |                           配置文件                           | 推理模型大小 | 精度   |
| :--------------: | :------------------------------------------: | :----------------------------------------------------------: | :----------: | ------ |
| ch_PP-OCRv30_det | 文本行检测大模型，支持中英、多语言文本行检测 | [ch_PP-OCRv30_det_dml.yml](./configs/det/ch_PP-OCRv30/ch_PP-OCRv30_det_dml.yml) |     122M     | 89.87% |

## 文字识别

| 模型名称 | 模型简介 | 配置文件 | 推理模型大小 | 精度 |
| :------: | :------: | :------: | :----------: | ---- |
|          |          |          |              |      |

## 📄 License
This project is released under <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>
