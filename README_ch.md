[English](README.md) | 简体中文

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

## 简介

PaddleOCR旨在打造一套丰富、领先、且实用的OCR工具库，助力开发者训练出更好的模型，并应用落地。

## 近期更新

- 2021.12.21《动手学OCR · 十讲》课程开讲，12月21日起每晚八点半线上授课！[免费报名地址](https://aistudio.baidu.com/aistudio/course/introduce/25207)。
- 2021.12.21 发布PaddleOCR v2.4。OCR算法新增1种文本检测算法（PSENet），3种文本识别算法（NRTR、SEED、SAR）；文档结构化算法新增1种关键信息提取算法（SDMGR，[文档](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/ppstructure/docs/kie.md)），3种DocVQA算法（LayoutLM、LayoutLMv2，LayoutXLM，[文档](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.4/ppstructure/vqa)）。
- 2021.9.7 发布PaddleOCR v2.3与[PP-OCRv2](#PP-OCRv2)，CPU推理速度相比于PP-OCR server提升220%；效果相比于PP-OCR mobile 提升7%。
- 2021.8.3 发布PaddleOCR v2.2，新增文档结构分析[PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/ppstructure/README_ch.md)工具包，支持版面分析与表格识别（含Excel导出）。

> [更多](./doc/doc_ch/update.md)

## 特性

![](./doc/features.png)

> 上述内容的使用方法建议从文档教程中的快速开始体验


## 零代码体验

- 在线网站体验：超轻量PP-OCR mobile模型体验地址：https://www.paddlepaddle.org.cn/hub/scene/ocr
- 移动端：[安装包DEMO下载地址](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)(基于EasyEdge和Paddle-Lite, 支持iOS和Android系统)


<a name="贡献代码"></a>

## 社区、社区贡献与社区常规赛

- 加入社区：微信扫描下方二维码加入官方交流群，与各行各业开发者充分交流，期待您的加入。
- 社区贡献：[社区贡献](./doc/doc_ch/thirdparty.md)文档中包含了社区用户**使用PaddleOCR开发的各种工具、应用**以及**为PaddleOCR贡献的功能、优化的文档与代码**等，是官方为社区开发者打造的荣誉墙、也是帮助优质项目宣传的广播站。如果您的OCR项目未被收集在文档中，可根据文档说明与我们联系。
- 社区常规赛：社区常规赛是面向OCR开发者的积分赛事，覆盖文档、代码、模型和应用四大类型，以季度为单位评选并发放奖励，赛题详情与报名方法可参考[链接](https://github.com/PaddlePaddle/PaddleOCR/issues/4982)。

<div align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/dygraph/doc/joinus.PNG"  width = "200" height = "200" />
</div>


<a name="模型下载"></a>
## PP-OCR系列模型列表（更新中）

| 模型简介                              | 模型名称                | 推荐场景        | 检测模型                                                     | 方向分类器                                                   | 识别模型                                                     |
| ------------------------------------- | ----------------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 中英文超轻量PP-OCRv2模型（13.0M）     | ch_PP-OCRv2_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar) |
| 中英文超轻量PP-OCR mobile模型（9.4M） | ch_ppocr_mobile_v2.0_xx | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
| 中英文通用PP-OCR server模型（143.4M） | ch_ppocr_server_v2.0_xx | 服务器端        | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |

更多模型下载（包括多语言），可以参考[PP-OCR 系列模型下载](./doc/doc_ch/models_list.md)

## 文档教程

- [运行环境准备](./doc/doc_ch/environment.md)
- [快速开始（中英文/多语言/文档分析）](./doc/doc_ch/quickstart.md)
- [PP-OCR文本检测识别](./doc/doc_ch/ppocr_introduction.md)
    - [模型库](./doc/doc_ch/models_list.md)
    - [模型训练](./doc/doc_ch/training.md)
        - [文本检测](./doc/doc_ch/detection.md)
        - [文本识别](./doc/doc_ch/recognition.md)
        - [文本方向分类器](./doc/doc_ch/angle_class.md)
    - 模型压缩
        - [模型量化](./deploy/slim/quantization/README.md)
        - [模型裁剪](./deploy/slim/prune/README.md)
        - [知识蒸馏](./doc/doc_ch/knowledge_distillation.md)
    - [推理部署](./deploy/readme_ch.md)
        - [基于Python预测引擎推理](./doc/doc_ch/inference_ppocr.md)
        - [基于C++预测引擎推理](./deploy/cpp_infer/readme.md)
        - [服务化部署](./deploy/pdserving/README_CN.md)
        - [端侧部署](./deploy/lite/readme.md)
        - [Paddle2ONNX模型转化与预测](./deploy/paddle2onnx/readme.md)
        - [Benchmark](./doc/doc_ch/benchmark.md)
- [PP-Structure文档分析](./ppstructure/README_ch.md)
    - [模型库](./doc/doc_ch/models_list_structrure.md)
    - [模型训练](./doc/doc_ch/training.md)
        - [版面分析](./ppstructure/layout/README_ch.md)
        - [表格识别](./ppstructure/table/README_ch.md)
        - [关键信息提取](./ppstructure/docs/kie.md)
        - [DocVQA](./ppstructure/docs/kie.md)
    - [推理部署](./deploy/readme_ch.md)
        - [基于Python预测引擎推理](./doc/doc_ch/inference_ppstructure.md)
        - [服务化部署](./deploy/pdserving/README_CN.md)
- [前沿算法与模型](./doc/doc_ch/algorithm.md)
    - [OCR算法与模型](./doc/doc_ch/algorithm_overview.md)
    - [文档分析算法与模型](./doc/doc_ch/algorithm_overview_structure.md)
    - [基于Python预测引擎推理](./doc/doc_ch/algorithm_inference.md)
    - [更多推理部署](./doc/doc_ch/algorithm_deploy.md) 
    - [使用PaddleOCR架构添加新算法](./doc/doc_ch/add_new_algorithm.md)
- 数据标注与合成
    - [半自动标注工具PPOCRLabel](./PPOCRLabel/README_ch.md)
    - [数据合成工具Style-Text](./StyleText/README_ch.md)
    - [其它数据标注工具](./doc/doc_ch/data_annotation.md)
    - [其它数据合成工具](./doc/doc_ch/data_synthesis.md)
- 数据集
    - [通用中英文OCR数据集](./doc/doc_ch/datasets.md)
    - [手写中文OCR数据集](./doc/doc_ch/handwritten_datasets.md)
    - [垂类多语言OCR数据集](./doc/doc_ch/vertical_and_multilingual_datasets.md)
    - [版面分析数据集](./doc/doc_ch/layout_datasets.md)
    - [表格识别数据集](./doc/doc_ch/table_datasets.md)
    - [DocVQA数据集](./doc/doc_ch/docvqa_datasets.md)
- [效果展示](#效果展示)
- FAQ
    - [通用问题](./doc/doc_ch/FAQ.md)
    - [PaddleOCR实战问题](./doc/doc_ch/FAQ.md)
- [参考文献](./doc/doc_ch/reference.md)
- [许可证书](#许可证书)
- [代码组织结构](./doc/doc_ch/tree.md)




<a name="PP-OCRv2"></a>

## PP-OCRv2 Pipeline
<div align="center">
    <img src="./doc/ppocrv2_framework.jpg" width="800">
</div>
[1] PP-OCR是一个实用的超轻量OCR系统。主要由DB文本检测、检测框矫正和CRNN文本识别三部分组成。该系统从骨干网络选择和调整、预测头部的设计、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型自动裁剪量化8个方面，采用19个有效策略，对各个模块的模型进行效果调优和瘦身(如绿框所示)，最终得到整体大小为3.5M的超轻量中英文OCR和2.8M的英文数字OCR。更多细节请参考PP-OCR技术方案 https://arxiv.org/abs/2009.09941

[2] PP-OCRv2在PP-OCR的基础上，进一步在5个方面重点优化，检测模型采用CML协同互学习知识蒸馏策略和CopyPaste数据增广策略；识别模型采用LCNet轻量级骨干网络、UDML 改进知识蒸馏策略和[Enhanced CTC loss](./doc/doc_ch/enhanced_ctc_loss.md)损失函数改进（如上图红框所示），进一步在推理速度和预测效果上取得明显提升。更多细节请参考PP-OCRv2[技术报告](https://arxiv.org/abs/2109.03144)。

<a name="效果展示"></a>

## 效果展示 [more](./doc/doc_ch/visualization.md)

<details open>
<summary>中文模型</summary>

<div align="center">
      <img src="doc/imgs_results/ch_ppocr_mobile_v2.0/test_add_91.jpg" width="800">
      <img src="doc/imgs_results/ch_ppocr_mobile_v2.0/00018069.jpg" width="800">
</div>
<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/00056221.jpg" width="800">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/rotate_00052204.jpg" width="800">
</div>
    
</details>


<details open>
<summary>英文模型</summary>
    
<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/img_12.jpg" width="800">
</div>

</details>


<details open>
<summary>其他语言模型</summary>
    
<div align="center">
    <img src="./doc/imgs_results/french_0.jpg" width="800">
    <img src="./doc/imgs_results/korean.jpg" width="800">
</div>
    
</details>


<a name="许可证书"></a>

## 许可证书
本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>许可认证。
