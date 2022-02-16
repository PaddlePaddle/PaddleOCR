[English](README.md) | 简体中文

- [1. 简介](#1-简介)
- [2. 近期更新](#2-近期更新)
- [3. 特性](#3-特性)
- [4. 效果展示](#4-效果展示)
  - [4.1 版面分析和表格识别](#41-版面分析和表格识别)
  - [4.2 DOC-VQA](#42-doc-vqa)
- [5. 快速体验](#5-快速体验)
- [6. PP-Structure 介绍](#6-pp-structure-介绍)
  - [6.1 版面分析+表格识别](#61-版面分析表格识别)
    - [6.1.1 版面分析](#611-版面分析)
    - [6.1.2 表格识别](#612-表格识别)
  - [6.2 DOC-VQA](#62-doc-vqa)
- [7. 模型库](#7-模型库)
  - [7.1 版面分析模型](#71-版面分析模型)
  - [7.2 OCR和表格识别模型](#72-ocr和表格识别模型)
  - [7.2 DOC-VQA 模型](#72-doc-vqa-模型)


## 1. 简介
PP-Structure是一个可用于复杂文档结构分析和处理的OCR工具包，旨在帮助开发者更好的完成文档理解相关任务。

## 2. 近期更新
* 2022.02.12 DOC-VQA增加LayoutLMv2模型。
* 2021.12.07 新增[DOC-VQA任务SER和RE](vqa/README.md)。

## 3. 特性

PP-Structure的主要特性如下：
- 支持对图片形式的文档进行版面分析，可以划分**文字、标题、表格、图片以及列表**5类区域（与Layout-Parser联合使用）
- 支持文字、标题、图片以及列表区域提取为文字字段（与PP-OCR联合使用）
- 支持表格区域进行结构化分析，最终结果输出Excel文件
- 支持python whl包和命令行两种方式，简单易用
- 支持版面分析和表格结构化两类任务自定义训练
- 支持文档视觉问答(Document Visual Question Answering，DOC-VQA)任务-语义实体识别(Semantic Entity Recognition，SER)和关系抽取(Relation Extraction，RE)

## 4. 效果展示

### 4.1 版面分析和表格识别

<img src="../doc/table/ppstructure.GIF" width="100%"/>

图中展示了版面分析+表格识别的整体流程，图片先有版面分析划分为图像、文本、标题和表格四种区域，然后对图像、文本和标题三种区域进行OCR的检测识别，对表格进行表格识别，其中图像还会被存储下来以便使用。

### 4.2 DOC-VQA

* SER

![](../doc/vqa/result_ser/zh_val_0_ser.jpg) | ![](../doc/vqa/result_ser/zh_val_42_ser.jpg)
---|---

图中不同颜色的框表示不同的类别，对于XFUN数据集，有`QUESTION`, `ANSWER`, `HEADER` 3种类别

* 深紫色：HEADER
* 浅紫色：QUESTION
* 军绿色：ANSWER

在OCR检测框的左上方也标出了对应的类别和OCR识别结果。

* RE

![](../doc/vqa/result_re/zh_val_21_re.jpg) | ![](../doc/vqa/result_re/zh_val_40_re.jpg)
---|---


图中红色框表示问题，蓝色框表示答案，问题和答案之间使用绿色线连接。在OCR检测框的左上方也标出了对应的类别和OCR识别结果。

## 5. 快速体验

请参考[快速安装](./docs/quickstart.md)教程。

## 6. PP-Structure 介绍

### 6.1 版面分析+表格识别

![pipeline](../doc/table/pipeline.jpg)

在PP-Structure中，图片会先经由Layout-Parser进行版面分析，在版面分析中，会对图片里的区域进行分类，包括**文字、标题、图片、列表和表格**5类。对于前4类区域，直接使用PP-OCR完成对应区域文字检测与识别。对于表格类区域，经过表格结构化处理后，表格图片转换为相同表格样式的Excel文件。

#### 6.1.1 版面分析

版面分析对文档数据进行区域分类，其中包括版面分析工具的Python脚本使用、提取指定类别检测框、性能指标以及自定义训练版面分析模型，详细内容可以参考[文档](layout/README_ch.md)。

#### 6.1.2 表格识别

表格识别将表格图片转换为excel文档，其中包含对于表格文本的检测和识别以及对于表格结构和单元格坐标的预测，详细说明参考[文档](table/README_ch.md)。

### 6.2 DOC-VQA

DOC-VQA指文档视觉问答，其中包括语义实体识别 (Semantic Entity Recognition, SER) 和关系抽取 (Relation Extraction, RE) 任务。基于 SER 任务，可以完成对图像中的文本识别与分类；基于 RE 任务，可以完成对图象中的文本内容的关系提取，如判断问题对(pair)，详细说明参考[文档](vqa/README.md)。

## 7. 模型库

PP-Structure系列模型列表（更新中）

### 7.1 版面分析模型

|模型名称|模型简介|下载地址|
| --- | --- | --- |
| ppyolov2_r50vd_dcn_365e_publaynet | PubLayNet 数据集训练的版面分析模型，可以划分**文字、标题、表格、图片以及列表**5类区域 | [PubLayNet](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar) |

### 7.2 OCR和表格识别模型

|模型名称|模型简介|模型大小|下载地址|
| --- | --- | --- | --- |
|ch_PP-OCRv2_det_slim|【最新】slim量化+蒸馏版超轻量模型，支持中英文、多语种文本检测| 3M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_slim_quant_infer.tar)|
|ch_PP-OCRv2_rec_slim|【最新】slim量化版超轻量模型，支持中英文、数字识别| 9M |[推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_slim_quant_train.tar) |
|en_ppocr_mobile_v2.0_table_structure|PubLayNet数据集训练的英文表格场景的表格结构预测|18.6M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar) |

### 7.2 DOC-VQA 模型

|模型名称|模型简介|模型大小|下载地址|
| --- | --- | --- | --- |
|ser_LayoutXLM_xfun_zhd|基于LayoutXLM在xfun中文数据集上训练的SER模型|1.4G|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/pplayout/ser_LayoutXLM_xfun_zh.tar) |
|re_LayoutXLM_xfun_zh|基于LayoutXLM在xfun中文数据集上训练的RE模型|1.4G|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/pplayout/re_LayoutXLM_xfun_zh.tar) |


更多模型下载，可以参考 [PP-OCR model_list](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/models_list.md) and  [PP-Structure model_list](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppstructure/docs/models_list.md)
