[English](README.md) | 简体中文

## 简介
PP-Structure是一个可用于复杂文档结构分析和处理的OCR工具包，旨在帮助开发者更好的完成文档理解相关任务。

## 近期更新
* 2021.12.07 新增VQA任务-SER和RE。

## 特性

PP-Structure是一个可用于复杂文档结构分析和处理的OCR工具包，主要特性如下：
- 支持对图片形式的文档进行版面分析，可以划分**文字、标题、表格、图片以及列表**5类区域（与Layout-Parser联合使用）
- 支持文字、标题、图片以及列表区域提取为文字字段（与PP-OCR联合使用）
- 支持表格区域进行结构化分析，最终结果输出Excel文件
- 支持python whl包和命令行两种方式，简单易用
- 支持版面分析和表格结构化两类任务自定义训练
- 支持文档视觉问答(Document Visual Question Answering，DOC-VQA)任务-语义实体识别(Semantic Entity Recognition，SER)和关系抽取(Relation Extraction，RE)


## 1. 效果展示

### 1.1 版面分析和表格识别

<img src="../doc/table/ppstructure.GIF" width="100%"/>

### 1.2 VQA

* SER

![](./vqa/images/result_ser/zh_val_0_ser.jpg) | ![](./vqa/images/result_ser/zh_val_42_ser.jpg)
---|---

图中不同颜色的框表示不同的类别，对于XFUN数据集，有`QUESTION`, `ANSWER`, `HEADER` 3种类别

* 深紫色：HEADER
* 浅紫色：QUESTION
* 军绿色：ANSWER

在OCR检测框的左上方也标出了对应的类别和OCR识别结果。

* RE

![](./vqa/images/result_re/zh_val_21_re.jpg) | ![](./vqa/images/result_re/zh_val_40_re.jpg)
---|---


图中红色框表示问题，蓝色框表示答案，问题和答案之间使用绿色线连接。在OCR检测框的左上方也标出了对应的类别和OCR识别结果。

## 2. 快速体验

代码体验：从 [快速安装](./docs/quickstart.md) 开始

## 3. PP-Structure  Pipeline介绍

### 3.1 版面分析+表格识别

![pipeline](../doc/table/pipeline.jpg)

在PP-Structure中，图片会先经由Layout-Parser进行版面分析，在版面分析中，会对图片里的区域进行分类，包括**文字、标题、图片、列表和表格**5类。对于前4类区域，直接使用PP-OCR完成对应区域文字检测与识别。对于表格类区域，经过表格结构化处理后，表格图片转换为相同表格样式的Excel文件。

#### 3.1.1 版面分析

版面分析对文档数据进行区域分类，其中包括版面分析工具的Python脚本使用、提取指定类别检测框、性能指标以及自定义训练版面分析模型，详细内容可以参考[文档](layout/README_ch.md)。

#### 3.1.2 表格识别

表格识别将表格图片转换为excel文档，其中包含对于表格文本的检测和识别以及对于表格结构和单元格坐标的预测，详细说明参考[文档](table/README_ch.md)


### 3.2 VQA

coming soon

## 4. 模型库

PP-Structure系列模型列表（更新中）

* LayoutParser 模型

|模型名称|模型简介|下载地址|
| --- | --- | --- |
| ppyolov2_r50vd_dcn_365e_publaynet | PubLayNet 数据集训练的版面分析模型，可以划分**文字、标题、表格、图片以及列表**5类区域 | [PubLayNet](https://paddle-model-ecology.bj.bcebos.com/model/layout-parser/ppyolov2_r50vd_dcn_365e_publaynet.tar) |


* OCR和表格识别模型

|模型名称|模型简介|模型大小|下载地址|
| --- | --- | --- | --- |
|ch_ppocr_mobile_slim_v2.0_det|slim裁剪版超轻量模型，支持中英文、多语种文本检测|2.6M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/slim/ch_ppocr_mobile_v2.0_det_prune_infer.tar) |
|ch_ppocr_mobile_slim_v2.0_rec|slim裁剪量化版超轻量模型，支持中英文、数字识别|6M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_slim_train.tar) |
|en_ppocr_mobile_v2.0_table_structure|PubLayNet数据集训练的英文表格场景的表格结构预测|18.6M|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/table/en_ppocr_mobile_v2.0_table_structure_train.tar) |

* VQA模型

|模型名称|模型简介|推理模型大小|下载地址|
| --- | --- | --- | --- |
|PP-Layout_v1.0_ser_pretrained|基于LayoutXLM在xfun中文数据集上训练的SER模型|1.4G|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/pplayout/PP-Layout_v1.0_ser_pretrained.tar) |
|PP-Layout_v1.0_re_pretrained|基于LayoutXLM在xfun中文数据集上训练的RE模型|1.4G|[推理模型 coming soon]() / [训练模型](https://paddleocr.bj.bcebos.com/pplayout/PP-Layout_v1.0_re_pretrained.tar) |


更多模型下载，可以参考 [模型库](./docs/model_list.md)
