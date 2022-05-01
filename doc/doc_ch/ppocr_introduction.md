[English](../doc_en/ppocr_introduction_en.md) | 简体中文

# PP-OCR

- [1. 简介](#1)
- [2. 特性](#2)
- [3. benchmark](#3)
- [4. 效果展示](#4)
- [5. 使用教程](#5)
    - [5.1 快速体验](#51)
    - [5.2 模型训练、压缩、推理部署](#52)
- [6. 模型库](#6)


<a name="1"></a>
## 1. 简介

PP-OCR是PaddleOCR自研的实用的超轻量OCR系统。在实现[前沿算法](algorithm.md)的基础上，考虑精度与速度的平衡，进行**模型瘦身**和**深度优化**，使其尽可能满足产业落地需求。

#### PP-OCR

PP-OCR是一个两阶段的OCR系统，其中文本检测算法选用[DB](algorithm_det_db.md)，文本识别算法选用[CRNN](algorithm_rec_crnn.md)，并在检测和识别模块之间添加[文本方向分类器](angle_class.md)，以应对不同方向的文本识别。

PP-OCR系统pipeline如下：

<div align="center">
    <img src="../ppocrv2_framework.jpg" width="800">
</div>


PP-OCR系统在持续迭代优化，目前已发布PP-OCR和PP-OCRv2两个版本：

PP-OCR从骨干网络选择和调整、预测头部的设计、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型自动裁剪量化8个方面，采用19个有效策略，对各个模块的模型进行效果调优和瘦身(如绿框所示)，最终得到整体大小为3.5M的超轻量中英文OCR和2.8M的英文数字OCR。更多细节请参考PP-OCR技术方案 https://arxiv.org/abs/2009.09941

#### PP-OCRv2

PP-OCRv2在PP-OCR的基础上，进一步在5个方面重点优化，检测模型采用CML协同互学习知识蒸馏策略和CopyPaste数据增广策略；识别模型采用LCNet轻量级骨干网络、UDML 改进知识蒸馏策略和[Enhanced CTC loss](./doc/doc_ch/enhanced_ctc_loss.md)损失函数改进（如上图红框所示），进一步在推理速度和预测效果上取得明显提升。更多细节请参考PP-OCRv2[技术报告](https://arxiv.org/abs/2109.03144)。

#### PP-OCRv3

PP-OCRv3在PP-OCRv2的基础上进一步升级。
PP-OCRv3文本检测从网络结构、蒸馏训练策略两个方向做了进一步优化:
- 网络结构改进：提出两种改进后的FPN网络结构，RSEFPN，LKPAN，分别从channel attention、更大感受野的角度优化FPN中的特征，优化FPN提取的特征。
- 蒸馏训练策略：首先，以resnet50作为backbone，改进后的LKPAN网络结构作为FPN，使用DML自蒸馏策略得到精度更高的teacher模型；然后，student模型FPN部分采用RSEFPN，采用PPOCRV2提出的CML蒸馏方法蒸馏，在训练过程中，动态调整CML蒸馏teacher loss的占比。

|序号|策略|模型大小|hmean|Intel Gold 6148CPU+mkldnn预测耗时|
|-|-|-|-|-|
|0|ppocr_mobile|3M|81.3|117ms|
|1|PPOCRV2|3M|83.3|117ms|
|2|teacher DML|124M|86.0|-|
|3|1 + 2 + RESFPN|3.6M|85.4|124ms|
|4|1 + 2 + LKPAN|4.6M|86.0|156ms|


- PP-OCRv3 文本识别

[SVTR](todo:add_link) 证明了强大的单视觉模型即可高效准确完成文本识别任务，在中英文数据上均有优秀的表现。基于SVTR的工作，PP-OCRv3首先验证了SVTR_tiny结构在中文数据上的效果。
实验发现SVTR_tiny在中文测试集上acc可以提升10.7%。

<img src="../ppocr_v3/svtr_tiny.jpg" width=800>

非常遗憾，由于 MKLDNN 加速库支持的模型结构有限，SVTR 在CPU+MKLDNN上相比PP-OCRv2慢了10倍。

PP-OCRv3 期望提升模型精度的同时不带来额外的推理耗时，以加速预测为目的，分析得到主要耗时部分在Transformer Block，并对 SVTR_tiny 的结构进行了一系列优化:

1. 将SVTR网络前半部分替换为LCNet的前三个stage，保留4个 SVTR 的 Global attenntion，精度为76%，速度基本不变。
<img src="../ppocr_v3/svtr_g4.png" width=800>
2. 将4个Global attention 减小到2个，精度为72.9%，速度提升3倍。
<img src="../ppocr_v3/svtr_g2.png" width=800>
3. attention 的预测速度与输入其feature的shape有关，因此移动Global attention至avg_pool后，精度下降为71.9%，速度超越 CNN-base 的PP-OCRv2 27%。
<img src="../ppocr_v3/ppocr_v3.png" width=800>

为了提升模型精度同时不引入额外推理成本，PP-OCRv3参考GTC策略，使用Attention监督CTC训练，预测时完全去除Attention模块，在推理阶段不增加任何耗时, 精度提升3.8%。
<img src="../ppocr_v3/GTC.png" width=800>

训练策略方面参考 [SSL](https://github.com/ku21fan/STR-Fewer-Labels) 设计了文本方向任务，训练了适用于文本识别的预训练模型，加速模型收敛过程，精度提升了0.6%; 使用UDML蒸馏策略，进一步提升精度1.5%。

<img src="../ppocr_v3/SSL.png" width="300"> <img src="../ppocr_v3/UDML.png" width="500">


数据增强方面基于 [ConCLR](https://www.cse.cuhk.edu.hk/~byu/papers/C139-AAAI2022-ConCLR.pdf) 中的ConAug方法，设计了 RecConAug 数据增强方法，增强数据多样性，精度提升0.5%，
<img src="../ppocr_v3/recconaug.png" width=800>


总体来讲PP-OCRv3识别从网络结构、训练策略、数据增强三个方向做了进一步优化:
- 网络结构上：结合 LCNet 与[SVTR](todo:add_link) 的 Transformer block，去除LSTM特征解码模块，并行处理序列特征，兼顾精度与预测速度；参考 [GTC](https://arxiv.org/pdf/2002.01276.pdf) 策略，使用注意力机制模块指导CTC训练，定位和识别字符，提升不规则文本的识别精度。
- 训练策略上：参考 [SSL](https://github.com/ku21fan/STR-Fewer-Labels) 设计了方向分类前序任务，获取更优预训练模型，加速模型收敛过程，提升精度; 使用UDML蒸馏策略、监督attention、ctc两个分支得到更优模型。
- 数据增强上：基于 [ConCLR](https://www.cse.cuhk.edu.hk/~byu/papers/C139-AAAI2022-ConCLR.pdf) 中的ConAug方法，改进得到 RecConAug 数据增广方法，支持随机结合任意多张图片，提升训练数据的上下文信息丰富度，增强模型鲁棒性。
基于上述策略，PP-OCRv3识别模型相比上一版本，速度基本无损，精度进一步提升4.5%。 具体消融实验：

实验细节：

| id | 策略 |  模型大小 | 精度 |
|-----|-----|--------|----|
| 01 | PP-OCRv2 | 8M | 69.3% |
| 02 | SVTR_tiny | 19M | 80.1% |
| 03 | LCNet_SVTR_G6 | 8.2M | 76% |
| 04 | LCNet_SVTR_G1 | - | 72.98% |
| 05 | PP-OCRv3 | 12M | 71.9% |
| 06 | + GTC | 12M | 75.8% |
| 07 | + RecConAug | 12M | 76.3% |
| 08 | + SSL pretrain | 12M | 76.9% |
| 09 | + UDML | 12M | 78.4% |
| 10 | + unlabeled data | 12M | 79.4% |



<a name="2"></a>
## 2. 特性

- 超轻量PP-OCRv2系列：检测（3.1M）+ 方向分类器（1.4M）+ 识别（8.5M）= 13.0M
- 超轻量PP-OCR mobile移动端系列：检测（3.0M）+方向分类器（1.4M）+ 识别（5.0M）= 9.4M
- 通用PP-OCR server系列：检测（47.1M）+方向分类器（1.4M）+ 识别（94.9M）= 143.4M
- 支持中英文数字组合识别、竖排文本识别、长文本识别
- 支持多语言识别：韩语、日语、德语、法语等约80种语言

<a name="3"></a>
## 3. benchmark

关于PP-OCR系列模型之间的性能对比，请查看[benchmark](./benchmark.md)文档。


<a name="4"></a>
## 4. 效果展示 [more](./visualization.md)

<details open>
<summary>PP-OCRv2 中文模型</summary>

<div align="center">
      <img src="../imgs_results/ch_ppocr_mobile_v2.0/test_add_91.jpg" width="800">
      <img src="../imgs_results/ch_ppocr_mobile_v2.0/00018069.jpg" width="800">
</div>
<div align="center">
    <img src="../imgs_results/ch_ppocr_mobile_v2.0/00056221.jpg" width="800">
    <img src="../imgs_results/ch_ppocr_mobile_v2.0/rotate_00052204.jpg" width="800">
</div>

</details>


<details open>
<summary>PP-OCRv2 英文模型</summary>

<div align="center">
    <img src="../imgs_results/ch_ppocr_mobile_v2.0/img_12.jpg" width="800">
</div>

</details>


<details open>
<summary>PP-OCRv2 其他语言模型</summary>

<div align="center">
    <img src="../imgs_results/french_0.jpg" width="800">
    <img src="../imgs_results/korean.jpg" width="800">
</div>

</details>


<a name="5"></a>
## 5. 使用教程

<a name="51"></a>
### 5.1 快速体验

- 在线网站体验：超轻量PP-OCR mobile模型体验地址：https://www.paddlepaddle.org.cn/hub/scene/ocr
- 移动端demo体验：[安装包DEMO下载地址](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)(基于EasyEdge和Paddle-Lite, 支持iOS和Android系统)
- 一行命令快速使用：[快速开始（中英文/多语言）](./doc/doc_ch/quickstart.md)

<a name="52"></a>
### 5.2 模型训练、压缩、推理部署

更多教程，包括模型训练、模型压缩、推理部署等，请参考[文档教程](../../README_ch.md#文档教程)。

<a name="6"></a>
## 6. 模型库

PP-OCR中英文模型列表如下：

| 模型简介                              | 模型名称                | 推荐场景        | 检测模型                                                     | 方向分类器                                                   | 识别模型                                                     |
| ------------------------------------- | ----------------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 中英文超轻量PP-OCRv3模型（16.2M）     | ch_PP-OCRv3_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
| 英文超轻量PP-OCRv3模型（13.4M）     | en_PP-OCRv3_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
| 中英文超轻量PP-OCRv2模型（13.0M）     | ch_PP-OCRv2_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar) |
| 中英文超轻量PP-OCR mobile模型（9.4M） | ch_ppocr_mobile_v2.0_xx | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar) |
| 中英文通用PP-OCR server模型（143.4M） | ch_ppocr_server_v2.0_xx | 服务器端        | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar) |

更多模型下载（包括英文数字模型、多语言模型、Paddle-Lite模型等），可以参考[PP-OCR 系列模型下载](./models_list.md)。
