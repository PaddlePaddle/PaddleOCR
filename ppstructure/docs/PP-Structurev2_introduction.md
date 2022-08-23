# PP-Structurev2

##  目录

- [1. 背景](#1-背景)
- [2. 简介](#3-简介)
- [3. 整图方向矫正](#3-整图方向矫正)
- [4. 版面信息结构化](#4-版面信息结构化)
  - [4.1 版面分析](#41-版面分析)
  - [4.2 表格识别](#42-表格识别)
  - [4.3 版面恢复](#43-版面恢复)
- [5. 关键信息抽取](#5-关键信息抽取)
- [6. Reference](#6-Reference)

## 1. 背景

现实场景中包含大量的文档图像，它们以图片等非结构化形式存储。基于文档图像的结构化分析与信息抽取对于数据的数字化存储以及产业的数字化转型至关重要。基于该考虑，PaddleOCR自研并发布了PP-Structure智能文档分析系统，旨在帮助开发者更好的完成版面分析、表格识别、关键信息抽取等文档理解相关任务。

近期，PaddleOCR团队针对PP-Structurev1的版面分析、表格识别、关键信息抽取模块，进行了共计8个方面的升级，同时新增整图方向矫正、文档复原等功能，打造出一个全新的、效果更优的文档分析系统：PP-Structurev2。

## 2. 简介

PP-Structurev2在PP-Structurev1的基础上进一步改进，主要有以下3个方面升级：

 * **系统功能升级** ：新增图像矫正和版面复原模块，图像转word/pdf、关键信息抽取能力全覆盖！
 * **系统性能优化** ：
	 * 版面分析：发布轻量级版面分析模型，速度提升**11倍**，平均CPU耗时仅需**41ms**！
	 * 表格识别：设计3大优化策略，预测耗时不变情况下，模型精度提升**6%**。
	 * 关键信息抽取：设计视觉无关模型结构，语义实体识别精度提升**2.8%**，关系抽取精度提升**9.1%**。
 * **中文场景适配** ：完成对版面分析与表格识别的中文场景适配，开源**开箱即用**的中文场景版面结构化模型！

PP-Structurev2系统流程图如下所示，文档图像首先经过图像矫正模块，判断整图方向并完成转正，随后可以完成版面信息分析与关键信息抽取2类任务。版面分析任务中，图像首先经过版面分析模型，将图像划分为文本、表格、图像等不同区域，随后对这些区域分别进行识别，如，将表格区域送入表格识别模块进行结构化识别，将文本区域送入OCR引擎进行文字识别，最后使用版面恢复模块将其恢复为与原始图像布局一致的word或者pdf格式的文件；关键信息抽取任务中，首先使用OCR引擎提取文本内容，然后由语义实体识别模块获取图像中的语义实体，最后经关系抽取模块获取语义实体之间的对应关系，从而提取需要的关键信息。

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185939247-57e53254-399c-46c4-a610-da4fa79232f5.png" width="1200">
</div>


从算法改进思路来看，对系统中的3个关键子模块，共进行了8个方面的改进。

* 版面分析
	* PP-PicoDet：轻量级版面分析模型
	* FGD：兼顾全局与局部特征的模型蒸馏算法

* 表格识别
	* PP-LCNet:  CPU友好型轻量级骨干网络
	* CSP-PAN：轻量级高低层特征融合模块
	* SLAHead：结构与位置信息对齐的特征解码模块

* 关键信息抽取
	* VI-LayoutXLM：视觉特征无关的多模态预训练模型结构
	* TB-YX：考虑阅读顺序的文本行排序逻辑
	* UDML：联合互学习知识蒸馏策略

最终，与PP-Structurev1相比：

- 版面分析模型参数量减少95.6%，推理速度提升11倍，精度提升0.4%；
- 表格识别预测耗时不变，模型精度提升6%，端到端TEDS提升2%；
- 关键信息抽取模型速度提升2.8倍，语义实体识别模型精度提升2.8%；关系抽取模型精度提升9.1%。

下面对各个模块进行详细介绍。

## 3. 整图方向矫正

由于训练集一般以正方向图像为主，旋转过的文档图像直接输入模型会增加识别难度，影响识别效果。PP-Structurev2引入了整图方向矫正模块来判断含文字图像的方向，并将其进行方向调整。

我们直接调用PaddleClas中提供的文字图像方向分类模型-[PULC_text_image_orientation](https://github.com/PaddlePaddle/PaddleClas/blob/develop/docs/zh_CN/PULC/PULC_text_image_orientation.md)，该模型部分数据集图像如下所示。不同于文本行方向分类器，文字图像方向分类模型针对整图进行方向判别。文字图像方向分类模型在验证集上精度高达99%，单张图像CPU预测耗时仅为`2.16ms`。

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185939683-f6465473-3303-4a0c-95be-51f04fb9f387.png" width="600">
</div>

## 4. 版面信息结构化

### 4.1 版面分析

版面分析指的是对图片形式的文档进行区域划分，定位其中的关键区域，如文字、标题、表格、图片等，PP-Structurev1使用了PaddleDetection中开源的高效检测算法PP-YOLOv2完成版面分析的任务。

在PP-Structurev2中，我们发布基于PP-PicoDet的轻量级版面分析模型，并针对版面分析场景定制图像尺度，同时使用FGD知识蒸馏算法，进一步提升模型精度。最终CPU上`41ms`即可完成版面分析过程(仅包含模型推理时间，数据预处理耗时大约50ms左右)。在公开数据集PubLayNet 上，消融实验如下：

| 实验序号 | 策略                          | 模型存储(M) | mAP     | CPU预测耗时(ms) |
|:------:|:------:|:------:|:------:|:------:|
| 1    |  PP-YOLOv2(640*640)  |  221  | 93.6% |  512  |
| 2    | PP-PicoDet-LCNet2.5x(640*640) |  29.7 | 92.5% |53.2|
| 3    | PP-PicoDet-LCNet2.5x(800*608) |   29.7  | 94.2% |83.1 |
| 4    | PP-PicoDet-LCNet1.0x(800*608) |    9.7  | 93.5% | 41.2|
| 5    | PP-PicoDet-LCNet1.0x(800*608) + FGD |  9.7  | 94% |41.2|

* 测试条件
	* paddle版本：2.3.0
	* CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz，开启mkldnn，线程数为10

在PubLayNet数据集上，与其他方法的性能对比如下表所示。可以看到，和基于Detectron2的版面分析工具layoutparser相比，我们的模型精度高出大约5%，预测速度快约69倍。

| 模型                | mAP | CPU预测耗时   |
|-------------------|-----------|------------|
| layoutparser (Detectron2)   | 88.98%    | 2.9s    |
| PP-Structurev2 (PP-PicoDet) | **94%**    |   41.2ms   |

[PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet)数据集是一个大型的文档图像数据集，包含Text、Title、Tale、Figure、List，共5个类别。数据集中包含335,703张训练集、11,245张验证集和11,405张测试集。训练数据与标注示例图如下所示：

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185940305-2cd3633b-4c43-4f84-8a6f-5ce6a24e88ce.png" width="600">
</div>


#### 4.1.1 优化策略

**（1）轻量级版面分析模型PP-PicoDet**

`PP-PicoDet`是PaddleDetection中提出的轻量级目标检测模型，通过使用PP-LCNet骨干网络、CSP-PAN特征融合模块、SimOTA标签分配方法等优化策略，最终在CPU与移动端具有卓越的性能。我们将PP-Structurev1中采用的PP-YOLOv2模型替换为`PP-PicoDet`，同时针对版面分析场景优化预测尺度，从针对目标检测设计的`640*640`调整为更适配文档图像的`800*608`，在`1.0x`配置下，模型精度与PP-YOLOv2相当，CPU平均预测速度可提升11倍。

**（1）FGD知识蒸馏**

FGD（Focal and Global Knowledge Distillation for Detectors），是一种兼顾局部全局特征信息的模型蒸馏方法，分为Focal蒸馏和Global蒸馏2个部分。Focal蒸馏分离图像的前景和背景，让学生模型分别关注教师模型的前景和背景部分特征的关键像素；Global蒸馏部分重建不同像素之间的关系并将其从教师转移到学生，以补偿Focal蒸馏中丢失的全局信息。我们基于FGD蒸馏策略，使用教师模型PP-PicoDet-LCNet2.5x（mAP=94.2%）蒸馏学生模型PP-PicoDet-LCNet1.0x（mAP=93.5%），可将学生模型精度提升0.5%，和教师模型仅差0.2%，而预测速度比教师模型快1倍。

#### 4.1.2 场景适配

**（1）中文版面分析**

除了英文公开数据集PubLayNet，我们也在中文场景进行了场景适配与方法验证。[CDLA](https://github.com/buptlihang/CDLA)是一个中文文档版面分析数据集，面向中文文献类（论文）场景，包含正文、标题等10个label。数据集中包含5,000张训练集和1,000张验证集。训练数据与标注示例图如下所示：


<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185940445-92b7613f-e431-43c2-9033-3b3618ddae02.png" width="600">
</div>


在CDLA 数据集上，消融实验如下：

| 实验序号 | 策略             |  mAP     |
|:------:|:------:|:------:|
| 1    |  PP-YOLOv2 |  84.7% |
| 2    |  PP-PicoDet-LCNet2.5x(800*608) |  87.8% |
| 3    |  PP-PicoDet-LCNet1.0x(800*608) | 84.5% |
| 4    |  PP-PicoDet-LCNet1.0x(800*608) + FGD |  86.8% |


**（2）表格版面分析**

在实际应用中，很多场景并不关注图像中的图片、文本等版面区域，而仅需要提取文档图像中的表格，此时版面分析任务退化为一个表格检测任务，表格检测往往也是表格识别的前序任务。面向中英文文档场景，我们整理了开源领域含表格的版面分析数据集，包括TableBank、DocBank等。融合后的数据集中包含496,405张训练集与9,495张验证集图像。

在表格数据集上，消融实验如下：

| 实验序号 | 策略            |  mAP     |
|:------:|:------:|:------:|
| 1    |  PP-YOLOv2  |91.3% |
| 2    |  PP-PicoDet-LCNet2.5x(800*608) |  95.9% |
| 3    |  PP-PicoDet-LCNet1.0x(800*608) |   95.2% |
| 4    |  PP-PicoDet-LCNet1.0x(800*608) + FGD |  95.7% |

表格检测效果示意图如下：

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185940654-956ef614-888a-4779-bf63-a6c2b61b97fa.png" width="600">
</div>

### 4.2 表格识别

基于深度学习的表格识别算法种类丰富，PP-Structurev1中，我们基于文本识别算法RARE研发了端到端表格识别算法TableRec-RARE，模型输出为表格结构的HTML表示，进而可以方便地转化为Excel文件。PP-Structurev2中，我们对模型结构和损失函数等5个方面进行升级，提出了 SLANet (Structure Location Alignment Network) ，模型结构如下图所示：

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185940811-089c9265-4be9-4776-b365-6d1125606b4b.png" width="1200">
</div>

在PubTabNet英文表格识别数据集上的消融实验如下：

|策略|Acc|TEDS|推理速度(CPU+MKLDNN)|模型大小|
|---|---|---|---|---|
|TableRec-RARE|	71.73% | 93.88% |779ms	|6.8M|
|+PP-LCNet|	74.71% |94.37%	|778ms|	8.7M|
|+CSP-PAN|	75.68%| 94.72%	|708ms|	9.3M|
|+SLAHead|	77.7%|94.85%|	766ms|	9.2M|
|+MergeToken|	76.31%|	95.89%|766ms|	9.2M|

* 测试环境
    * paddle版本：2.3.1
    * CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz，开启mkldnn，线程数为10

在PubtabNet英文表格识别数据集上，和其他方法对比如下：

|策略|Acc|TEDS|推理速度(CPU+MKLDNN)|模型大小|
|---|---|---|---|---|
|TableMaster|77.9%|96.12%|2144ms|253M|
|TableRec-RARE|	71.73% | 93.88% |779ms	|6.8M|
|SLANet|76.31%|	95.89%|766ms|9.2M|

#### 4.2.1 优化策略

**（1） CPU友好型轻量级骨干网络PP-LCNet**

PP-LCNet是结合Intel-CPU端侧推理特性而设计的轻量高性能骨干网络，该方案在图像分类任务上取得了比ShuffleNetV2、MobileNetV3、GhostNet等轻量级模型更优的“精度-速度”均衡。PP-Structurev2中，我们采用PP-LCNet作为骨干网络，表格识别模型精度从71.73%提升至72.98%；同时加载通过SSLD知识蒸馏方案训练得到的图像分类模型权重作为表格识别的预训练模型，最终精度进一步提升2.95%至74.71%。

**（2）轻量级高低层特征融合模块CSP-PAN**

对骨干网络提取的特征进行融合，可以有效解决尺度变化较大等复杂场景中的模型预测问题。早期，FPN模块被提出并用于特征融合，但是它的特征融合过程仅包含单向（高->低），融合不够充分。CSP-PAN基于PAN进行改进，在保证特征融合更为充分的同时，使用CSP block、深度可分离卷积等策略减小了计算量。在表格识别场景中，我们进一步将CSP-PAN的通道数从128降低至96以降低模型大小。最终表格识别模型精度提升0.97%至75.68%，预测速度提升10%。

**（3）结构与位置信息对齐的特征解码模块SLAHead**

TableRec-RARE的TableAttentionHead如下图a所示，TableAttentionHead在执行完全部step的计算后拿到最终隐藏层状态表征(hiddens)，随后hiddens经由SDM(Structure Decode Module)和CLDM(Cell Location Decode Module)模块生成全部的表格结构token和单元格坐标。但是这种设计忽略了单元格token和坐标之间一一对应的关系。

PP-Structurev2中，我们设计SLAHead模块，对单元格token和坐标之间做了对齐操作，如下图b所示。在SLAHead中，每一个step的隐藏层状态表征会分别送入SDM和CLDM来得到当前step的token和坐标，每个step的token和坐标输出分别进行concat得到表格的html表达和全部单元格的坐标。此外，考虑到表格识别模型的单元格准确率依赖于表格结构的识别准确，我们将损失函数中表格结构分支与单元格定位分支的权重比从1:1提升到8:1，并使用收敛更稳定的Smoothl1 Loss替换定位分支中的MSE Loss。最终模型精度从75.68%提高至77.7%。


<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185940968-e3a2fbac-78d7-4b74-af54-a1dab860f470.png" width="1200">
</div>


**（4）其他**

TableRec-RARE算法中，我们使用`<td>`和`</td>`两个单独的token来表示一个非跨行列单元格，这种表示方式限制了网络对于单元格数量较多表格的处理能力。

PP-Structurev2中，我们参考TableMaster中的token处理方法，将`<td>`和`</td>`合并为一个token-`<td></td>`。合并token后，验证集中token长度大于500的图片也参与模型评估，最终模型精度降低为76.31%，但是端到端TEDS提升1.04%。

#### 4.2.2 中文场景适配

除了上述模型策略的升级外，本次升级还开源了中文表格识别模型。在实际应用场景中，表格图像存在着各种各样的倾斜角度（PubTabNet数据集不存在该问题），因此在中文模型中，我们将单元格坐标回归的点数从2个（左上，右下）增加到4个(左上，右上，右下，左下)。在内部测试集上，模型升级前后指标如下：
|模型|acc|
|---|---|
|TableRec-RARE|44.3%|
|SLANet|59.35%|

可视化结果如下，左为输入图像，右为识别的html表格


<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185941221-c94e3d45-524c-4073-9644-21ba6a9fd93e.png" width="800">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185941254-31f2b1fa-d594-4037-b1c7-0f24543e5d19.png" width="800">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185941273-2f7131df-3fe7-43b8-9c64-77ad2cf3b947.png" width="800">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185941295-d0672aa8-548d-4e6a-812c-ac5d5fd8a269.png" width="800">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185941324-8036e959-abdc-4dc5-a9f2-730b07b4e3d3.png" width="800">
</div>




### 4.3 版面恢复

版面恢复指的是文档图像经过OCR识别、版面分析、表格识别等方法处理后的内容可以与原始文档保持相同的排版方式，并输出到word等文档中。PP-Structurev2中，我们版面恢复系统，包含版面分析、表格识别、OCR文本检测与识别等子模块。
下图展示了版面恢复的结果：

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185941816-4dabb3e8-a0db-4094-98ea-52e0a0fda8e8.png" width="1200">
</div>

## 5. 关键信息抽取

关键信息抽取指的是针对文档图像的文字内容，提取出用户关注的关键信息，如身份证中的姓名、住址等字段。PP-Structure中支持了基于多模态LayoutLM系列模型的语义实体识别 (Semantic Entity Recognition, SER) 以及关系抽取 (Relation Extraction, RE) 任务。PP-Structurev2中，我们对模型结构以及下游任务训练方法进行升级，提出了VI-LayoutXLM（Visual-feature Independent LayoutXLM），具体流程图如下所示。


<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185941978-abec7d4a-5e3a-4141-83f8-088d04ef898e.png" width="1000">
</div>


具体优化策略包括：

* VI-LayoutXLM：视觉特征无关的多模态预训练模型结构
* TB-YX：考虑人类阅读顺序的文本行排序逻辑
* UDML：联合互学习知识蒸馏策略

XFUND-zh数据集上，SER任务的消融实验如下所示。

| 实验序号 | 策略                          | 模型大小(G) | 精度     | GPU预测耗时(ms) | CPU预测耗时(ms) |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 1    | LayoutXLM          | 1.4     | 89.50% | 59.35       | 766.16      |
| 2    | VI-LayoutXLM | 1.1     | 90.46% | 23.71       | 675.58      |
| 3    | 实验2 + TB-YX文本行排序              | 1.1     | 92.50% | 23.71       | 675.58      |
| 4    | 实验3 + UDML蒸馏                | 1.1     | 93.19% | 23.71       | 675.58      |
| 5    | 实验3 + UDML蒸馏                | 1.1     | **93.19%** | **15.49**       | **675.58**      |

* 测试条件
	* paddle版本：2.3.0
	* GPU：V100，实验5的GPU预测耗时使用`trt+fp16`测试得到，环境为cuda10.2+ cudnn8.1.1 + trt7.2.3.4，其他实验的预测耗时统计中没有使用TRT。
	* CPU：Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz，开启mkldnn，线程数为10

在XFUND数据集上，与其他方法的效果对比如下所示。

| 模型                | SER Hmean | RE Hmean   |
|-------------------|-----------|------------|
| LayoutLMv2-base   | 85.44%    | 67.77%     |
| LayoutXLM-base    | 89.24%    | 70.73%     |
| StrucTexT-large   | 92.29%    | **86.81%** |
| VI-LayoutXLM-base (ours) | **93.19%**    | 83.92%     |


### 5.1 优化策略

**(1) VI-LayoutXLM（Visual-feature Independent LayoutXLM）**

LayoutLMv2以及LayoutXLM中引入视觉骨干网络，用于提取视觉特征，并与后续的text embedding进行联合，作为多模态的输入embedding。但是该模块为基于`ResNet_x101_64x4d`的特征提取网络，特征抽取阶段耗时严重，因此我们将其去除，同时仍然保留文本、位置以及布局等信息，最终发现针对LayoutXLM进行改进，下游SER任务精度无损，针对LayoutLMv2进行改进，下游SER任务精度仅降低`2.1%`，而模型大小减小了约`340M`。具体消融实验如下所示。

| 模型              | 模型大小 (G) | F-score | 精度收益   |
|-----------------|----------|---------|--------|
| LayoutLMv2      | 0.76     | 84.20%  | -      |
| VI-LayoutLMv2 | 0.42     | 82.10%  | -2.10% |
| LayoutXLM       | 1.4      | 89.50%  | -      |
| VI-LayouXLM   | 1.1      | 90.46%  | +0.96%  |

同时，基于XFUND数据集，VI-LayoutXLM在RE任务上的精度也进一步提升了`1.06%`。

**(2) TB-YX排序方法（Threshold-Based YX sorting algorithm）**

文本阅读顺序对于信息抽取与文本理解等任务至关重要，传统多模态模型中，没有考虑不同OCR工具可能产生的不正确阅读顺序，而模型输入中包含位置编码，阅读顺序会直接影响预测结果，在预处理中，我们对文本行按照从上到下，从左到右（YX）的顺序进行排序，为防止文本行位置轻微干扰带来的排序结果不稳定问题，在排序的过程中，引入位置偏移阈值Th，对于Y方向距离小于Th的2个文本内容，使用x方向的位置从左到右进行排序。TB-YX排序方法伪代码如下所示。

```py
def order_by_tbyx(ocr_info, th=20):
	"""
	ocr_info: a list of dict, which contains bbox information([x1, y1, x2, y2])
	th: threshold of the position threshold
	"""
    res = sorted(ocr_info, key=lambda r: (r["bbox"][1], r["bbox"][0])) # sort using y1 first and then x1
    for i in range(len(res) - 1):
        for j in range(i, 0, -1):
            # restore the order using the
            if abs(res[j + 1]["bbox"][1] - res[j]["bbox"][1]) < th and \
                    (res[j + 1]["bbox"][0] < res[j]["bbox"][0]):
                tmp = deepcopy(res[j])
                res[j] = deepcopy(res[j + 1])
                res[j + 1] = deepcopy(tmp)
            else:
                break
    return res
```

不同排序方法的结果对比如下所示，可以看出引入偏离阈值之后，排序结果更加符合人类的阅读顺序。

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185942080-9d4bafc9-fa7f-4da4-b139-b2bd703dc76d.png" width="800">
</div>


使用该策略，最终XFUND数据集上，SER任务F1指标提升`2.06%`，RE任务F1指标提升`7.04%`。

**(3) 互学习蒸馏策略**

UDML（Unified-Deep Mutual Learning）联合互学习是PP-OCRv2与PP-OCRv3中采用的对于文本识别非常有效的提升模型效果的策略。在训练时，引入2个完全相同的模型进行互学习，计算2个模型之间的互蒸馏损失函数(DML loss)，同时对transformer中间层的输出结果计算距离损失函数(L2 loss)。使用该策略，最终XFUND数据集上，SER任务F1指标提升`0.6%`，RE任务F1指标提升`5.01%`。

最终优化后模型基于SER任务的可视化结果如下所示。

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185942213-0909135b-3bcd-4d79-9e69-847dfb1c3b82.png" width="800">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185942237-72923b42-8590-42eb-b687-fa819b1c3afd.png" width="800">
</div>


RE任务的可视化结果如下所示。


<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185942400-8920dc3c-de7f-46d0-b0bc-baca9536e0e1.png" width="800">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185942416-ca4fd8b0-9227-4c65-b969-0afbda525b85.png" width="800">
</div>

### 5.2 更多场景消融实验

我们在FUNSD数据集上，同时基于RE任务进行对本次升级策略进行验证，具体实验结果如下所示，可以看出该方案针对不同任务，在不同数据集上均有非常明显的精度收益。

#### 5.2.1 XFUND_zh数据集

**RE任务结果**

| 实验序号 | 策略                          | 模型大小(G) | F1-score |
|:------:|:------------:|:---------:|:----------:|
| 1    | LayoutXLM          | 1.4     | 70.81%   |
| 2    | VI-LayoutXLM | 1.1     | 71.87%   |
| 3    | 实验2 + PP-OCR排序              | 1.1     | 78.91%   |
| 4    | 实验3 + UDML蒸馏                | 1.1     | **83.92%**   |


#### 5.2.2 FUNSD数据集

**SER任务结果**

| 实验序号 | 策略                  | F1-score |
|:------:|:------:|:------:|
| 1    | LayoutXLM  | 82.28%   |
| 2    | PP-Structurev2 SER       | **87.79%**   |


**RE任务结果**

| 实验序号 | 策略                 | F1-score |
|:------:|:------:|:------:|
| 1    | LayoutXLM    |  53.13%  |
| 2    | PP-Structurev2 SER    | **74.87%**   |


## 6. Reference
* [1] Zhong X, ShafieiBavani E, Jimeno Yepes A. Image-based table recognition: data, model, and evaluation[C]//European Conference on Computer Vision. Springer, Cham, 2020: 564-580.
* [2] Cui C, Gao T, Wei S. Yuning Du, Ruoyu Guo, Shuilong Dong, Bin Lu, Ying Zhou, Xueying Lv, Qiwen Liu, Xiaoguang Hu, Dianhai Yu, and Yanjun Ma* [J]. Pplcnet: A lightweight cpu convolutional neural network, 2021, 3.
* [3] Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 2117-2125.
* [4] Yu G, Chang Q, Lv W, et al. PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices[J]. arXiv preprint arXiv:2111.00902, 2021.
* [5] Bochkovskiy A, Wang C Y, Liao H Y M. Yolov4: Optimal speed and accuracy of object detection[J]. arXiv preprint arXiv:2004.10934, 2020.
* [6] Ye J, Qi X, He Y, et al. PingAn-VCGroup's Solution for ICDAR 2021 Competition on Scientific Literature Parsing Task B: Table Recognition to HTML[J]. arXiv preprint arXiv:2105.01848, 2021.
* [7] Zhong X, Tang J, Yepes A J. Publaynet: largest dataset ever for document layout analysis[C]//2019 International Conference on Document Analysis and Recognition (ICDAR). IEEE, 2019: 1015-1022.
* [8] CDLA：https://github.com/buptlihang/CDLA
* [9]Gao L, Huang Y, Déjean H, et al. ICDAR 2019 competition on table detection and recognition (cTDaR)[C]//2019 International Conference on Document Analysis and Recognition (ICDAR). IEEE, 2019: 1510-1515.
* [10] Mondal A, Lipps P, Jawahar C V. IIIT-AR-13K: a new dataset for graphical object detection in documents[C]//International Workshop on Document Analysis Systems. Springer, Cham, 2020: 216-230.
* [11] Tal ocr_tabel：https://ai.100tal.com/dataset
* [12] Li M, Cui L, Huang S, et al. Tablebank: A benchmark dataset for table detection and recognition[J]. arXiv preprint arXiv:1903.01949, 2019.
* [13]Li M, Xu Y, Cui L, et al. DocBank: A benchmark dataset for document layout analysis[J]. arXiv preprint arXiv:2006.01038, 2020.
* [14] Xu Y, Li M, Cui L, et al. Layoutlm: Pre-training of text and layout for document image understanding[C]//Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020: 1192-1200.
* [15] Xu Y, Xu Y, Lv T, et al. LayoutLMv2: Multi-modal pre-training for visually-rich document understanding[J]. arXiv preprint arXiv:2012.14740, 2020.
* [16] Xu Y, Lv T, Cui L, et al. Layoutxlm: Multimodal pre-training for multilingual visually-rich document understanding[J]. arXiv preprint arXiv:2104.08836, 2021.
* [17] Xu Y, Lv T, Cui L, et al. XFUND: A Benchmark Dataset for Multilingual Visually Rich Form Understanding[C]//Findings of the Association for Computational Linguistics: ACL 2022. 2022: 3214-3224.
* [18] Jaume G, Ekenel H K, Thiran J P. Funsd: A dataset for form understanding in noisy scanned documents[C]//2019 International Conference on Document Analysis and Recognition Workshops (ICDARW). IEEE, 2019, 2: 1-6.
