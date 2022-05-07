[English](../doc_en/PP-OCRv3_introduction_en.md) | 简体中文

# PP-OCRv3

- [1. 简介](#1)
- [2. 检测优化](#2)
- [3. 识别优化](#3)
- [4. 端到端评估](#4)


<a name="1"></a>
## 1. 简介

PP-OCRv3在PP-OCRv2的基础上进一步升级。整体的框架图保持了与PP-OCRv2相同的pipeline，针对检测模型和识别模型进行了优化。其中，检测模型仍基于DB模型优化，而识别模型不再采用CRNN，换成了会议IJCAI 2022中的最新方法[SVTR](https://arxiv.org/abs/2205.00159)，PP-OCRv3系统框图如下所示（粉色框中为PP-OCRv3新增策略）：

<div align="center">
    <img src="../ppocrv3_framework.png" width="800">
</div>


从算法改进思路上看，分别针对检测和识别模型，进行了共八个方面的改进：


- 检测模型优化：
    - LK-PAN：增大感受野的PAN模块；
    - DML：教师模型互学习策略；
    - RSE-FPN：带残差注意力机制的FPN模块；


- 识别模型优化：
    - SVTR_LCNet：轻量级文本识别网络；
    - GTC：Attention指导CTC训练策略；
    - TextConAug：丰富图像上下文信息的数据增广策略；
    - TextRotNet：自监督的预训练模型；
    - UIM：无标签数据挖掘方案。

从效果上看，速度可比情况下，多种场景精度均有大幅提升：
- 中文场景，相对于PP-OCRv2中文模型提升超5%；
- 英文数字场景，相比于PP-OCRv2英文模型提升11%；
- 多语言场景，优化80+语种识别效果，平均准确率提升超5%。


<a name="2"></a>
## 2. 检测优化

PP-OCRv3检测模型整体训练方案仍采用PP-OCRv2的[CML](https://arxiv.org/pdf/2109.03144.pdf)蒸馏策略，CML蒸馏包含一个教师模型和两个学生模型，在训练过程中，教师模型不参与训练，学生模型受到来自标签和教师模型的监督，同时两个学生模型互相学习。PP-OCRv3分别针对教师模型、学生模型进一步优化。其中，在对教师模型优化时，采用了增大感受野的PAN模块LK-PAN和DML蒸馏策略；在对学生模型优化时，采用了带残差注意力机制的FPN模块RSE-FPN。

PP-OCRv3 CML蒸馏训练框架图如下：

<div align="center">
    <img src=".././ppocr_v3/ppocrv3_det_cml.png" width="800">
</div>

消融实验如下：

|序号|策略|模型大小|hmean|速度（cpu + mkldnn)|
|-|-|-|-|-|
|baseline teacher|PP-OCR server|49M|83.2%|171ms|
|teacher1|DB-R50-LK-PAN|124M|85.0%|396ms|
|teacher2|DB-R50-LK-PAN-DML|124M|86.0%|396ms|
|baseline student|PP-OCRv2|3M|83.2%|117ms|
|student0|DB-MV3-RSE-FPN|3.6M|84.5%|124ms|
|student1|DB-MV3-CML（teacher2）|3M|84.3%|117ms|
|student2|DB-MV3-RSE-FPN-CML（teacher2）|3.6M|85.4%|124ms|

测试环境： Intel Gold 6148 CPU，预测时开启MKLDNN加速。

**（1）增大感受野的PAN模块LK-PAN（Large Kernel PAN）**

LK-PAN(Large Kernel PAN)是一个具有更大感受野的轻量级[PAN](https://arxiv.org/pdf/1803.01534.pdf)结构。在LK-PAN的path augmentation中，使用卷积核为`9*9`的卷积；更大的卷积核意味着更大的感受野，更容易检测大字体的文字以及极端长宽比的文字。LK-PAN将PP-OCR server检测模型的hmean从83.2%提升到85.0%。

<div align="center">
    <img src="../ppocr_v3/LKPAN.png" width="1000">
</div>

**（2）DML（Deep Mutual Learning）蒸馏进一步提升teacher模型精度。**

[DML](https://arxiv.org/abs/1706.00384) 互学习蒸馏方法，通过两个结构相同的模型互相学习，相比于传统的教师模型监督学生模型的蒸馏方法，DML 摆脱了对大的教师模型的依赖，蒸馏训练的流程更加简单。在PP-OCRv3的检测模型训练中，使用DML蒸馏策略进一步提升教师模型的精度，并使用ResNet50作为Backbone。DML策略将教师模型的Hmean从85%进一步提升至86%。

教师模型DML训练流程图如下：

<div align="center">
    <img src="../ppocr_v3/teacher_dml.png" width="800">
</div>

**（3）带残差注意力机制的FPN模块RSE-FPN（Residual SE-FPN）。**

残差结构的通道注意力模块RSE-FPN结构如下图所示，RSE-FPN在PP-OCRv2的FPN基础上，将FPN中的卷积层更换为通道注意力结构的RSEConv层。考虑到PP-OCRv2的FPN通道数仅为96和24，如果直接用SEblock代替FPN中卷积会导致某些通道的特征被抑制，进而导致精度下降，RSEConv引入残差结构防止训练中包含重要特征的通道被抑制。直接添加RSE-FPN模块，可将PP-OCR检测模型的精度Hmean从81.3%提升到84.5%。在学生模型中加入RSE-FPN后进行CML蒸馏，比不加时，Hmean指标从83.2提升到84.3%。

<div align="center">
    <img src=".././ppocr_v3/RSEFPN.png" width="1000">
</div>


<a name="3"></a>
## 3. 识别优化

PP-OCRv3识别模型从网络结构、训练策略、数据增广等多个方面进行了优化，PP-OCRv3系统流程图如下所示：

<div align="center">
    <img src="../ppocr_v3/v3_rec_pipeline.png" width=800>
</div>

上图中，蓝色方块中列举了PP-OCRv3识别模型的6个主要模块。首先在模块①，将base模型从CRNN替换为精度更高的单一视觉模型[SVTR](https://arxiv.org/abs/2205.00159)，并进行一系列的结构优化进行加速，得到全新的轻量级文本识别网络SVTR_LCNet（如图中红色虚线框所示）；在模块②，借鉴[GTC](https://arxiv.org/pdf/2002.01276.pdf)策略，引入Attention指导CTC训练，进一步提升模型精度；在模块③，使用基于上下文信息的数据增广策略TextConAug，丰富训练数据上下文信息，提升训练数据多样性；在模块④，使用TextRotNet训练自监督的预训练模型，充分利用无标注识别数据的信息；模块⑤基于PP-OCRv2中提出的UDML蒸馏策略进行蒸馏学习，除计算2个模型的CTC分支的DMLLoss外，也计算2个模型的Attention分支之间的DMLLoss，从而得到更优模型；在模块⑥中，基于UIM无标注数据挖掘方法，使用效果好但速度相对较慢的SVTR_tiny模型进行无标签数据挖掘，为模型训练增加更多真实数据。


基于上述策略，PP-OCRv3识别模型相比PP-OCRv2，在速度可比的情况下，精度进一步提升4.6%。 具体消融实验如下所示：

| ID | 策略 |  模型大小 | 精度 | 预测耗时（CPU + MKLDNN)|
|-----|-----|--------|----| --- |
| 01 | PP-OCRv2 | 8M | 74.8% | 8.54ms |
| 02 | SVTR_Tiny | 21M | 80.1% | 97ms |
| 03 | SVTR_LCNet | 12M | 71.9% | 6.6ms |
| 04 | + GTC | 12M | 75.8% | 7.6ms |
| 05 | + TextConAug | 12M | 76.3% | 7.6ms |
| 06 | + TextRotNet | 12M | 76.9% | 7.6ms |
| 07 | + UDML | 12M | 78.4% | 7.6ms |
| 08 | + UIM | 12M | 79.4% | 7.6ms |

注： 测试速度时，实验01-03输入图片尺寸均为(3,32,320)，04-08输入图片尺寸均为(3,48,320)。在实际预测时，图像为变长输入，速度会有所变化。


**（1）轻量级文本识别网络SVTR_LCNet。**

PP-OCRv3将base模型从CRNN替换成了[SVTR](https://arxiv.org/abs/2205.00159)，SVTR证明了强大的单视觉模型（无需序列模型）即可高效准确完成文本识别任务，在中英文数据上均有优秀的表现。经过实验验证，SVTR_Tiny 在自建的[中文数据集](https://arxiv.org/abs/2109.03144)上 ，识别精度可以提升至80.1%，SVTR_Tiny 网络结构如下所示：

<div align="center">
    <img src="../ppocr_v3/svtr_tiny.png" width=800>
</div>


由于 MKLDNN 加速库支持的模型结构有限，SVTR 在 CPU+MKLDNN 上相比 PP-OCRv2 慢了10倍。PP-OCRv3 期望在提升模型精度的同时，不带来额外的推理耗时。通过分析发现，SVTR_Tiny 结构的主要耗时模块为 Mixing Block，因此我们对 SVTR_Tiny 的结构进行了一系列优化（详细速度数据请参考下方消融实验表格）:


1. 将 SVTR 网络前半部分替换为 PP-LCNet 的前三个stage，保留4个 Global Mixing Block ，精度为76%，加速69%，网络结构如下所示：
<div align="center">
    <img src="../ppocr_v3/svtr_g4.png" width=800>
</div>
2. 将4个 Global Mixing Block 减小到2个，精度为72.9%，加速69%，网络结构如下所示：
<div align="center">
    <img src="../ppocr_v3/svtr_g2.png" width=800>
</div>
3. 实验发现 Global Mixing Block 的预测速度与输入其特征的shape有关，因此后移 Global Mixing Block 的位置到池化层之后，精度下降为71.9%，速度超越基于CNN结构的PP-OCRv2-baseline 22%，网络结构如下所示：
<div align="center">
    <img src="../ppocr_v3/LCNet_SVTR.png" width=800>
</div>

具体消融实验如下所示：

| ID | 策略 |  模型大小 | 精度 | 速度（CPU + MKLDNN)|
|-----|-----|--------|----| --- |
| 01 | PP-OCRv2-baseline | 8M | 69.3%  | 8.54ms |
| 02 | SVTR_Tiny | 21M | 80.1% | 97ms |
| 03 | SVTR_LCNet(G4) | 9.2M | 76% | 30ms |
| 04 | SVTR_LCNet(G2) | 13M | 72.98% | 9.37ms |
| 05 | SVTR_LCNet | 12M | 71.9% | 6.6ms |

注： 测试速度时，输入图片尺寸均为(3,32,320)； PP-OCRv2-baseline 代表没有借助蒸馏方法训练得到的模型

**（2）采用Attention指导CTC训练。**

为了提升模型精度同时不引入额外推理成本，PP-OCRv3 参考 GTC(Guided Training of CTC) 策略，使用 Attention 监督 CTC 训练，预测时完全去除 Attention 模块，在推理阶段不增加任何耗时, 精度提升3.8%，训练流程如下所示：
<div align="center">
    <img src="../ppocr_v3/GTC.png" width=800>
</div>

**（3）TextConAug数据增广策略。**

在论文[ConCLR](https://www.cse.cuhk.edu.hk/~byu/papers/C139-AAAI2022-ConCLR.pdf)中，作者提出ConAug数据增广，在一个batch内对2张不同的图像进行联结，组成新的图像并进行自监督对比学习。PP-OCRv3将此方法应用到有监督的学习任务中，设计了TextConAug数据增强方法，支持更多图像的联结，从而进一步丰富了图像的上下文信息。最终将识别模型精度进一步提升0.5%。TextConAug示意图如下所示：

<div align="center">
    <img src="../ppocr_v3/recconaug.png" width=800>
</div>


**（4）TextRotNet自监督训练优化预训练模型。**

为了充分利用自然场景中的大量无标注文本数据，PP-OCRv3参考论文[STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels)，设计TextRotNet自监督任务，对识别图像进行旋转并预测其旋转角度，同时结合中文场景文字识别任务的特点，在训练时适当调整图像的尺寸，添加文本识别数据增广，最终产出针对文本识别任务的PP-LCNet预训练模型，帮助识别模型精度进一步提升0.6%。TextRotNet训练流程如下图所示：

<div align="center">
    <img src="../ppocr_v3/SSL.png" width="500"> 
</div>


**（5）UIM（Unlabeled Images Mining）无标注数据挖掘策略。**

为更直接利用自然场景中包含大量无标注数据，使用PP-OCRv2检测模型以及SVTR_tiny识别模型对百度开源的40W [LSVT弱标注数据集](https://ai.baidu.com/broad/introduction?dataset=lsvt)进行检测与识别，并筛选出识别得分大于0.95的文本，共81W文本行数据，将其补充到训练数据中，最终进一步提升模型精度1.0%。

<div align="center">
    <img src="../ppocr_v3/UIM.png" width="500"> 
</div>


<a name="4"></a>
## 4. 端到端评估

经过以上优化，最终PP-OCRv3在速度可比情况下，中文场景端到端Hmean指标相比于PP-OCRv2提升5%，效果大幅提升。具体指标如下表所示：

| Model | Hmean |  Model Size (M) | Time Cost (CPU, ms) | Time Cost (T4 GPU, ms) |
|-----|-----|--------|----| --- |
| PP-OCR mobile | 50.3% | 8.1 | 356  | 116 |
| PP-OCR server | 57.0% | 155.1 | 1056 | 200 |
| PP-OCRv2 | 57.6% | 11.6 | 330 | 111 |
| PP-OCRv3 | 62.9% | 15.6 | 331 | 86.64 |

测试环境：CPU型号为Intel Gold 6148，CPU预测时开启MKLDNN加速。


除了更新中文模型，本次升级也同步优化了英文数字模型，端到端效果提升11%，如下表所示：

| Model | Recall |  Precision | Hmean |
|-----|-----|--------|----|
| PP-OCR_en | 38.99% | 45.91% | 42.17%  |
| PP-OCRv3_en | 50.95% | 55.53% | 53.14% |

同时，也对已支持的80余种语言识别模型进行了升级更新，在有评估集的四种语系识别准确率平均提升5%以上，如下表所示：

| Model | 拉丁语系 |  阿拉伯语系 | 日语 | 韩语 |
|-----|-----|--------|----| --- |
| PP-OCR_mul | 69.6% | 40.5% | 38.5%  | 55.4% |
| PP-OCRv3_mul | 75.2%| 45.37% | 45.8% | 60.1% |
