# 模型训练

本文将介绍模型训练时需掌握的基本概念，和训练时的调优方法。

同时会简单介绍PaddleOCR模型训练数据的组成部分，以及如何在垂类场景中准备数据finetune模型。

- [1.配置文件说明](#配置文件)
- [2. 基本概念](#基本概念)
  * [2.1 学习率](#学习率)
  * [2.2 正则化](#正则化)
  * [2.3 评估指标](#评估指标)
- [3. 数据与垂类场景](#数据与垂类场景)
  * [3.1 训练数据](#训练数据)
  * [3.2 垂类场景](#垂类场景)
  * [3.3 自己构建数据集](#自己构建数据集)
* [4. 常见问题](#常见问题)

<a name="基本概念"></a>
## 1. 基本概念

OCR(Optical Character Recognition,光学字符识别)是指对图像进行分析识别处理，获取文字和版面信息的过程，是典型的计算机视觉任务，
通常由文本检测和文本识别两个子任务构成。

模型调优时需要关注以下参数：

<a name="学习率"></a>
### 2.1 学习率

学习率是训练神经网络的重要超参数之一，它代表在每一次迭代中梯度向损失函数最优解移动的步长。
在PaddleOCR中提供了多种学习率更新策略,可以通过配置文件修改，例如：

```
Optimizer:
  ...
  lr:
    name: Piecewise
    decay_epochs : [700, 800]
    values : [0.001, 0.0001]
    warmup_epoch: 5
```

Piecewise 代表分段常数衰减，在不同的学习阶段指定不同的学习率，在每段内学习率相同。
warmup_epoch 代表在前5个epoch中，学习率将逐渐从0增加到base_lr。全部策略可以参考代码[learning_rate.py](../../ppocr/optimizer/learning_rate.py) 。

<a name="正则化"></a>
### 2.2 正则化

正则化可以有效的避免算法过拟合，PaddleOCR中提供了L1、L2正则方法，L1 和 L2 正则化是最常用的正则化方法。L1 正则化向目标函数添加正则化项，以减少参数的绝对值总和；而 L2 正则化中，添加正则化项的目的在于减少参数平方的总和。配置方法如下：

```
Optimizer:
  ...
  regularizer:
    name: L2
    factor: 2.0e-05
```

<a name="评估指标"></a>
### 2.3 评估指标

（1）检测阶段：先按照检测框和标注框的IOU评估，IOU大于某个阈值判断为检测准确。这里检测框和标注框不同于一般的通用目标检测框，是采用多边形进行表示。检测准确率：正确的检测框个数在全部检测框的占比，主要是判断检测指标。检测召回率：正确的检测框个数在全部标注框的占比，主要是判断漏检的指标。

（2）识别阶段： 字符识别准确率，即正确识别的文本行占标注的文本行数量的比例，只有整行文本识别对才算正确识别。

（3）端到端统计： 端对端召回率：准确检测并正确识别文本行在全部标注文本行的占比； 端到端准确率：准确检测并正确识别文本行在 检测到的文本行数量 的占比； 准确检测的标准是检测框与标注框的IOU大于某个阈值，正确识别的的检测框中的文本与标注的文本相同。

<a name="数据与垂类场景"></a>

## 3. 数据与垂类场景

<a name="训练数据"></a>
### 3.1 训练数据
目前开源的模型，数据集和量级如下：

    - 检测：  
        - 英文数据集，ICDAR2015  
        - 中文数据集，LSVT街景数据集训练数据3w张图片
    
    - 识别：  
        - 英文数据集，MJSynth和SynthText合成数据，数据量上千万。  
        - 中文数据集，LSVT街景数据集根据真值将图crop出来，并进行位置校准，总共30w张图像。此外基于LSVT的语料，合成数据500w。
        - 小语种数据集，使用不同语料和字体，分别生成了100w合成数据集，并使用ICDAR-MLT作为验证集。

其中，公开数据集都是开源的，用户可自行搜索下载，也可参考[中文数据集](./datasets.md)，合成数据暂不开源，用户可使用开源合成工具自行合成，可参考的合成工具包括[text_renderer](https://github.com/Sanster/text_renderer) 、[SynthText](https://github.com/ankush-me/SynthText) 、[TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) 等。

<a name="垂类场景"></a>
### 3.2 垂类场景

PaddleOCR主要聚焦通用OCR，如果有垂类需求，您可以用PaddleOCR+垂类数据自己训练；
如果缺少带标注的数据，或者不想投入研发成本，建议直接调用开放的API，开放的API覆盖了目前比较常见的一些垂类。

<a name="自己构建数据集"></a>

### 3.3 自己构建数据集

在构建数据集时有几个经验可供参考：

（1） 训练集的数据量：

    a. 检测需要的数据相对较少，在PaddleOCR模型的基础上进行Fine-tune，一般需要500张可达到不错的效果。
    b. 识别分英文和中文，一般英文场景需要几十万数据可达到不错的效果，中文则需要几百万甚至更多。


（2）当训练数据量少时，可以尝试以下三种方式获取更多的数据：

    a. 人工采集更多的训练数据，最直接也是最有效的方式。
    b. 基于PIL和opencv基本图像处理或者变换。例如PIL中ImageFont, Image, ImageDraw三个模块将文字写到背景中，opencv的旋转仿射变换，高斯滤波等。
    c. 利用数据生成算法合成数据，例如pix2pix或StyleText等算法。

<a name="常见问题"></a>

## 4. 常见问题

**Q**：训练CRNN识别时，如何选择合适的网络输入shape？

    A：一般高度采用32，最长宽度的选择，有两种方法：
    
    （1）统计训练样本图像的宽高比分布。最大宽高比的选取考虑满足80%的训练样本。
    
    （2）统计训练样本文字数目。最长字符数目的选取考虑满足80%的训练样本。然后中文字符长宽比近似认为是1，英文认为3：1，预估一个最长宽度。

**Q**：识别训练时，训练集精度已经到达90了，但验证集精度一直在70，涨不上去怎么办？

    A：训练集精度90，测试集70多的话，应该是过拟合了，有两个可尝试的方法：
    
    （1）加入更多的增广方式或者调大增广prob的[概率](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppocr/data/imaug/rec_img_aug.py#L341)，默认为0.4。
    
    （2）调大系统的[l2 dcay值](https://github.com/PaddlePaddle/PaddleOCR/blob/a501603d54ff5513fc4fc760319472e59da25424/configs/rec/ch_ppocr_v1.1/rec_chinese_lite_train_v1.1.yml#L47)

**Q**: 识别模型训练时，loss能正常下降，但acc一直为0

    A：识别模型训练初期acc为0是正常的，多训一段时间指标就上来了。


***
具体的训练教程可点击下方链接跳转：  
- [文本检测模型训练](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/detection.md)  
- [文本识别模型训练](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/recognition.md)  
- [文本方向分类器训练](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/angle_class.md)  