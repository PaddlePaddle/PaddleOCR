# FAQ

## 写在前面

- 我们收集整理了开源以来在issues和用户群中的常见问题并且给出了简要解答，旨在为OCR的开发者提供一些参考，也希望帮助大家少走一些弯路。

- OCR领域大佬众多，本文档回答主要依赖有限的项目实践，难免挂一漏万，如有遗漏和不足，也**希望有识之士帮忙补充和修正**，万分感谢。


## PaddleOCR常见问题汇总(持续更新)

* [近期更新（2021.1.4）](#近期更新)
* [【精选】OCR精选10个问题](#OCR精选10个问题)
* [【理论篇】OCR通用32个问题](#OCR通用问题)
  * [基础知识7题](#基础知识)
  * [数据集7题](#数据集2)
  * [模型训练调优18题](#模型训练调优2)
* [【实战篇】PaddleOCR实战100个问题](#PaddleOCR实战问题)
  * [使用咨询31题](#使用咨询)
  * [数据集17题](#数据集3)
  * [模型训练调优26题](#模型训练调优3)
  * [预测部署26题](#预测部署3)


<a name="近期更新"></a>
## 近期更新（2021.1.4）

#### Q2.3.18: 请问有哪些修改骨干网络的技巧？

**A**: 可以参考HS-ResNet这篇文章:https://arxiv.org/pdf/2010.07621.pdf

#### Q3.1.29: PPOCRLabel创建矩形框时只能拖出正方形，如何进行矩形标注？

**A** 取消勾选：“编辑”-“正方形标注”

#### Q3.1.30: Style-Text 如何不文字风格迁移，就像普通文本生成程序一样默认字体直接输出到分割的背景图？

**A** 使用image_synth模式会输出fake_bg.jpg，即为背景图。如果想要批量提取背景，可以稍微修改一下代码，将fake_bg保存下来即可。要修改的位置：
https://github.com/PaddlePaddle/PaddleOCR/blob/de3e2e7cd3b8b65ee02d7a41e570fa5b511a3c1d/StyleText/engine/synthesisers.py#L68

#### Q3.1.31: 怎么输出网络结构以及每层的参数信息？

**A** 可以使用 `paddle.summary`， 具体参考:https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/paddle/hapi/model_summary/summary_cn.html#summary。

### Q3.4.26: 目前paddle hub serving 只支持 imgpath，如果我想用imgurl 去哪里改呢？

**A**: 图片是在这里读取的：https://github.com/PaddlePaddle/PaddleOCR/blob/67ef25d593c4eabfaaceb22daade4577f53bed81/deploy/hubserving/ocr_system/module.py#L55，
可以参考下面的写法，将url path转化为np array（https://cloud.tencent.com/developer/article/1467840）
```
response = request.urlopen('http://i1.whymtj.com/uploads/tu/201902/9999/52491ae4ba.jpg')
img_array = np.array(bytearray(response.read()), dtype=np.uint8)
img = cv.imdecode(img_array, -1)
```

<a name="OCR精选10个问题"></a>
## 【精选】OCR精选10个问题

#### Q1.1.1：基于深度学习的文字检测方法有哪几种？各有什么优缺点？

**A**：常用的基于深度学习的文字检测方法一般可以分为基于回归的、基于分割的两大类，当然还有一些将两者进行结合的方法。

（1）基于回归的方法分为box回归和像素值回归。a. 采用box回归的方法主要有CTPN、Textbox系列和EAST，这类算法对规则形状文本检测效果较好，但无法准确检测不规则形状文本。 b. 像素值回归的方法主要有CRAFT和SA-Text，这类算法能够检测弯曲文本且对小文本效果优秀但是实时性能不够。

（2）基于分割的算法，如PSENet，这类算法不受文本形状的限制，对各种形状的文本都能取得较好的效果，但是往往后处理比较复杂，导致耗时严重。目前也有一些算法专门针对这个问题进行改进，如DB，将二值化进行近似，使其可导，融入训练，从而获取更准确的边界，大大降低了后处理的耗时。

#### Q1.1.2：对于中文行文本识别，CTC和Attention哪种更优？

**A**：（1）从效果上来看，通用OCR场景CTC的识别效果优于Attention，因为带识别的字典中的字符比较多，常用中文汉字三千字以上，如果训练样本不足的情况下，对于这些字符的序列关系挖掘比较困难。中文场景下Attention模型的优势无法体现。而且Attention适合短语句识别，对长句子识别比较差。

（2）从训练和预测速度上，Attention的串行解码结构限制了预测速度，而CTC网络结构更高效，预测速度上更有优势。

#### Q1.1.3：弯曲形变的文字识别需要怎么处理？TPS应用场景是什么，是否好用？

**A**：（1）在大多数情况下，如果遇到的场景弯曲形变不是太严重，检测4个顶点，然后直接通过仿射变换转正识别就足够了。

（2）如果不能满足需求，可以尝试使用TPS（Thin Plate Spline），即薄板样条插值。TPS是一种插值算法，经常用于图像变形等，通过少量的控制点就可以驱动图像进行变化。一般用在有弯曲形变的文本识别中，当检测到不规则的/弯曲的（如，使用基于分割的方法检测算法）文本区域，往往先使用TPS算法对文本区域矫正成矩形再进行识别，如，STAR-Net、RARE等识别算法中引入了TPS模块。
**Warning**：TPS看起来美好，在实际应用时经常发现并不够鲁棒，并且会增加耗时，需要谨慎使用。

#### Q1.1.4：简单的对于精度要求不高的OCR任务，数据集需要准备多少张呢？

**A**：（1）训练数据的数量和需要解决问题的复杂度有关系。难度越大，精度要求越高，则数据集需求越大，而且一般情况实际中的训练数据越多效果越好。

（2）对于精度要求不高的场景，检测任务和识别任务需要的数据量是不一样的。对于检测任务，500张图像可以保证基本的检测效果。对于识别任务，需要保证识别字典中每个字符出现在不同场景的行文本图像数目需要大于200张（举例，如果有字典中有5个字，每个字都需要出现在200张图片以上，那么最少要求的图像数量应该在200-1000张之间），这样可以保证基本的识别效果。

#### Q1.1.5：背景干扰的文字（如印章盖到落款上，需要识别落款或者印章中的文字），如何识别？

**A**：（1）在人眼确认可识别的条件下，对于背景有干扰的文字，首先要保证检测框足够准确，如果检测框不准确，需要考虑是否可以通过过滤颜色等方式对图像预处理并且增加更多相关的训练数据；在识别的部分，注意在训练数据中加入背景干扰类的扩增图像。

（2）如果MobileNet模型不能满足需求，可以尝试ResNet系列大模型来获得更好的效果
。

#### Q1.1.6：OCR领域常用的评估指标是什么？

**A**：对于两阶段的可以分开来看，分别是检测和识别阶段

（1）检测阶段：先按照检测框和标注框的IOU评估，IOU大于某个阈值判断为检测准确。这里检测框和标注框不同于一般的通用目标检测框，是采用多边形进行表示。检测准确率：正确的检测框个数在全部检测框的占比，主要是判断检测指标。检测召回率：正确的检测框个数在全部标注框的占比，主要是判断漏检的指标。


（2）识别阶段：
字符识别准确率，即正确识别的文本行占标注的文本行数量的比例，只有整行文本识别对才算正确识别。

（3）端到端统计：
端对端召回率：准确检测并正确识别文本行在全部标注文本行的占比；
端到端准确率：准确检测并正确识别文本行在 检测到的文本行数量 的占比；
准确检测的标准是检测框与标注框的IOU大于某个阈值，正确识别的的检测框中的文本与标注的文本相同。


#### Q1.1.7：单张图上多语种并存识别（如单张图印刷体和手写文字并存），应该如何处理？

**A**：单张图像中存在多种类型文本的情况很常见，典型的以学生的试卷为代表，一张图像同时存在手写体和印刷体两种文本，这类情况下，可以尝试”1个检测模型+1个N分类模型+N个识别模型”的解决方案。
其中不同类型文本共用同一个检测模型，N分类模型指额外训练一个分类器，将检测到的文本进行分类，如手写+印刷的情况就是二分类，N种语言就是N分类，在识别的部分，针对每个类型的文本单独训练一个识别模型，如手写+印刷的场景，就需要训练一个手写体识别模型，一个印刷体识别模型，如果一个文本框的分类结果是手写体，那么就传给手写体识别模型进行识别，其他情况同理。

#### Q1.1.8：请问PaddleOCR项目中的中文超轻量和通用模型用了哪些数据集？训练多少样本，gpu什么配置，跑了多少个epoch，大概跑了多久？

**A**：
（1）检测的话，LSVT街景数据集共3W张图像，超轻量模型，150epoch左右，2卡V100 跑了不到2天；通用模型：2卡V100 150epoch 不到4天。
（2）
识别的话，520W左右的数据集（真实数据26W+合成数据500W）训练，超轻量模型：4卡V100，总共训练了5天左右。通用模型：4卡V100，共训练6天。

超轻量模型训练分为2个阶段：
(1)全量数据训练50epoch，耗时3天
(2)合成数据+真实数据按照1:1数据采样，进行finetune训练200epoch，耗时2天

通用模型训练：
真实数据+合成数据，动态采样(1：1)训练，200epoch，耗时 6天左右。


#### Q1.1.9：PaddleOCR模型推理方式有几种？各自的优缺点是什么

**A**：目前推理方式支持基于训练引擎推理和基于预测引擎推理。

（1）基于训练引擎推理不需要转换模型，但是需要先组网再load参数，语言只支持python，不适合系统集成。

（2）基于预测引擎的推理需要先转换模型为inference格式，然后可以进行不需要组网的推理，语言支持c++和python，适合系统集成。

#### Q1.1.10：PaddleOCR中，对于模型预测加速，CPU加速的途径有哪些？基于TenorRT加速GPU对输入有什么要求？

**A**：（1）CPU可以使用mkldnn进行加速；对于python inference的话，可以把enable_mkldnn改为true，[参考代码](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/tools/infer/utility.py#L84)，对于cpp inference的话，在配置文件里面配置use_mkldnn 1即可，[参考代码](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/deploy/cpp_infer/tools/config.txt#L6)

（2）GPU需要注意变长输入问题等，TRT6 之后才支持变长输入

<a name="OCR通用问题"></a>
## 【理论篇】OCR通用问题

<a name="基础知识"></a>
### 基础知识

#### Q2.1.1：CRNN能否识别两行的文字?还是说必须一行？

**A**：CRNN是一种基于1D-CTC的算法，其原理决定无法识别2行或多行的文字，只能单行识别。

#### Q2.1.2：怎么判断行文本图像是否是颠倒的？

**A**：有两种方案：（1）原始图像和颠倒图像都进行识别预测，取得分较高的为识别结果。
（2）训练一个正常图像和颠倒图像的方向分类器进行判断。

#### Q2.1.3：目前OCR普遍是二阶段，端到端的方案在业界落地情况如何？

**A**：端到端在文字分布密集的业务场景，效率会比较有保证，精度的话看自己业务数据积累情况，如果行级别的识别数据积累比较多的话two-stage会比较好。百度的落地场景，比如工业仪表识别、车牌识别都用到端到端解决方案。

#### Q2.1.4 印章如何识别
**A**: 1. 使用带tps的识别网络或abcnet,2.使用极坐标变换将图片拉平之后使用crnn

#### Q2.1.5 多语言的字典里是混合了不同的语种，这个是有什么讲究吗？统一到一个字典里会对精度造成多大的损失？
**A**：统一到一个字典里，会造成最后一层FC过大，增加模型大小。如果有特殊需求的话，可以把需要的几种语言合并字典训练模型，合并字典之后如果引入过多的形近字，可能会造成精度损失，字符平衡的问题可能也需要考虑一下。在PaddleOCR里暂时将语言字典分开。

#### Q2.1.6 预处理部分，图片的长和宽为什么要处理成32的倍数？
**A**：以检测中的resnet骨干网络为例，图像输入网络之后，需要经过5次2倍降采样，共32倍，因此建议输入的图像尺寸为32的倍数。

#### Q2.1.7：类似泰语这样的小语种，部分字会占用两个字符甚至三个字符，请问如何制作字典。

**A**：处理字符的时候，把多字符的当作一个字就行，字典中每行是一个字。

<a name="数据集2"></a>
### 数据集

#### Q2.2.1：支持空格的模型，标注数据的时候是不是要标注空格？中间几个空格都要标注出来么？

**A**：如果需要检测和识别模型，就需要在标注的时候把空格标注出来，而且在字典中增加空格对应的字符。标注过程中，如果中间几个空格标注一个就行。

#### Q2.2.2：如果考虑支持竖排文字识别，相关的数据集如何合成？

**A**：竖排文字与横排文字合成方式相同，只是选择了垂直字体。合成工具推荐：[text_renderer](https://github.com/Sanster/text_renderer)

#### Q2.2.3：训练文字识别模型，真实数据有30w，合成数据有500w，需要做样本均衡吗？

**A**：需要，一般需要保证一个batch中真实数据样本和合成数据样本的比例是1：1~1：3左右效果比较理想。如果合成数据过大，会过拟合到合成数据，预测效果往往不佳。还有一种**启发性**的尝试是可以先用大量合成数据训练一个base模型，然后再用真实数据微调，在一些简单场景效果也是会有提升的。

#### Q2.2.4：请问一下，竖排文字识别时候，字的特征已经变了，这种情况在数据集和字典标注是新增一个类别还是多个角度的字共享一个类别？

**A**：可以根据实际场景做不同的尝试，共享一个类别是可以收敛，效果也还不错。但是如果分开训练，同类样本之间一致性更好，更容易收敛，识别效果会更优。

#### Q2.2.5： 文本行较紧密的情况下如何准确检测？

**A**：使用基于分割的方法，如DB，检测密集文本行时，最好收集一批数据进行训练，并且在训练时，并将生成二值图像的shrink_ratio参数调小一些。

#### Q2.2.6: 当训练数据量少时，如何获取更多的数据？

**A**: 当训练数据量少时，可以尝试以下三种方式获取更多的数据：（1）人工采集更多的训练数据，最直接也是最有效的方式。（2）基于PIL和opencv基本图像处理或者变换。例如PIL中ImageFont, Image, ImageDraw三个模块将文字写到背景中，opencv的旋转仿射变换，高斯滤波等。（3）利用数据生成算法合成数据，例如pix2pix等算法。

#### Q2.2.7: 论文《Editing Text in the Wild》中文本合成方法SRNet有什么特点？

**A**: SRNet是借鉴GAN中图像到图像转换、风格迁移的想法合成文本数据。不同于通用GAN的方法只选择一个分支，SRNet将文本合成任务分解为三个简单的子模块，提升合成数据的效果。这三个子模块为不带背景的文本风格迁移模块、背景抽取模块和融合模块。PaddleOCR计划将在2020年12月中旬开源基于SRNet的实用模型。

<a name="模型训练调优2"></a>
### 模型训练调优

#### Q2.3.1：如何更换文本检测/识别的backbone？
**A**：无论是文字检测，还是文字识别，骨干网络的选择是预测效果和预测效率的权衡。一般，选择更大规模的骨干网络，例如ResNet101_vd，则检测或识别更准确，但预测耗时相应也会增加。而选择更小规模的骨干网络，例如MobileNetV3_small_x0_35，则预测更快，但检测或识别的准确率会大打折扣。幸运的是不同骨干网络的检测或识别效果与在ImageNet数据集图像1000分类任务效果正相关。[**飞桨图像分类套件PaddleClas**](https://github.com/PaddlePaddle/PaddleClas)汇总了ResNet_vd、Res2Net、HRNet、MobileNetV3、GhostNet等23种系列的分类网络结构，在上述图像分类任务的top1识别准确率，GPU(V100和T4)和CPU(骁龙855)的预测耗时以及相应的[**117个预训练模型下载地址**](https://paddleclas.readthedocs.io/zh_CN/latest/models/models_intro.html)。

 （1）文字检测骨干网络的替换，主要是确定类似与ResNet的4个stages，以方便集成后续的类似FPN的检测头。此外，对于文字检测问题，使用ImageNet训练的分类预训练模型，可以加速收敛和效果提升。

 （2）文字识别的骨干网络的替换，需要注意网络宽高stride的下降位置。由于文本识别一般宽高比例很大，因此高度下降频率少一些，宽度下降频率多一些。可以参考PaddleOCR中[MobileNetV3骨干网络](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/modeling/backbones/rec_mobilenet_v3.py)的改动。

#### Q2.3.2：文本识别训练不加LSTM是否可以收敛？

**A**：理论上是可以收敛的，加上LSTM模块主要是为了挖掘文字之间的序列关系，提升识别效果。对于有明显上下文语义的场景效果会比较明显。

#### Q2.3.3：文本识别中LSTM和GRU如何选择？

**A**：从项目实践经验来看，序列模块采用LSTM的识别效果优于GRU，但是LSTM的计算量比GRU大一些，可以根据自己实际情况选择。

#### Q2.3.4：对于CRNN模型，backbone采用DenseNet和ResNet_vd，哪种网络结构更好？

**A**：Backbone的识别效果在CRNN模型上的效果，与Imagenet 1000 图像分类任务上识别效果和效率一致。在图像分类任务上ResnNet_vd（79%+）的识别精度明显优于DenseNet（77%+），此外对于GPU，Nvidia针对ResNet系列模型做了优化，预测效率更高，所以相对而言，resnet_vd是较好选择。如果是移动端，可以优先考虑MobileNetV3系列。

#### Q2.3.5：训练识别时，如何选择合适的网络输入shape？

**A**：一般高度采用32，最长宽度的选择，有两种方法：

（1）统计训练样本图像的宽高比分布。最大宽高比的选取考虑满足80%的训练样本。

（2）统计训练样本文字数目。最长字符数目的选取考虑满足80%的训练样本。然后中文字符长宽比近似认为是1，英文认为3：1，预估一个最长宽度。

#### Q2.3.6：如何识别文字比较长的文本？

**A**：在中文识别模型训练时，并不是采用直接将训练样本缩放到[3,32,320]进行训练，而是先等比例缩放图像，保证图像高度为32，宽度不足320的部分补0，宽高比大于10的样本直接丢弃。预测时，如果是单张图像预测，则按上述操作直接对图像缩放，不做宽度320的限制。如果是多张图预测，则采用batch方式预测，每个batch的宽度动态变换，采用这个batch中最长宽度。

#### Q2.3.7：识别训练时，训练集精度已经到达90了，但验证集精度一直在70，涨不上去怎么办？

**A**：训练集精度90，测试集70多的话，应该是过拟合了，有两个可尝试的方法：

（1）加入更多的增广方式或者调大增广prob的[概率](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppocr/data/imaug/rec_img_aug.py#L341)，默认为0.4。

（2）调大系统的[l2 dcay值](https://github.com/PaddlePaddle/PaddleOCR/blob/a501603d54ff5513fc4fc760319472e59da25424/configs/rec/ch_ppocr_v1.1/rec_chinese_lite_train_v1.1.yml#L47)

#### Q2.3.8：请问对于图片中的密集文字，有什么好的处理办法吗？

**A**：可以先试用预训练模型测试一下，例如DB+CRNN，判断下密集文字图片中是检测还是识别的问题，然后针对性的改善。还有一种是如果图象中密集文字较小，可以尝试增大图像分辨率，对图像进行一定范围内的拉伸，将文字稀疏化，提高识别效果。

#### Q2.3.9：对于一些在识别时稍微模糊的文本，有没有一些图像增强的方式？

**A**：在人类肉眼可以识别的前提下，可以考虑图像处理中的均值滤波、中值滤波或者高斯滤波等模糊算子尝试。也可以尝试从数据扩增扰动来强化模型鲁棒性，另外新的思路有对抗性训练和超分SR思路，可以尝试借鉴。但目前业界尚无普遍认可的最优方案，建议优先在数据采集阶段增加一些限制提升图片质量。

#### Q2.3.10：对于特定文字检测，例如身份证只检测姓名，检测指定区域文字更好，还是检测全部区域再筛选更好？

**A**：两个角度来说明一般检测全部区域再筛选更好。

（1）由于特定文字和非特定文字之间的视觉特征并没有很强的区分行，只检测指定区域，容易造成特定文字漏检。

（2）产品的需求可能是变化的，不排除后续对于模型需求变化的可能性（比如又需要增加一个字段），相比于训练模型，后处理的逻辑会更容易调整。

#### Q2.3.11：对于小白如何快速入门中文OCR项目实践？

**A**：建议可以先了解OCR方向的基础知识，大概了解基础的检测和识别模型算法。然后在Github上可以查看OCR方向相关的repo。目前来看，从内容的完备性来看，PaddleOCR的中英文双语教程文档是有明显优势的，在数据集、模型训练、预测部署文档详实，可以快速入手。而且还有微信用户群答疑，非常适合学习实践。项目地址：[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

#### Q3.12：如何识别带空格的英文行文本图像？

**A**：空格识别可以考虑以下两种方案：

(1)优化文本检测算法。检测结果在空格处将文本断开。这种方案在检测数据标注时，需要将含有空格的文本行分成好多段。

(2)优化文本识别算法。在识别字典里面引入空格字符，然后在识别的训练数据中，如果用空行，进行标注。此外，合成数据时，通过拼接训练数据，生成含有空格的文本。

#### Q2.3.13：中英文一起识别时也可以加空格字符来训练吗

**A**：中文识别可以加空格当做分隔符训练，具体的效果如何没法给出直接评判，根据实际业务数据训练来判断。

#### Q2.3.14：低像素文字或者字号比较小的文字有什么超分辨率方法吗

**A**：超分辨率方法分为传统方法和基于深度学习的方法。基于深度学习的方法中，比较经典的有SRCNN，另外CVPR2020也有一篇超分辨率的工作可以参考文章：Unpaired Image Super-Resolution using Pseudo-Supervision，但是没有充分的实践验证过，需要看实际场景下的效果。

#### Q2.3.15：表格识别有什么好的模型 或者论文推荐么

**A**：表格目前学术界比较成熟的解决方案不多 ，可以尝试下分割的论文方案。

#### Q2.3.16：弯曲文本有试过opencv的TPS进行弯曲校正吗？

**A**：opencv的tps需要标出上下边界对应的点，这个点很难通过传统方法或者深度学习方法获取。PaddleOCR里StarNet网络中的tps模块实现了自动学点，自动校正，可以直接尝试这个。

#### Q2.3.17: StyleText 合成数据效果不好？
**A**：StyleText模型生成的数据主要用于OCR识别模型的训练。PaddleOCR目前识别模型的输入为32 x N，因此当前版本模型主要适用高度为32的数据。
建议要合成的数据尺寸设置为32 x N。尺寸相差不多的数据也可以生成，尺寸很大或很小的数据效果确实不佳。

#### Q2.3.18: 请问有哪些修改骨干网络的技巧？

**A**: 可以参考HS-ResNet这篇文章:https://arxiv.org/pdf/2010.07621.pdf


<a name="PaddleOCR实战问题"></a>
## 【实战篇】PaddleOCR实战问题

<a name="使用咨询"></a>
### 使用咨询

#### Q3.1.1：OSError： [WinError 126] 找不到指定的模块。mac pro python 3.4 shapely import 问题

**A**：这个问题是因为shapely库安装有误，可以参考 [#212](https://github.com/PaddlePaddle/PaddleOCR/issues/212) 这个issue重新安装一下

#### Q3.1.2：安装了paddle-gpu，运行时提示没有安装gpu版本的paddle，可能是什么原因?

**A**：用户同时安装了paddle cpu和gpu版本，都删掉之后，重新安装gpu版本的padle就好了

#### Q3.1.3：试用报错：Cannot load cudnn shared library，是什么原因呢？

**A**：需要把cudnn lib添加到LD_LIBRARY_PATH中去。

#### Q3.1.4：PaddlePaddle怎么指定GPU运行 os.environ["CUDA_VISIBLE_DEVICES"]这种不生效

**A**：通过设置 export CUDA_VISIBLE_DEVICES='0'环境变量

#### Q3.1.5：windows下训练没有问题，aistudio中提示数据路径有问题

**A**：需要把`\`改为`/`（windows和linux的文件夹分隔符不一样，windows下的是`\`，linux下是`/`）

#### Q3.1.6：gpu版的paddle虽然能在cpu上运行，但是必须要有gpu设备

**A**：export CUDA_VISIBLE_DEVICES=''，CPU是可以正常跑的

#### Q3.1.7：预测报错ImportError： dlopen： cannot load any more object with static TLS

**A**：glibc的版本问题，运行需要glibc的版本号大于2.23。

#### Q3.1.8：提供的inference model和预训练模型的区别

**A**：inference model为固化模型，文件中包含网络结构和网络参数，多用于预测部署。预训练模型是训练过程中保存好的模型，多用于fine-tune训练或者断点训练。

#### Q3.1.9：模型的解码部分有后处理？

**A**：有的检测的后处理在ppocr/postprocess路径下

#### Q3.1.10：PaddleOCR中文模型是否支持数字识别？

**A**：支持的，可以看下ppocr/utils/ppocr_keys_v1.txt 这个文件，是支持的识别字符列表，其中包含了数字识别。

#### Q3.1.11：PaddleOCR如何做到横排和竖排同时支持的？

**A**：合成了一批竖排文字，逆时针旋转90度后加入训练集与横排一起训练。预测时根据图片长宽比判断是否为竖排，若为竖排则将crop出的文本逆时针旋转90度后送入识别网络。

#### Q3.1.12：如何获取检测文本框的坐标？

**A**：文本检测的结果有box和文本信息, 具体 [参考代码](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/tools/infer/predict_system.py)

#### Q3.1.13：识别模型框出来的位置太紧凑，会丢失边缘的文字信息，导致识别错误

**A**： 可以在命令中加入 --det_db_unclip_ratio ，参数[定义位置](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/tools/infer/utility.py#L48)，这个参数是检测后处理时控制文本框大小的，默认1.6，可以尝试改成2.5或者更大，反之，如果觉得文本框不够紧凑，也可以把该参数调小。

#### Q3.1.14：英文手写体识别有计划提供的预训练模型吗?

**A**：近期也在开展需求调研，如果企业用户需求较多，我们会考虑增加相应的研发投入，后续提供对应的预训练模型，如果有需求欢迎通过issue或者加入微信群联系我们。

#### Q3.1.15：PaddleOCR的算法可以用于手写文字检测识别吗?后续有计划推出手写预训练模型么？
**A**：理论上只要有相应的数据集，都是可以的。当然手写识别毕竟和印刷体有区别，对应训练调优策略可能需要适配性优化。


#### Q3.1.16：PaddleOCR是否支持在Windows或Mac系统上运行？

**A**：PaddleOCR已完成Windows和Mac系统适配，运行时注意两点：

（1）在[快速安装](./installation.md)时，如果不想安装docker，可跳过第一步，直接从第二步安装paddle开始。

（2）inference模型下载时，如果没有安装wget，可直接点击模型链接或将链接地址复制到浏览器进行下载，并解压放置到相应目录。

#### Q3.1.17：PaddleOCR开源的超轻量模型和通用OCR模型的区别？
**A**：目前PaddleOCR开源了2个中文模型，分别是8.6M超轻量中文模型和通用中文OCR模型。两者对比信息如下：
- 相同点：两者使用相同的**算法**和**训练数据**；  
- 不同点：不同之处在于**骨干网络**和**通道参数**，超轻量模型使用MobileNetV3作为骨干网络，通用模型使用Resnet50_vd作为检测模型backbone，Resnet34_vd作为识别模型backbone，具体参数差异可对比两种模型训练的配置文件.

|模型|骨干网络|检测训练配置|识别训练配置|
|-|-|-|-|
|8.6M超轻量中文OCR模型|MobileNetV3+MobileNetV3|det_mv3_db.yml|rec_chinese_lite_train.yml|
|通用中文OCR模型|Resnet50_vd+Resnet34_vd|det_r50_vd_db.yml|rec_chinese_common_train.yml|

#### Q3.1.18：如何加入自己的检测算法？
**A**：1. 在ppocr/modeling对应目录下分别选择backbone，head。如果没有可用的可以新建文件并添加
       2. 在ppocr/data下选择对应的数据处理处理方式，如果没有可用的可以新建文件并添加
       3. 在ppocr/losses下新建文件并编写loss
       4. 在ppocr/postprocess下新建文件并编写后处理算法
       5. 将上面四个步骤里新添加的类或函数参照yml文件写到配置中


#### Q3.1.19：训练的时候报错`reader raised an exception`，但是具体不知道是啥问题？

**A**：这个一般是因为标注文件格式有问题或者是标注文件中的图片路径有问题导致的，在[tools/train.py](../../tools/train.py)文件中有一个`test_reader`的函数，基于这个去检查一下数据的格式以及标注，确认没问题之后再进行模型训练。

#### Q3.1.20：PaddleOCR与百度的其他OCR产品有什么区别？

**A**：PaddleOCR主要聚焦通用ocr，如果有垂类需求，您可以用PaddleOCR+垂类数据自己训练；
如果缺少带标注的数据，或者不想投入研发成本，建议直接调用开放的API，开放的API覆盖了目前比较常见的一些垂类。

#### Q3.1.21：PaddleOCR支持动态图吗？

**A**：动态图版本正在紧锣密鼓开发中，将于2020年12月16日发布，敬请关注。

#### Q3.1.22：ModuleNotFoundError: No module named 'paddle.nn'，
**A**：paddle.nn是Paddle2.0版本特有的功能，请安装大于等于Paddle 2.0.0rc1的版本，安装方式为
```
python3 -m pip install paddlepaddle-gpu==2.0.0rc1 -i https://mirror.baidu.com/pypi/simple
```

#### Q3.1.23： ImportError: /usr/lib/x86_64_linux-gnu/libstdc++.so.6:version `CXXABI_1.3.11` not found (required by /usr/lib/python3.6/site-package/paddle/fluid/core+avx.so)
**A**：这个问题是glibc版本不足导致的，Paddle2.0rc1版本对gcc版本和glib版本有更高的要求，推荐gcc版本为8.2，glibc版本2.12以上。
如果您的环境不满足这个要求，或者使用的docker镜像为:
`hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda9.0-cudnn7-dev`
`hub.baidubce.com/paddlepaddle/paddle:latest-gpu-cuda9.0-cudnn7-dev`，安装Paddle2.0rc版本可能会出现上述错误，2.0版本推荐使用新的docker镜像 `paddlepaddle/paddle:latest-dev-cuda10.1-cudnn7-gcc82`。
或者访问[dockerhub](https://hub.docker.com/r/paddlepaddle/paddle/tags/)获得与您机器适配的镜像。


#### Q3.1.24: PaddleOCR develop分支和dygraph分支的区别？
**A** 目前PaddleOCR有四个分支，分别是：

- develop：基于Paddle静态图开发的分支，推荐使用paddle1.8 或者2.0版本，该分支具备完善的模型训练、预测、推理部署、量化裁剪等功能，领先于release/1.1分支。
- release/1.1：PaddleOCR 发布的第一个稳定版本，基于静态图开发，具备完善的训练、预测、推理部署、量化裁剪等功能。
- dygraph：基于Paddle动态图开发的分支，目前仍在开发中，未来将作为主要开发分支，运行要求使用Paddle2.0rc1版本，目前仍在开发中。
- release/2.0-rc1-0：PaddleOCR发布的第二个稳定版本，基于动态图和paddle2.0版本开发，动态图开发的工程更易于调试，目前支，支持模型训练、预测，暂不支持移动端部署。

如果您已经上手过PaddleOCR，并且希望在各种环境上部署PaddleOCR，目前建议使用静态图分支，develop或者release/1.1分支。如果您是初学者，想快速训练，调试PaddleOCR中的算法，建议尝鲜PaddleOCR dygraph分支。

**注意**：develop和dygraph分支要求的Paddle版本、本地环境有差别，请注意不同分支环境安装部分的差异。

#### Q3.1.25: 使用dygraph分支，在docker中训练PaddleOCR的时候，数据路径没有任何问题，但是一直报错`reader rasied an exception`，这是为什么呢？

**A** 创建docker的时候，`/dev/shm`的默认大小为64M，如果使用多进程读取数据，共享内存可能不够，因此需要给`/dev/shm`分配更大的空间，在创建docker的时候，传入`--shm-size=8g`表示给`/dev/shm`分配8g的空间。

#### Q3.1.26: 在repo中没有找到Lite和PaddleServing相关的部署教程，这是在哪里呢？

**A** 目前PaddleOCR的默认分支为dygraph，关于Lite和PaddleLite的动态图部署还在适配中，如果希望在Lite端或者使用PaddleServing部署，推荐使用develop分支（静态图）的代码。

#### Q3.1.27: 如何可视化acc,loss曲线图,模型网络结构图等？

**A** 在配置文件里有`use_visualdl`的参数，设置为True即可，更多的使用命令可以参考：[VisualDL使用指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/guides/03_VisualDL/visualdl.html)。

#### Q3.1.28: 在使用StyleText数据合成工具的时候，报错`ModuleNotFoundError: No module named 'utils.config'`，这是为什么呢？

**A** 有2个解决方案
- 在StyleText路径下面设置PYTHONPATH：`export PYTHONPATH=./`
- 拉取最新的代码

#### Q3.1.29: PPOCRLabel创建矩形框时只能拖出正方形，如何进行矩形标注？

**A** 取消勾选：“编辑”-“正方形标注”

#### Q3.1.30: Style-Text 如何不文字风格迁移，就像普通文本生成程序一样默认字体直接输出到分割的背景图？

**A** 使用image_synth模式会输出fake_bg.jpg，即为背景图。如果想要批量提取背景，可以稍微修改一下代码，将fake_bg保存下来即可。要修改的位置：
https://github.com/PaddlePaddle/PaddleOCR/blob/de3e2e7cd3b8b65ee02d7a41e570fa5b511a3c1d/StyleText/engine/synthesisers.py#L68

#### Q3.1.31: 怎么输出网络结构以及每层的参数信息？

**A** 可以使用 `paddle.summary`， 具体参考:https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0-rc1/api/paddle/hapi/model_summary/summary_cn.html#summary。

<a name="数据集3"></a>
### 数据集

#### Q3.2.1：如何制作PaddleOCR支持的数据格式

**A**：可以参考检测与识别训练文档，里面有数据格式详细介绍。[检测文档](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/detection.md)，[识别文档](https：//github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/recognition.md)

#### Q3.2.2：请问一下，如果想用预训练模型，但是我的数据里面又出现了预训练模型字符集中没有的字符，新的字符是在字符集前面添加还是在后面添加？

**A**：在后面添加，修改dict之后，就改变了模型最后一层fc的结构，之前训练到的参数没有用到，相当于从头训练，因此acc是0。

#### Q3.2.3：如何调试数据读取程序？

**A**：tools/train.py中有一个test_reader()函数用于调试数据读取。

#### Q3.2.4：开源模型使用的训练数据是什么，能否开源？

**A**：目前开源的模型，数据集和量级如下：

- 检测：  
    - 英文数据集，ICDAR2015  
    - 中文数据集，LSVT街景数据集训练数据3w张图片

- 识别：  
    - 英文数据集，MJSynth和SynthText合成数据，数据量上千万。  
    - 中文数据集，LSVT街景数据集根据真值将图crop出来，并进行位置校准，总共30w张图像。此外基于LSVT的语料，合成数据500w。  

其中，公开数据集都是开源的，用户可自行搜索下载，也可参考[中文数据集](./datasets.md)，合成数据暂不开源，用户可使用开源合成工具自行合成，可参考的合成工具包括[text_renderer](https://github.com/Sanster/text_renderer)、[SynthText](https://github.com/ankush-me/SynthText)、[TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)等。

#### Q3.2.5：请问中文字符集多大呢？支持生僻字识别吗？

**A**：中文字符集是6623， 支持生僻字识别。训练样本中有部分生僻字，但样本不多，如果有特殊需求建议使用自己的数据集做fine-tune。

#### Q3.2.6：中文文本检测、文本识别构建训练集的话，大概需要多少数据量

**A**：检测需要的数据相对较少，在PaddleOCR模型的基础上进行Fine-tune，一般需要500张可达到不错的效果。
识别分英文和中文，一般英文场景需要几十万数据可达到不错的效果，中文则需要几百万甚至更多。

#### Q3.2.7：中文识别模型如何选择？

**A**：中文模型共有2大类：通用模型和超轻量模型。他们各自的优势如下：
超轻量模型具有更小的模型大小，更快的预测速度。适合用于端侧使用。
通用模型具有更高的模型精度，适合对模型大小不敏感的场景。
此外基于以上模型，PaddleOCR还提供了支持空格识别的模型，主要针对中文场景中的英文句子。
您可以根据实际使用需求进行选择。

#### Q3.2.8：图像旋转90° 文本检测可以正常检测到具体文本位置，但是识别准确度大幅降低，是否会考虑增加相应的旋转预处理？

**A**：目前模型只支持两种方向的文字：水平和垂直。 为了降低模型大小，加快模型预测速度，PaddleOCR暂时没有加入图片的方向判断。建议用户在识别前自行转正，后期也会考虑添加选择角度判断。

#### Q3.2.9：同一张图通用检测出21个条目，轻量级检测出26个 ，难道不是轻量级的好吗？

**A**：可以主要参考可视化效果，通用模型更倾向于检测一整行文字，轻量级可能会有一行文字被分成两段检测的情况，不是数量越多，效果就越好。


#### Q3.2.10：crnn+ctc模型训练所用的垂直文本（旋转至水平方向）是如何生成的？

**A**：方法与合成水平方向文字一致，只是将字体替换成了垂直字体。

#### Q3.2.11：有哪些标注工具可以标注OCR数据集？

**A**：您可以参考：https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/data_annotation_en.md。
我们计划推出高效标注OCR数据的标注工具，请您持续关注PaddleOCR的近期更新。

#### Q3.2.12：一些特殊场景的数据识别效果差，但是数据量很少，不够用来finetune怎么办？

**A**：您可以合成一些接近使用场景的数据用于训练。
我们计划推出基于特定场景的文本数据合成工具，请您持续关注PaddleOCR的近期更新。

#### Q3.2.13：特殊字符（例如一些标点符号）识别效果不好怎么办？

**A**：首先请您确认要识别的特殊字符是否在字典中。
如果字符在已经字典中但效果依然不好，可能是由于识别数据较少导致的，您可以增加相应数据finetune模型。

#### Q3.2.14：PaddleOCR可以识别灰度图吗？

**A**：PaddleOCR的模型均为三通道输入。如果您想使用灰度图作为输入，建议直接用3通道的模式读入灰度图，
或者将单通道图像转换为三通道图像再识别。例如，opencv的cvtColor函数就可以将灰度图转换为RGB三通道模式。

#### Q3.2.15: 文本标注工具PPOCRLabel有什么特色？

**A**: PPOCRLabel是一个半自动文本标注工具，它使用基于PPOCR的中英文OCR模型，预先预测文本检测和识别结果，然后用户对上述结果进行校验和修正就行，大大提高用户的标注效率。同时导出的标注结果直接适配PPOCR训练所需要的数据格式，

#### Q3.2.16: 文本标注工具PPOCRLabel，可以更换模型吗？

**A**: PPOCRLabel中OCR部署方式采用的基于pip安装whl包快速推理，可以参考相关文档更换模型路径，进行特定任务的标注适配。基于pip安装whl包快速推理的文档如下，https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/whl.md。

#### Q3.2.17: 文本标注工具PPOCRLabel支持的运行环境有哪些？

**A**: PPOCRLabel可运行于Linux、Windows、MacOS等多种系统。操作步骤可以参考文档，https://github.com/PaddlePaddle/PaddleOCR/blob/develop/PPOCRLabel/README.md

<a name="模型训练调优3"></a>

### 模型训练调优

#### Q3.3.1：文本长度超过25，应该怎么处理？

**A**：默认训练时的文本可识别的最大长度为25，超过25的文本会被忽略不参与训练。如果您训练样本中的长文本较多，可以修改配置文件中的 max\_text\_length 字段，设置为更大的最长文本长度，具体位置在[这里](https://github.com/PaddlePaddle/PaddleOCR/blob/fb9e47b262529386983edc21b33abfa16bbf06ac/configs/rec/rec_chinese_lite_train.yml#L13)。

#### Q3.3.2：配置文件里面检测的阈值设置么?

**A**：有的，检测相关的参数主要有以下几个：
``det_limit_side_len：预测时图像resize的长边尺寸
det_db_thresh: 用于二值化输出图的阈值
det_db_box_thresh:用于过滤文本框的阈值，低于此阈值的文本框不要
det_db_unclip_ratio: 文本框扩张的系数，关系到文本框的大小``

这些参数的默认值见[代码](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/tools/infer/utility.py#L42)，可以通过从命令行传递参数进行修改。

#### Q3.3.3：我想请教一下，你们在训练识别时候，lsvt里的非矩形框文字，你们是怎么做处理的呢。忽略掉还是去最小旋转框？

**A**：现在是忽略处理的

#### Q3.3.4：训练过程中，如何恰当的停止训练（直接kill，经常还有显存占用的问题）

**A**：可以通过下面的脚本终止所有包含train.py字段的进程，

```shell
ps -axu | grep train.py | awk '{print $2}' | xargs kill -9
```


#### Q3.3.5：可不可以将pretrain_weights设置为空呢？想从零开始训练一个model

**A**：这个是可以的，在训练通用识别模型的时候，pretrain_weights就设置为空，但是这样可能需要更长的迭代轮数才能达到相同的精度。

#### Q3.3.6：PaddleOCR默认不是200个step保存一次模型吗？为啥文件夹下面都没有生成

**A**：因为默认保存的起始点不是0，而是4000，将eval_batch_step [4000, 5000]改为[0, 2000] 就是从第0次迭代开始，每2000迭代保存一次模型

#### Q3.3.7：如何进行模型微调？

**A**：注意配置好合适的数据集，对齐数据格式，然后在finetune训练时，可以加载我们提供的预训练模型，设置配置文件中Global.pretrain_weights 参数为要加载的预训练模型路径。

#### Q3.3.8：文本检测换成自己的数据没法训练，有一些”###”是什么意思？

**A**：数据格式有问题，”###” 表示要被忽略的文本区域，所以你的数据都被跳过了，可以换成其他任意字符或者就写个空的。

#### Q3.3.9：copy_from_cpu这个地方，这块input不变(t_data的size不变)连续调用两次copy_from_cpu()时，这里面的gpu_place会重新malloc GPU内存吗？还是只有当ele_size变化时才会重新在GPU上malloc呢？

**A**：小于等于的时候都不会重新分配，只有大于的时候才会重新分配

#### Q3.3.10：自己训练出来的未inference转换的模型 可以当作预训练模型吗？

**A**：可以的，但是如果训练数据量少的话，可能会过拟合到少量数据上，泛化性能不佳。

#### Q3.3.11：使用带TPS的识别模型预测报错

**A**：TPS模块暂时不支持导出，后续更新。

#### Q3.3.12：如何更换文本检测/识别的backbone？报错信息：``Input(X) dims[3] and Input(Grid) dims[2] should be equal, but received X dimension[3](320) != Grid dimension[2](100)  ``

**A**：直接更换配置文件里的Backbone.name即可，格式为：网络文件路径,网络Class名词。如果所需的backbone在PaddleOCR里没有提供，可以参照PaddleClas里面的网络结构，进行修改尝试。具体修改原则可以参考OCR通用问题中 "如何更换文本检测/识别的backbone" 的回答。

#### Q3.3.13： 训练中使用的字典需要与加载的预训练模型使用的字典一样吗？

**A**：分情况，1. 不改变识别字符，训练的字典与你使用该模型进行预测的字典需要保持一致的。
             2. 改变识别的字符，这种情况可以不一样，最后一层会重新训练

#### Q3.3.14: 如何对检测模型finetune，比如冻结前面的层或某些层使用小的学习率学习？

**A**：

**A**：如果是冻结某些层，可以将变量的stop_gradient属性设置为True，这样计算这个变量之前的所有参数都不会更新了，参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/faq/train_cn.html#id4

如果对某些层使用更小的学习率学习，静态图里还不是很方便，一个方法是在参数初始化的时候，给权重的属性设置固定的学习率，参考：https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/fluid/param_attr/ParamAttr_cn.html#paramattr

实际上我们实验发现，直接加载模型去fine-tune，不设置某些层不同学习率，效果也都不错

#### Q3.3.15: 使用通用中文模型作为预训练模型，更改了字典文件，出现ctc_fc_b not used的错误

**A**：修改了字典之后，识别模型的最后一层FC纬度发生了改变，没有办法加载参数。这里是一个警告，可以忽略，正常训练即可。

#### Q3.3.16:  cpp_infer 在Windows下使用vs2015编译不通过

**A**：1. windows上建议使用VS2019工具编译，具体编译细节参考[链接](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/cpp_infer/docs/windows_vs2019_build.md)

**A**：2. 在release模式下而不是debug模式下编译，参考[issue](https://github.com/PaddlePaddle/PaddleOCR/issues/1023)

#### Q3.3.17:  No module named 'tools.infer'

**A**：1. 确保在PaddleOCR/目录下执行的指令，执行'export PYTHONPATH=.'

**A**：2. 拉取github上最新代码，这个问题在10月底已修复。

#### Q3.3.18:  训练模型和测试模型的检测结果差距较大

**A**：1. 检查两个模型使用的后处理参数是否是一样的，训练的后处理参数在配置文件中的PostProcess部分，测试模型的后处理参数在tools/infer/utility.py中，最新代码中两个后处理参数已保持一致。


#### Q3.3.19: 使用合成数据精调小模型后，效果可以，但是还没开源的小infer模型效果好，这是为什么呢？

**A**：

（1）要保证使用的配置文件和pretrain weights是对应的；

（2）在微调时，一般都需要真实数据，如果使用合成数据，效果反而可能会有下降，PaddleOCR中放出的识别inference模型也是基于预训练模型在真实数据上微调得到的，效果提升比较明显；

（3）在训练的时候，文本长度超过25的训练图像都会被丢弃，因此需要看下真正参与训练的图像有多少，太少的话也容易过拟合。

#### Q3.3.20: 文字检测时怎么模糊的数据增强？

**A**: 模糊的数据增强需要修改代码进行添加，以DB为例，参考[Normalize](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppocr/data/imaug/operators.py#L60) ,添加模糊的增强就行

#### Q3.3.21: 文字检测时怎么更改图片旋转的角度，实现360度任意旋转？

**A**: 将[这里](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppocr/data/imaug/iaa_augment.py#L64) 的(-10,10) 改为(-180,180)即可

#### Q3.3.22: 训练数据的长宽比过大怎么修改shape

**A**: 识别修改[这里](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yaml#L75) ,
检测修改[这里](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml#L85)

#### Q3.3.23：检测模型训练或预测时出现elementwise_add报错

**A**：设置的输入尺寸必须是32的倍数，否则在网络多次下采样和上采样后，feature map会产生1个像素的diff，从而导致elementwise_add时报shape不匹配的错误。

#### Q3.3.24: DB检测训练输入尺寸640，可以改大一些吗？

**A**: 不建议改大。检测模型训练输入尺寸是预处理中random crop后的尺寸，并非直接将原图进行resize，多数场景下这个尺寸并不小了，改大后可能反而并不合适，而且训练会变慢。另外，代码里可能有的地方参数按照预设输入尺寸适配的，改大后可能有隐藏风险。

#### Q3.3.25: 识别模型训练时，loss能正常下降，但acc一直为0

**A**: 识别模型训练初期acc为0是正常的，多训一段时间指标就上来了。

#### Q3.3.26: PaddleOCR在训练的时候一直使用cosine_decay的学习率下降策略，这是为什么呢？

**A**: cosine_decay表示在训练的过程中，学习率按照cosine的变化趋势逐渐下降至0，在迭代轮数更长的情况下，比常量的学习率变化策略会有更好的收敛效果，因此在实际训练的时候，均采用了cosine_decay，来获得精度更高的模型。

<a name="预测部署3"></a>

### 预测部署

#### Q3.4.1：如何pip安装opt模型转换工具？

**A**：由于OCR端侧部署需要某些算子的支持，这些算子仅在Paddle-Lite 最新develop分支中，所以需要自己编译opt模型转换工具。opt工具可以通过编译PaddleLite获得，编译步骤参考[lite部署文档](https://github.com/PaddlePaddle/PaddleOCR/blob/0791714b91/deploy/lite/readme.md)  中2.1 模型优化部分。

#### Q3.4.2：如何将PaddleOCR预测模型封装成SDK

**A**：如果是Python的话，可以使用tools/infer/predict_system.py中的TextSystem进行sdk封装，如果是c++的话，可以使用deploy/cpp_infer/src下面的DBDetector和CRNNRecognizer完成封装

#### Q3.4.3：服务部署可以只发布文本识别，而不带文本检测模型么？

**A**：可以的。默认的服务部署是检测和识别串联预测的。也支持单独发布文本检测或文本识别模型，比如使用PaddleHUBPaddleOCR 模型时，deploy下有三个文件夹，分别是

- ocr_det：检测预测
- ocr_rec: 识别预测
- ocr_system: 检测识别串联预测

每个模块是单独分开的，所以可以选择只发布文本识别模型。使用PaddleServing部署时同理。


#### Q3.4.4：为什么PaddleOCR检测预测是只支持一张图片测试？即test_batch_size_per_card=1

**A**：测试的时候，对图像等比例缩放，最长边960，不同图像等比例缩放后长宽不一致，无法组成batch，所以设置为test_batch_size为1。

#### Q3.4.5：为什么使用c++ inference和python inference结果不一致?

**A**：可能是导出的inference model版本与预测库版本需要保持一致，比如在Windows下，Paddle官网提供的预测库版本是1.8，而PaddleOCR提供的inference model 版本是1.7，因此最终预测结果会有差别。可以在Paddle1.8环境下导出模型，再基于该模型进行预测。
此外也需要保证两者的预测参数配置完全一致。

#### Q3.4.6：为什么第一张张图预测时间很长，第二张之后预测时间会降低？

**A**：第一张图需要显存资源初始化，耗时较多。完成模型加载后，之后的预测时间会明显缩短。

#### Q3.4.7：请问opt工具可以直接转int8量化后的模型为.nb文件吗

**A**：有的，PaddleLite提供完善的opt工具，可以参考[文档](https://paddle-lite.readthedocs.io/zh/latest/user_guides/post_quant_with_data.html)

#### Q3.4.8：请问在安卓端怎么设置这个参数 --det_db_unclip_ratio=3

**A**：在安卓APK上无法设置，没有暴露这个接口，如果使用的是PaddledOCR/deploy/lite/的demo，可以修改config.txt中的对应参数来设置

#### Q3.4.9：PaddleOCR模型是否可以转换成ONNX模型?

**A**：目前暂不支持转ONNX，相关工作在研发中。

#### Q3.4.10：使用opt工具对检测模型转换时报错 can not found op arguments for node conv2_b_attr

**A**：这个问题大概率是编译opt工具的Paddle-Lite不是develop分支，建议使用Paddle-Lite 的develop分支编译opt工具。

#### Q3.4.11：libopenblas.so找不到是什么意思？

**A**：目前包括mkl和openblas两种版本的预测库，推荐使用mkl的预测库，如果下载的预测库是mkl的，编译的时候也需要勾选`with_mkl`选项
，以Linux下编译为例，需要在设置这里为ON，`-DWITH_MKL=ON`，[参考链接](https://github.com/PaddlePaddle/PaddleOCR/blob/569deedc41c2fa5e126a4d14b6c0c46a6bca43b8/deploy/cpp_infer/tools/build.sh#L12) 。此外，使用预测库时，推荐在Linux或者Windows上进行开发，不推荐在MacOS上开发。

#### Q3.4.12：使用自定义字典训练，inference时如何修改

**A**：使用了自定义字典的话，用inference预测时，需要通过 --rec_char_dict_path 修改字典路径。详细操作可参考[文档](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/inference.md#4-%E8%87%AA%E5%AE%9A%E4%B9%89%E6%96%87%E6%9C%AC%E8%AF%86%E5%88%AB%E5%AD%97%E5%85%B8%E7%9A%84%E6%8E%A8%E7%90%86)

#### Q3.4.13：能否返回单字字符的位置？

**A**：训练的时候标注是整个文本行的标注，所以预测的也是文本行位置，如果要获取单字符位置信息，可以根据预测的文本，计算字符数量，再去根据整个文本行的位置信息，估计文本块中每个字符的位置。

#### Q3.4.14：PaddleOCR模型部署方式有哪几种？

**A**：目前有Inference部署，serving部署和手机端Paddle Lite部署，可根据不同场景做灵活的选择：Inference部署适用于本地离线部署，serving部署适用于云端部署，Paddle Lite部署适用于手机端集成。

#### Q3.4.15： hubserving、pdserving这两种部署方式区别是什么？

**A**：hubserving原本是paddlehub的配套服务部署工具，可以很方便的将paddlehub内置的模型部署为服务，paddleocr使用了这个功能，并将模型路径等参数暴露出来方便用户自定义修改。paddle serving是面向所有paddle模型的部署工具，文档中可以看到我们提供了快速版和标准版，其中快速版和hubserving的本质是一样的，而标准版基于rpc，更稳定，更适合分布式部署。

#### Q3.4.16： hub serving部署服务时如何多gpu同时利用起来，export CUDA_VISIBLE_DEVICES=0,1 方式吗？

**A**：hubserving的部署方式目前暂不支持多卡预测，除非手动启动多个serving，不同端口对应不同卡。或者可以使用paddleserving进行部署，部署工具已经发布：https://github.com/PaddlePaddle/PaddleOCR/tree/develop/deploy/pdserving ，在启动服务时--gpu_id 0,1 这样就可以

#### Q3.4.17： 预测内存泄漏问题

**A**：1. 使用hubserving出现内存泄漏，该问题为已知问题，预计在paddle2.0正式版中解决。相关讨论见[issue](https://github.com/PaddlePaddle/PaddleHub/issues/682)

**A**：2. C++ 预测出现内存泄漏，该问题已经在paddle2.0rc版本中解决，建议安装paddle2.0rc版本，并更新PaddleOCR代码到最新。

#### Q3.4.18：对于一些尺寸较大的文档类图片，在检测时会有较多的漏检，怎么避免这种漏检的问题呢？

**A**：PaddleOCR中在图像最长边大于960时，将图像等比例缩放为长边960的图像再进行预测，对于这种图像，可以通过修改det_limit_side_len，增大检测的最长边：[tools/infer/utility.py#L42](../../tools/infer/utility.py#L42)

#### Q3.4.19：在使用训练好的识别模型进行预测的时候，发现有很多重复的字，这个怎么解决呢？

**A**：可以看下训练的尺度和预测的尺度是否相同，如果训练的尺度为`[3, 32, 320]`，预测的尺度为`[3, 64, 640]`，则会有比较多的重复识别现象。

#### Q3.4.20：文档场景中，使用DB模型会出现整行漏检的情况应该怎么解决？

**A**：可以在预测时调小 det_db_box_thresh 阈值，默认为0.5, 可调小至0.3观察效果。

#### Q3.4.21：自己训练的det模型，在同一张图片上，inference模型与eval模型结果差别很大，为什么？

**A**：这是由于图片预处理不同造成的。如果训练的det模型图片输入并不是默认的shape[600, 600]，eval的程序中图片预处理方式与train时一致
（由xxx_reader.yml中的test_image_shape参数决定缩放大小，但predict_eval.py中的图片预处理方式由程序里的preprocess_params决定，
最好不要传入max_side_len，而是传入和训练时一样大小的test_image_shape。

#### Q3.4.22：训练ccpd车牌数据集，训练集准确率高，测试均是错误的，这是什么原因？

**A**：这是因为训练时将shape修改为[3, 70, 220], 预测时对图片resize，会把高度压缩至32，影响测试结果。注释掉[resize代码](https://github.com/PaddlePaddle/PaddleOCR/blob/569deedc41c2fa5e126a4d14b6c0c46a6bca43b8/tools/infer/predict_rec.py#L56-L57) 可以解决问题。

#### Q3.4.23：安装paddleocr后，提示没有paddle

**A**：这是因为paddlepaddle gpu版本和cpu版本的名称不一致，现在已经在[whl的文档](./whl.md)里做了安装说明。

#### Q3.4.24：DB模型能正确推理预测，但换成EAST或SAST模型时报错或结果不正确

**A**：使用EAST或SAST模型进行推理预测时，需要在命令中指定参数--det_algorithm="EAST" 或 --det_algorithm="SAST"，使用DB时不用指定是因为该参数默认值是"DB"：https://github.com/PaddlePaddle/PaddleOCR/blob/e7a708e9fdaf413ed7a14da8e4a7b4ac0b211e42/tools/infer/utility.py#L43

#### Q3.4.25 : PaddleOCR模型Python端预测和C++预测结果不一致？
正常来说，python端预测和C++预测文本是一致的，如果预测结果差异较大，
建议首先排查diff出现在检测模型还是识别模型，或者尝试换其他模型是否有类似的问题。
其次，检查python端和C++端数据处理部分是否存在差异，建议保存环境，更新PaddleOCR代码再试下。
如果更新代码或者更新代码都没能解决，建议在PaddleOCR微信群里或者issue中抛出您的问题。

### Q3.4.26: 目前paddle hub serving 只支持 imgpath，如果我想用imgurl 去哪里改呢？

**A**: 图片是在这里读取的：https://github.com/PaddlePaddle/PaddleOCR/blob/67ef25d593c4eabfaaceb22daade4577f53bed81/deploy/hubserving/ocr_system/module.py#L55，
可以参考下面的写法，将url path转化为np array（https://cloud.tencent.com/developer/article/1467840）
```
response = request.urlopen('http://i1.whymtj.com/uploads/tu/201902/9999/52491ae4ba.jpg')
img_array = np.array(bytearray(response.read()), dtype=np.uint8)
img = cv.imdecode(img_array, -1)
```
