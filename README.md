[English](README_en.md) | 简体中文

## 简介
PaddleOCR旨在打造一套丰富、领先、且实用的OCR工具库，助力使用者训练出更好的模型，并应用落地。

**近期更新**
- 2020.8.26 更新OCR相关的84个常见问题及解答，具体参考[FAQ](./doc/doc_ch/FAQ.md)
- 2020.8.24 支持通过whl包安装使用PaddleOCR，具体参考[Paddleocr Package使用说明](./doc/doc_ch/whl.md)
- 2020.8.21 更新8月18日B站直播课回放和PPT，课节2，易学易用的OCR工具大礼包，[获取地址](https://aistudio.baidu.com/aistudio/education/group/info/1519)
- 2020.8.16 开源文本检测算法[SAST](https://arxiv.org/abs/1908.05498)和文本识别算法[SRN](https://arxiv.org/abs/2003.12294)
- 2020.7.23 发布7月21日B站直播课回放和PPT，课节1，PaddleOCR开源大礼包全面解读，[获取地址](https://aistudio.baidu.com/aistudio/course/introduce/1519)
- 2020.7.15 添加基于EasyEdge和Paddle-Lite的移动端DEMO，支持iOS和Android系统
- [more](./doc/doc_ch/update.md)


## 特性
- 超轻量级中文OCR模型，总模型仅8.6M
    - 单模型支持中英文数字组合识别、竖排文本识别、长文本识别
    - 检测模型DB（4.1M）+识别模型CRNN（4.5M）
- 实用通用中文OCR模型
- 多种预测推理部署方案，包括服务部署和端侧部署
- 多种文本检测训练算法，EAST、DB、SAST
- 多种文本识别训练算法，Rosetta、CRNN、STAR-Net、RARE、SRN
- 可运行于Linux、Windows、MacOS等多种系统

## 快速体验

<div align="center">
    <img src="doc/imgs_results/11.jpg" width="800">
</div>

上图是超轻量级中文OCR模型效果展示，更多效果图请见[效果展示页面](./doc/doc_ch/visualization.md)。

- 超轻量级中文OCR在线体验地址：https://www.paddlepaddle.org.cn/hub/scene/ocr
- 移动端DEMO体验(基于EasyEdge和Paddle-Lite, 支持iOS和Android系统)：[安装包二维码获取地址](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)

   Android手机也可以扫描下面二维码安装体验。

<div align="center">
<img src="./doc/ocr-android-easyedge.png"  width = "200" height = "200" />
</div>


## 中文OCR模型列表

|模型名称|模型简介|检测模型地址|识别模型地址|支持空格的识别模型地址|
|-|-|-|-|-|
|chinese_db_crnn_mobile|超轻量级中文OCR模型|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)
|chinese_db_crnn_server|通用中文OCR模型|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)

## 文档教程
- [快速安装](./doc/doc_ch/installation.md)
- [中文OCR模型快速使用](./doc/doc_ch/quickstart.md)
- 算法介绍
    - [文本检测](#文本检测算法)
    - [文本识别](#文本识别算法)
- 模型训练/评估
    - [文本检测](./doc/doc_ch/detection.md)
    - [文本识别](./doc/doc_ch/recognition.md)
    - [yml参数配置文件介绍](./doc/doc_ch/config.md)
    - [中文OCR训练预测技巧](./doc/doc_ch/tricks.md)
- 预测部署
    - [基于Python预测引擎推理](./doc/doc_ch/inference.md)
    - [基于C++预测引擎推理](./deploy/cpp_infer/readme.md)
    - [服务化部署](./doc/doc_ch/serving.md)
    - [端侧部署](./deploy/lite/readme.md)
    - 模型量化压缩（coming soon）
    - [Benchmark](./doc/doc_ch/benchmark.md)
- 数据集
    - [通用中英文OCR数据集](./doc/doc_ch/datasets.md)
    - [手写中文OCR数据集](./doc/doc_ch/handwritten_datasets.md)
    - [垂类多语言OCR数据集](./doc/doc_ch/vertical_and_multilingual_datasets.md)
    - [常用数据标注工具](./doc/doc_ch/data_annotation.md)
    - [常用数据合成工具](./doc/doc_ch/data_synthesis.md)
- 效果展示
    - [超轻量级中文OCR效果展示](#超轻量级中文OCR效果展示)
    - [通用中文OCR效果展示](#通用中文OCR效果展示)
    - [支持空格的中文OCR效果展示](#支持空格的中文OCR效果展示)
- FAQ
    - [【精选】OCR精选10个问题](./doc/doc_ch/FAQ.md)
    - [【理论篇】OCR通用21个问题](./doc/doc_ch/FAQ.md)
    - [【实战篇】PaddleOCR实战53个问题](./doc/doc_ch/FAQ.md)
- [技术交流群](#欢迎加入PaddleOCR技术交流群)
- [参考文献](./doc/doc_ch/reference.md)
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)

<a name="算法介绍"></a>
## 算法介绍
<a name="文本检测算法"></a>
### 1.文本检测算法

PaddleOCR开源的文本检测算法列表：
- [x]  EAST([paper](https://arxiv.org/abs/1704.03155))
- [x]  DB([paper](https://arxiv.org/abs/1911.08947))
- [x]  SAST([paper](https://arxiv.org/abs/1908.05498))(百度自研)

在ICDAR2015文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
|-|-|-|-|-|-|
|EAST|ResNet50_vd|88.18%|85.51%|86.82%|[下载链接](https://paddleocr.bj.bcebos.com/det_r50_vd_east.tar)|
|EAST|MobileNetV3|81.67%|79.83%|80.74%|[下载链接](https://paddleocr.bj.bcebos.com/det_mv3_east.tar)|
|DB|ResNet50_vd|83.79%|80.65%|82.19%|[下载链接](https://paddleocr.bj.bcebos.com/det_r50_vd_db.tar)|
|DB|MobileNetV3|75.92%|73.18%|74.53%|[下载链接](https://paddleocr.bj.bcebos.com/det_mv3_db.tar)|
|SAST|ResNet50_vd|92.18%|82.96%|87.33%|[下载链接](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_icdar2015.tar)|

在Total-text文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
|-|-|-|-|-|-|
|SAST|ResNet50_vd|88.74%|79.80%|84.03%|[下载链接](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_total_text.tar)|

**说明：** SAST模型训练额外加入了icdar2013、icdar2017、COCO-Text、ArT等公开数据集进行调优。PaddleOCR用到的经过整理格式的英文公开数据集下载：[百度云地址](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (提取码: 2bpi)


使用[LSVT](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/datasets.md#1icdar2019-lsvt)街景数据集共3w张数据，训练中文检测模型的相关配置和预训练文件如下：

|模型|骨干网络|配置文件|预训练模型|
|-|-|-|-|
|超轻量中文模型|MobileNetV3|det_mv3_db.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|
|通用中文OCR模型|ResNet50_vd|det_r50_vd_db.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|

* 注： 上述DB模型的训练和评估，需设置后处理参数box_thresh=0.6，unclip_ratio=1.5，使用不同数据集、不同模型训练，可调整这两个参数进行优化

PaddleOCR文本检测算法的训练和使用请参考文档教程中[模型训练/评估中的文本检测部分](./doc/doc_ch/detection.md)。

<a name="文本识别算法"></a>
### 2.文本识别算法

PaddleOCR开源的文本识别算法列表：
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))
- [x]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))
- [x]  RARE([paper](https://arxiv.org/abs/1603.03915v1))
- [x]  SRN([paper](https://arxiv.org/abs/2003.12294))(百度自研)

参考[DTRB](https://arxiv.org/abs/1904.01906)文字识别训练和评估流程，使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法效果如下：

|模型|骨干网络|Avg Accuracy|模型存储命名|下载链接|
|-|-|-|-|-|
|Rosetta|Resnet34_vd|80.24%|rec_r34_vd_none_none_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_none_ctc.tar)|
|Rosetta|MobileNetV3|78.16%|rec_mv3_none_none_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_none_none_ctc.tar)|
|CRNN|Resnet34_vd|82.20%|rec_r34_vd_none_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_bilstm_ctc.tar)|
|CRNN|MobileNetV3|79.37%|rec_mv3_none_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_none_bilstm_ctc.tar)|
|STAR-Net|Resnet34_vd|83.93%|rec_r34_vd_tps_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_ctc.tar)|
|STAR-Net|MobileNetV3|81.56%|rec_mv3_tps_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_ctc.tar)|
|RARE|Resnet34_vd|84.90%|rec_r34_vd_tps_bilstm_attn|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_attn.tar)|
|RARE|MobileNetV3|83.32%|rec_mv3_tps_bilstm_attn|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_attn.tar)|
|SRN|Resnet50_vd_fpn|88.33%|rec_r50fpn_vd_none_srn|[下载链接](https://paddleocr.bj.bcebos.com/SRN/rec_r50fpn_vd_none_srn.tar)|

**说明：** SRN模型使用了数据扰动方法对上述提到对两个训练集进行增广，增广后的数据可以在[百度网盘](https://pan.baidu.com/s/1-HSZ-ZVdqBF2HaBZ5pRAKA)上下载，提取码: y3ry。
原始论文使用两阶段训练平均精度为89.74%，PaddleOCR中使用one-stage训练，平均精度为88.33%。两种预训练权重均在[下载链接](https://paddleocr.bj.bcebos.com/SRN/rec_r50fpn_vd_none_srn.tar)中。

使用[LSVT](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/datasets.md#1icdar2019-lsvt)街景数据集根据真值将图crop出来30w数据，进行位置校准。此外基于LSVT语料生成500w合成数据训练中文模型，相关配置和预训练文件如下：  

|模型|骨干网络|配置文件|预训练模型|
|-|-|-|-|
|超轻量中文模型|MobileNetV3|rec_chinese_lite_train.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|
|通用中文OCR模型|Resnet34_vd|rec_chinese_common_train.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|

PaddleOCR文本识别算法的训练和使用请参考文档教程中[模型训练/评估中的文本识别部分](./doc/doc_ch/recognition.md)。

## 效果展示

<a name="超轻量级中文OCR效果展示"></a>
### 1.超轻量级中文OCR效果展示  [more](./doc/doc_ch/visualization.md)

<div align="center">
    <img src="doc/imgs_results/1.jpg" width="800">
</div>

<a name="通用中文OCR效果展示"></a>
### 2.通用中文OCR效果展示  [more](./doc/doc_ch/visualization.md)

<div align="center">
    <img src="doc/imgs_results/chinese_db_crnn_server/11.jpg" width="800">
</div>

<a name="支持空格的中文OCR效果展示"></a>
### 3.支持空格的中文OCR效果展示  [more](./doc/doc_ch/visualization.md)

<div align="center">
    <img src="doc/imgs_results/chinese_db_crnn_server/en_paper.jpg" width="800">
</div>

<a name="欢迎加入PaddleOCR技术交流群"></a>
## 欢迎加入PaddleOCR技术交流群
请扫描下面二维码，完成问卷填写，获取加群二维码和OCR方向的炼丹秘籍

<div align="center">
<img src="./doc/joinus.PNG"  width = "200" height = "200" />
</div>

<a name="许可证书"></a>
## 许可证书
本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>许可认证。

<a name="贡献代码"></a>
## 贡献代码
我们非常欢迎你为PaddleOCR贡献代码，也十分感谢你的反馈。

- 非常感谢 [Khanh Tran](https://github.com/xxxpsyduck) 和 [Karl Horky](https://github.com/karlhorky) 贡献修改英文文档
- 非常感谢 [zhangxin](https://github.com/ZhangXinNan)([Blog](https://blog.csdn.net/sdlypyzq)) 贡献新的可视化方式、添加.gitgnore、处理手动设置PYTHONPATH环境变量的问题
- 非常感谢 [lyl120117](https://github.com/lyl120117) 贡献打印网络结构的代码
- 非常感谢 [xiangyubo](https://github.com/xiangyubo) 贡献手写中文OCR数据集
- 非常感谢 [authorfu](https://github.com/authorfu) 贡献Android和[xiadeye](https://github.com/xiadeye) 贡献IOS的demo代码
- 非常感谢 [BeyondYourself](https://github.com/BeyondYourself) 给PaddleOCR提了很多非常棒的建议，并简化了PaddleOCR的部分代码风格。
- 非常感谢 [tangmq](https://gitee.com/tangmq) 给PaddleOCR增加Docker化部署服务，支持快速发布可调用的Restful API服务。
