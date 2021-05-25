[English](README.md) | 简体中文

## 简介
PaddleOCR旨在打造一套丰富、领先、且实用的OCR工具库，助力使用者训练出更好的模型，并应用落地。
## 注意
PaddleOCR同时支持动态图与静态图两种编程范式
- 动态图版本：release/2.1（默认分支，开发分支为dygraph分支），需将paddle版本升级至2.0.0或以上版本（[快速安装](./doc/doc_ch/installation.md)）
- 静态图版本：develop分支

**近期更新**
- 2021.5.24 [FAQ](./doc/doc_ch/FAQ.md)新增5个高频问题，总数228个，每周一都会更新，欢迎大家持续关注。
- PaddleOCR研发团队对最新发版内容技术深入解读，4月13日晚上19:00，[直播地址](https://live.bilibili.com/21689802)。
- 2021.4.8 release 2.1版本，新增AAAI 2021论文[端到端识别算法PGNet](./doc/doc_ch/pgnet.md)开源，[多语言模型](./doc/doc_ch/multi_languages.md)支持种类增加到80+。
- 2021.2.8 正式发布PaddleOCRv2.0(branch release/2.0)并设置为推荐用户使用的默认分支. 发布的详细内容，请参考: https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v2.0.0
- 2021.1.26,28,29 PaddleOCR官方研发团队带来技术深入解读三日直播课，1月26日、28日、29日晚上19:30，[直播地址](https://live.bilibili.com/21689802)
- [More](./doc/doc_ch/update.md)



## 特性

- PPOCR系列高质量预训练模型，准确的识别效果
    - 超轻量ppocr_mobile移动端系列：检测（3.0M）+方向分类器（1.4M）+ 识别（5.0M）= 9.4M
    - 通用ppocr_server系列：检测（47.1M）+方向分类器（1.4M）+ 识别（94.9M）= 143.4M
    - 支持中英文数字组合识别、竖排文本识别、长文本识别
    - 支持80+多语言识别，详见[多语言模型](./doc/doc_ch/multi_languages.md)
- 丰富易用的OCR相关工具组件
    - 半自动数据标注工具PPOCRLabel：支持快速高效的数据标注
    - 数据合成工具Style-Text：批量合成大量与目标场景类似的图像
- 支持用户自定义训练，提供丰富的预测推理部署方案
- 支持PIP快速安装使用
- 可运行于Linux、Windows、MacOS等多种系统

## 效果展示

<div align="center">
    <img src="doc/imgs_results/ch_ppocr_mobile_v2.0/test_add_91.jpg" width="800">
    <img src="doc/imgs_results/ch_ppocr_mobile_v2.0/00018069.jpg" width="800">
</div>

上图是通用ppocr_server模型效果展示，更多效果图请见[效果展示页面](./doc/doc_ch/visualization.md)。

<a name="欢迎加入PaddleOCR技术交流群"></a>
## 欢迎加入PaddleOCR技术交流群
- 微信扫描二维码加入官方交流群，获得更高效的问题答疑，与各行各业开发者充分交流，期待您的加入。

<div align="center">
<img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/dygraph/doc/joinus.PNG"  width = "200" height = "200" />
</div>

## 快速体验
- PC端：超轻量级中文OCR在线体验地址：https://www.paddlepaddle.org.cn/hub/scene/ocr

- 移动端：[安装包DEMO下载地址](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)(基于EasyEdge和Paddle-Lite, 支持iOS和Android系统)，Android手机也可以直接扫描下面二维码安装体验。


<div align="center">
<img src="./doc/ocr-android-easyedge.png"  width = "200" height = "200" />
</div>

- 代码体验：从[快速安装](./doc/doc_ch/quickstart.md) 开始

<a name="模型下载"></a>
## PP-OCR 2.0系列模型列表（更新中）
**说明** ：2.0版模型和[1.1版模型](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/models_list.md)的主要区别在于动态图训练vs.静态图训练，模型性能上无明显差距。
| 模型简介     | 模型名称     |推荐场景          | 检测模型 | 方向分类器 | 识别模型 |
| ------------ | --------------- | ----------------|---- | ---------- | -------- |
| 中英文超轻量OCR模型（9.4M） | ch_ppocr_mobile_v2.0_xx |移动端&服务器端|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar)      |
| 中英文通用OCR模型（143.4M）   |ch_ppocr_server_v2.0_xx|服务器端 |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)    |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar)    |[推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar)  |  

更多模型下载（包括多语言），可以参考[PP-OCR v2.0 系列模型下载](./doc/doc_ch/models_list.md)

## 文档教程
- [快速安装](./doc/doc_ch/installation.md)
- [中文OCR模型快速使用](./doc/doc_ch/quickstart.md)
- [多语言OCR模型快速使用](./doc/doc_ch/multi_languages.md)
- [代码组织结构](./doc/doc_ch/tree.md)
- 算法介绍
    - [文本检测](./doc/doc_ch/algorithm_overview.md)
    - [文本识别](./doc/doc_ch/algorithm_overview.md)
    - [PP-OCR Pipeline](#PP-OCR)
    - [端到端PGNet算法](./doc/doc_ch/pgnet.md)
- 模型训练/评估
    - [文本检测](./doc/doc_ch/detection.md)
    - [文本识别](./doc/doc_ch/recognition.md)
    - [方向分类器](./doc/doc_ch/angle_class.md)
    - [yml参数配置文件介绍](./doc/doc_ch/config.md)
- 预测部署
    - [基于pip安装whl包快速推理](./doc/doc_ch/whl.md)
    - [基于Python脚本预测引擎推理](./doc/doc_ch/inference.md)
    - [基于C++预测引擎推理](./deploy/cpp_infer/readme.md)
    - [服务化部署](./deploy/hubserving/readme.md)
    - [端侧部署](./deploy/lite/readme.md)
    - [Benchmark](./doc/doc_ch/benchmark.md)
- 数据集
    - [通用中英文OCR数据集](./doc/doc_ch/datasets.md)
    - [手写中文OCR数据集](./doc/doc_ch/handwritten_datasets.md)
    - [垂类多语言OCR数据集](./doc/doc_ch/vertical_and_multilingual_datasets.md)
- 数据标注与合成
    - [半自动标注工具PPOCRLabel](./PPOCRLabel/README_ch.md)
    - [数据合成工具Style-Text](./StyleText/README_ch.md)
    - [其它数据标注工具](./doc/doc_ch/data_annotation.md)
    - [其它数据合成工具](./doc/doc_ch/data_synthesis.md)
- [效果展示](#效果展示)
- FAQ
    - [【精选】OCR精选10个问题](./doc/doc_ch/FAQ.md)
    - [【理论篇】OCR通用44个问题](./doc/doc_ch/FAQ.md)
    - [【实战篇】PaddleOCR实战174个问题](./doc/doc_ch/FAQ.md)
- [技术交流群](#欢迎加入PaddleOCR技术交流群)
- [参考文献](./doc/doc_ch/reference.md)
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)


<a name="PP-OCR"></a>
## PP-OCR Pipeline
<div align="center">
    <img src="./doc/ppocr_framework.png" width="800">
</div>

PP-OCR是一个实用的超轻量OCR系统。主要由DB文本检测[2]、检测框矫正和CRNN文本识别三部分组成[7]。该系统从骨干网络选择和调整、预测头部的设计、数据增强、学习率变换策略、正则化参数选择、预训练模型使用以及模型自动裁剪量化8个方面，采用19个有效策略，对各个模块的模型进行效果调优和瘦身，最终得到整体大小为3.5M的超轻量中英文OCR和2.8M的英文数字OCR。更多细节请参考PP-OCR技术方案 https://arxiv.org/abs/2009.09941 。其中FPGM裁剪器[8]和PACT量化[9]的实现可以参考[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)。

<a name="效果展示"></a>
## 效果展示 [more](./doc/doc_ch/visualization.md)
- 中文模型
<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/test_add_91.jpg" width="800">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/00015504.jpg" width="800">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/00056221.jpg" width="800">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/rotate_00052204.jpg" width="800">
</div>

- 英文模型
<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/img_12.jpg" width="800">
</div>

- 其他语言模型
<div align="center">
    <img src="./doc/imgs_results/french_0.jpg" width="800">
    <img src="./doc/imgs_results/korean.jpg" width="800">
</div>


<a name="许可证书"></a>
## 许可证书
本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>许可认证。

<a name="贡献代码"></a>
## 贡献代码
我们非常欢迎你为PaddleOCR贡献代码，也十分感谢你的反馈。


- 非常感谢 [Khanh Tran](https://github.com/xxxpsyduck) 和 [Karl Horky](https://github.com/karlhorky) 贡献修改英文文档
- 非常感谢 [zhangxin](https://github.com/ZhangXinNan)([Blog](https://blog.csdn.net/sdlypyzq)) 贡献新的可视化方式、添加.gitignore、处理手动设置PYTHONPATH环境变量的问题
- 非常感谢 [lyl120117](https://github.com/lyl120117) 贡献打印网络结构的代码
- 非常感谢 [xiangyubo](https://github.com/xiangyubo) 贡献手写中文OCR数据集
- 非常感谢 [authorfu](https://github.com/authorfu) 贡献Android和[xiadeye](https://github.com/xiadeye) 贡献IOS的demo代码
- 非常感谢 [BeyondYourself](https://github.com/BeyondYourself) 给PaddleOCR提了很多非常棒的建议，并简化了PaddleOCR的部分代码风格。
- 非常感谢 [tangmq](https://gitee.com/tangmq) 给PaddleOCR增加Docker化部署服务，支持快速发布可调用的Restful API服务。
- 非常感谢 [lijinhan](https://github.com/lijinhan) 给PaddleOCR增加java SpringBoot 调用OCR Hubserving接口完成对OCR服务化部署的使用。
- 非常感谢 [Mejans](https://github.com/Mejans) 给PaddleOCR增加新语言奥克西坦语Occitan的字典和语料。
- 非常感谢 [Evezerest](https://github.com/Evezerest)， [ninetailskim](https://github.com/ninetailskim)， [edencfc](https://github.com/edencfc)， [BeyondYourself](https://github.com/BeyondYourself)， [1084667371](https://github.com/1084667371) 贡献了PPOCRLabel的完整代码。
