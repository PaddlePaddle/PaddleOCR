[English](README_en.md) | 简体中文

## 简介
PaddleOCR旨在打造一套丰富、领先、且实用的OCR工具库，助力使用者训练出更好的模型，并应用落地。

**近期更新**
- 2020.9.17 更新超轻量ppocr_mobile系列和通用ppocr_server系列系列中英文ocr模型，效果媲美商业效果。[模型下载](#模型下载)
- 2020.8.26 更新OCR相关的84个常见问题及解答，具体参考[FAQ](./doc/doc_ch/FAQ.md)
- 2020.8.24 支持通过whl包安装使用PaddleOCR，具体参考[Paddleocr Package使用说明](./doc/doc_ch/whl.md)
- 2020.8.21 更新8月18日B站直播课回放和PPT，课节2，易学易用的OCR工具大礼包，[获取地址](https://aistudio.baidu.com/aistudio/education/group/info/1519)
- 2020.8.16 开源文本检测算法[SAST](https://arxiv.org/abs/1908.05498)和文本识别算法[SRN](https://arxiv.org/abs/2003.12294)
- 2020.7.23 发布7月21日B站直播课回放和PPT，课节1，PaddleOCR开源大礼包全面解读，[获取地址](https://aistudio.baidu.com/aistudio/course/introduce/1519)
- 2020.7.15 添加基于EasyEdge和Paddle-Lite的移动端DEMO，支持iOS和Android系统
- [more](./doc/doc_ch/update.md)


## 特性

- PPOCR系列高质量预训练模型，媲美商业效果
    - 超轻量ppocr_mobile系列：检测（2.5M）+方向分类器（0.9M
    ）+ 识别（4.5M）= 7.9M
    - 通用ppocr_server系列：检测（47.2M）+方向分类器（0.9M）+ 识别（107M）= 155.1M
    - 超轻量压缩ppocr_mobile_slim系列：(coming soon)
- 支持中英文数字组合识别、竖排文本识别、长文本识别
- 支持多语言识别：韩语、日语、德语、法语 (coming soon)
- 支持用户自定义训练，提供丰富的预测推理部署方案
- 支持PIP快速安装使用
- 可运行于Linux、Windows、MacOS等多种系统

## 效果展示

<div align="center">
    <img src="doc/imgs_results/1101.jpg" width="800">
    <img src="doc/imgs_results/1102.jpg" width="800">
</div>

上图是超轻量级中文OCR模型效果展示，更多效果图请见[效果展示页面](./doc/doc_ch/visualization.md)。

## 快速体验
- PC端：超轻量级中文OCR在线体验地址：https://www.paddlepaddle.org.cn/hub/scene/ocr

- 移动端：[安装包DEMO下载地址](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)(基于EasyEdge和Paddle-Lite, 支持iOS和Android系统)，Android手机也可以直接扫描下面二维码安装体验。


<div align="center">
<img src="./doc/ocr-android-easyedge.png"  width = "200" height = "200" />
</div>

- 代码体验：可以直接进入[快速安装](./doc/doc_ch/installation.md)  

<a name="模型下载"></a>
## PP-OCR 1.1系列模型列表（9月17日更新）

| 模型简介     | 模型名称     |推荐场景          | 检测模型 | 方向分类器 | 识别模型 |      |
| ------------ | --------------- | ----------------|---- | ---------- | -------- | ---- |
| 中英文超轻量OCR模型（7.9M） | ch_ppocr_mobile_v1.1_xx |移动端&服务器端|[预测模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_train.tar)|[预测模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile-v1.1.cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile-v1.1.cls_train.tar) |[预测模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_pre.tar)      | |
| 中英文通用OCR模型（155.1M）   |ch_ppocr_server_v1.1_xx|服务器端 |[预测模型](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/server/det/ch_ppocr_server_v1.1_det_train.tar)          |[预测模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile-v1.1.cls_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile-v1.1.cls_train.tar)    |[预测模型](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/20-09-22/server/rec/ch_ppocr_server_v1.1_rec_pre.tar)  |  |
| 中英文超轻量压缩OCR模型 | ch_ppocr_mobile_slim_v1.1_xx| 移动端 |即将开源 |即将开源|即将开源| |      ||

更多V1.1版本模型下载，可以参考[OCR1.1模型列表](./doc/doc_ch/models_list.md)

## PP-OCR 1.0系列模型列表（7月16日更新）

| 模型简介     | 模型名称               | 检测模型 | 识别模型 | 支持空格的识别模型 |      |
| ------------ | ---------------------- | -------- | ---------- | -------- | ---- |
| 超轻量中英文OCR模型（8.6M） | chinese_db_crnn_mobile_xx |[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar) |      |
|通用中文OCR模型（212M）|chinese_db_crnn_server_xx|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|[inference模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) / [预训练模型](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)| |


## 文档教程
- [快速安装](./doc/doc_ch/installation.md)
- [中文OCR模型快速使用](./doc/doc_ch/quickstart.md)
- 算法介绍
    - [文本检测](./doc/doc_ch/algorithm_overview.md)
    - [文本识别](./doc/doc_ch/algorithm_overview.md)
    - PP-OCR (coming soon)
- 模型训练/评估
    - [文本检测](./doc/doc_ch/detection.md)
    - [文本识别](./doc/doc_ch/recognition.md)
    - [yml参数配置文件介绍](./doc/doc_ch/config.md)
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
- [效果展示](#超轻量级中文OCR效果展示)
- FAQ
    - [【精选】OCR精选10个问题](./doc/doc_ch/FAQ.md)
    - [【理论篇】OCR通用21个问题](./doc/doc_ch/FAQ.md)
    - [【实战篇】PaddleOCR实战53个问题](./doc/doc_ch/FAQ.md)
- [技术交流群](#欢迎加入PaddleOCR技术交流群)
- [参考文献](./doc/doc_ch/reference.md)
- [许可证书](#许可证书)
- [贡献代码](#贡献代码)

<a name="PP-OCR"></a>





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
