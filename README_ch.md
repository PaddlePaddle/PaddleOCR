[English](README.md) | 简体中文 | [हिन्दी](./doc/doc_i18n/README_हिन्द.md) | [日本語](./doc/doc_i18n/README_日本語.md) | [한국인](./doc/doc_i18n/README_한국어.md) | [Pу́сский язы́к](./doc/doc_i18n/README_Ру́сский_язы́к.md)

<p align="center">
 <img src="./doc/PaddleOCR_log.png" align="middle" width = "600"/>
<p align="center">
<p align="center">
    <a href="https://discord.gg/z9xaRVjdbD"><img src="https://img.shields.io/badge/Chat-on%20discord-7289da.svg?sanitize=true" alt="Chat"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
</p>

## 简介
PaddleOCR 旨在打造一套丰富、领先、且实用的 OCR 工具库，助力开发者训练出更好的模型，并应用落地。

<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/test_add_91.jpg" width="800">
</div>

<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/00006737.jpg" width="800">
</div>

## 🚀 社区
PaddleOCR 由 [PMC](https://github.com/PaddlePaddle/PaddleOCR/issues/12122) 监督。Issues 和 PRs 将在尽力的基础上进行审查。欲了解 PaddlePaddle 社区的完整概况，请访问 [community](https://github.com/PaddlePaddle/community)。

⚠️注意：[Issues](https://github.com/PaddlePaddle/PaddleOCR/issues)模块仅用来报告程序🐞Bug，其余提问请移步[Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions)模块提问。如所提Issue不是Bug，会被移到Discussions模块，敬请谅解。

## 📣 近期更新
- **📚直播和OCR实战打卡营预告**：《PP-ChatOCRv2赋能金融报告信息智能化抽取，新金融效率再升级》课程上线，破解复杂版面、表格识别、信息抽取OCR解析难题，直播时间：6月6日（周四）19：00。并于6月11日启动【政务采购合同信息抽取】实战打卡营。报名链接：https://www.wjx.top/vm/eBcYmqO.aspx?udsid=197406

- **🔥2024.7 添加 PaddleOCR 算法模型挑战赛冠军方案**：
    - 赛题一：OCR 端到端识别任务冠军方案——[场景文本识别算法-SVTRv2](doc/doc_ch/algorithm_rec_svtrv2.md)；
    - 赛题二：通用表格识别任务冠军方案——[表格识别算法-SLANet-LCNetV2](doc/doc_ch/algorithm_table_slanet.md)。

- **🔥2024.5.10 上线星河零代码产线(OCR 相关)**：全面覆盖了以下四大 OCR 核心任务，提供极便捷的 Badcase 分析和实用的在线体验：
  - [通用 OCR](https://aistudio.baidu.com/community/app/91660) (PP-OCRv4)。
  - [通用表格识别](https://aistudio.baidu.com/community/app/91661) (SLANet)。
  - [通用图像信息抽取](https://aistudio.baidu.com/community/app/91662) (PP-ChatOCRv2-common)。
  - [文档场景信息抽取](https://aistudio.baidu.com/community/app/70303) (PP-ChatOCRv2-doc)。

  同时采用了 **[全新的场景任务开发范式](https://aistudio.baidu.com/pipeline/mine)** ,将模型统一汇聚，实现训练部署的零代码开发，并支持在线服务化部署和导出离线服务化部署包。

- **🔥2023.8.7 发布 PaddleOCR [release/2.7](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.7)**
    - 发布[PP-OCRv4](./doc/doc_ch/PP-OCRv4_introduction.md)，提供 mobile 和 server 两种模型
      - PP-OCRv4-mobile：速度可比情况下，中文场景效果相比于 PP-OCRv3 再提升 4.5%，英文场景提升 10%，80 语种多语言模型平均识别准确率提升 8%以上
      - PP-OCRv4-server：发布了目前精度最高的 OCR 模型，中英文场景上检测模型精度提升 4.9%， 识别模型精度提升 2%
      可参考[快速开始](./doc/doc_ch/quickstart.md) 一行命令快速使用，同时也可在飞桨 AI 套件(PaddleX)中的[通用 OCR 产业方案](https://aistudio.baidu.com/aistudio/modelsdetail?modelId=286)中低代码完成模型训练、推理、高性能部署全流程
- 🔨**2022.11 新增实现[4 种前沿算法](doc/doc_ch/algorithm_overview.md)**：文本检测 [DRRG](doc/doc_ch/algorithm_det_drrg.md),  文本识别 [RFL](doc/doc_ch/algorithm_rec_rfl.md), 文本超分[Text Telescope](doc/doc_ch/algorithm_sr_telescope.md)，公式识别[CAN](doc/doc_ch/algorithm_rec_can.md)
- **2022.10 优化[JS 版 PP-OCRv3 模型](./deploy/paddlejs/README_ch.md)**：模型大小仅 4.3M，预测速度提升 8 倍，配套 web demo 开箱即用
- **💥 直播回放：PaddleOCR 研发团队详解 PP-StructureV2 优化策略**。微信扫描[下方二维码](#开源社区)，关注公众号并填写问卷后进入官方交流群，获取直播回放链接与 20G 重磅 OCR 学习大礼包（内含 PDF 转 Word 应用程序、10 种垂类模型、《动手学 OCR》电子书等）
- **🔥2022.8.24 发布 PaddleOCR [release/2.6](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)**
  - 发布[PP-StructureV2](./ppstructure/README_ch.md)，系统功能性能全面升级，适配中文场景，新增支持[版面复原](./ppstructure/recovery/README_ch.md)，支持**一行命令完成 PDF 转 Word**；
  - [版面分析](./ppstructure/layout/README_ch.md)模型优化：模型存储减少 95%，速度提升 11 倍，平均 CPU 耗时仅需 41ms；
  - [表格识别](./ppstructure/table/README_ch.md)模型优化：设计 3 大优化策略，预测耗时不变情况下，模型精度提升 6%；
  - [关键信息抽取](./ppstructure/kie/README_ch.md)模型优化：设计视觉无关模型结构，语义实体识别精度提升 2.8%，关系抽取精度提升 9.1%。
- 🔥**2022.8 发布 [OCR 场景应用集合](./applications)**：包含数码管、液晶屏、车牌、高精度 SVTR 模型、手写体识别等**9 个垂类模型**，覆盖通用，制造、金融、交通行业的主要 OCR 垂类应用。

> [更多](./doc/doc_ch/update.md)

## 🌟 特性

支持多种 OCR 相关前沿算法，在此基础上打造产业级特色模型[PP-OCR](./doc/doc_ch/ppocr_introduction.md)、[PP-Structure](./ppstructure/README_ch.md)和[PP-ChatOCRv2](https://aistudio.baidu.com/community/app/70303)，并打通数据生产、模型训练、压缩、预测部署全流程。

<div align="center">
    <img src="https://raw.githubusercontent.com/tink2123/test/master/ppocrv4.png">
</div>

> 上述内容的使用方法建议从文档教程中的快速开始体验


## ⚡ 快速开始

- 在线免费体验：
    - PP-OCRv4 在线体验地址：https://aistudio.baidu.com/community/app/91660
    - SLANet 在线体验地址：https://aistudio.baidu.com/community/app/91661
    - PP-ChatOCRv2-common 在线体验地址：https://aistudio.baidu.com/community/app/91662
    - PP-ChatOCRv2-doc 在线体验地址：https://aistudio.baidu.com/community/app/70303

- 一行命令快速使用：[快速开始（中英文/多语言/文档分析）](./doc/doc_ch/quickstart.md)
- 移动端 demo 体验：[安装包 DEMO 下载地址](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)(基于 EasyEdge 和 Paddle-Lite, 支持 iOS 和 Android 系统)

## 📖 技术交流合作

- 飞桨低代码开发工具 PaddleX 官方交流频道：https://aistudio.baidu.com/community/channel/610

## 📚《动手学 OCR》电子书
- [《动手学 OCR》电子书](./doc/doc_ch/ocr_book.md)

## 🛠️ PP-OCR 系列模型列表（更新中）

| 模型简介                              | 模型名称                | 推荐场景        | 检测模型                                                     | 方向分类器                                                   | 识别模型                                                     |
| ------------------------------------- | ----------------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 中英文超轻量 PP-OCRv4 模型（15.8M）     | ch_PP-OCRv4_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar) |
| 中英文超轻量 PP-OCRv3 模型（16.2M）     | ch_PP-OCRv3_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
| 英文超轻量 PP-OCRv3 模型（13.4M）     | en_PP-OCRv3_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |

- 超轻量 OCR 系列更多模型下载（包括多语言），可以参考[PP-OCR 系列模型下载](./doc/doc_ch/models_list.md)，文档分析相关模型参考[PP-Structure 系列模型下载](./ppstructure/docs/models_list.md)

### PaddleOCR 场景应用模型

| 行业 | 类别         | 亮点                               | 文档说明                                                     | 模型下载                                      |
| ---- | ------------ | ---------------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| 制造 | 数码管识别   | 数码管数据合成、漏识别调优         | [光功率计数码管字符识别](./applications/光功率计数码管字符识别/光功率计数码管字符识别.md) | [下载链接](./applications/README.md#模型下载) |
| 金融 | 通用表单识别 | 多模态通用表单结构化提取           | [多模态表单识别](./applications/多模态表单识别.md)           | [下载链接](./applications/README.md#模型下载) |
| 交通 | 车牌识别     | 多角度图像处理、轻量模型、端侧部署 | [轻量级车牌识别](./applications/轻量级车牌识别.md)           | [下载链接](./applications/README.md#模型下载) |

- 更多制造、金融、交通行业的主要 OCR 垂类应用模型（如电表、液晶屏、高精度 SVTR 模型等），可参考[场景应用模型下载](./applications)

## 📖 文档教程

- [运行环境准备](./doc/doc_ch/environment.md)
- [PP-OCR 文本检测识别🔥](./doc/doc_ch/ppocr_introduction.md)
    - [快速开始](./doc/doc_ch/quickstart.md)
    - [模型库](./doc/doc_ch/models_list.md)
    - [模型训练](./doc/doc_ch/training.md)
        - [文本检测](./doc/doc_ch/detection.md)
        - [文本识别](./doc/doc_ch/recognition.md)
        - [文本方向分类器](./doc/doc_ch/angle_class.md)
    - 模型压缩
        - [模型量化](./deploy/slim/quantization/README.md)
        - [模型裁剪](./deploy/slim/prune/README.md)
        - [知识蒸馏](./doc/doc_ch/knowledge_distillation.md)
    - [推理部署](./deploy/README_ch.md)
        - [基于 Python 预测引擎推理](./doc/doc_ch/inference_ppocr.md)
        - [基于 C++预测引擎推理](./deploy/cpp_infer/readme_ch.md)
        - [服务化部署](./deploy/pdserving/README_CN.md)
        - [端侧部署](./deploy/lite/readme.md)
        - [Paddle2ONNX 模型转化与预测](./deploy/paddle2onnx/readme_ch.md)
        - [云上飞桨部署工具](./deploy/paddlecloud/README.md)
        - [Benchmark](./doc/doc_ch/benchmark.md)
- [PP-Structure 文档分析🔥](./ppstructure/README_ch.md)
    - [快速开始](./ppstructure/docs/quickstart.md)
    - [模型库](./ppstructure/docs/models_list.md)
    - [模型训练](./doc/doc_ch/training.md)
        - [版面分析](./ppstructure/layout/README_ch.md)
        - [表格识别](./ppstructure/table/README_ch.md)
        - [关键信息提取](./ppstructure/kie/README_ch.md)
    - [推理部署](./deploy/README_ch.md)
        - [基于 Python 预测引擎推理](./ppstructure/docs/inference.md)
        - [基于 C++预测引擎推理](./deploy/cpp_infer/readme_ch.md)
        - [服务化部署](./deploy/hubserving/readme.md)
- [前沿算法与模型🚀](./doc/doc_ch/algorithm_overview.md)
    - [文本检测算法](./doc/doc_ch/algorithm_overview.md)
    - [文本识别算法](./doc/doc_ch/algorithm_overview.md)
    - [端到端 OCR 算法](./doc/doc_ch/algorithm_overview.md)
    - [表格识别算法](./doc/doc_ch/algorithm_overview.md)
    - [关键信息抽取算法](./doc/doc_ch/algorithm_overview.md)
    - [使用 PaddleOCR 架构添加新算法](./doc/doc_ch/add_new_algorithm.md)
- [场景应用](./applications)
- 数据标注与合成
    - [半自动标注工具 PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel/blob/main/README_ch.md)
    - [数据合成工具 Style-Text](https://github.com/PFCCLab/StyleText/blob/main/README_ch.md)
    - [其它数据标注工具](./doc/doc_ch/data_annotation.md)
    - [其它数据合成工具](./doc/doc_ch/data_synthesis.md)
- 数据集
    - [通用中英文 OCR 数据集](doc/doc_ch/dataset/datasets.md)
    - [手写中文 OCR 数据集](doc/doc_ch/dataset/handwritten_datasets.md)
    - [垂类多语言 OCR 数据集](doc/doc_ch/dataset/vertical_and_multilingual_datasets.md)
    - [版面分析数据集](doc/doc_ch/dataset/layout_datasets.md)
    - [表格识别数据集](doc/doc_ch/dataset/table_datasets.md)
    - [关键信息提取数据集](doc/doc_ch/dataset/kie_datasets.md)
- [代码组织结构](./doc/doc_ch/tree.md)
- [效果展示](#效果展示)
- [《动手学 OCR》电子书📚](./doc/doc_ch/ocr_book.md)
- [开源社区](#开源社区)
- FAQ
    - [通用问题](./doc/doc_ch/FAQ.md)
    - [PaddleOCR 实战问题](./doc/doc_ch/FAQ.md)
- [参考文献](./doc/doc_ch/reference.md)
- [许可证书](#许可证书)


<a name="效果展示"></a>

## 👀 效果展示 [more](./doc/doc_ch/visualization.md)

<details open>
<summary>PP-OCRv3 中文模型</summary>

<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic001.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic002.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic003.jpg" width="800">
</div>

</details>


<details open>
<summary>PP-OCRv3 英文模型</summary>

<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/en/en_1.png" width="800">
    <img src="doc/imgs_results/PP-OCRv3/en/en_2.png" width="800">
</div>

</details>


<details open>
<summary>PP-OCRv3 多语言模型</summary>

<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/multi_lang/japan_2.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/multi_lang/korean_1.jpg" width="800">
</div>

</details>

<details open>
<summary>PP-Structure 文档分析</summary>

- 版面分析+表格识别
<div align="center">
    <img src="./ppstructure/docs/table/ppstructure.GIF" width="800">
</div>

- SER（语义实体识别）
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808.jpg" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a.png" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/197464552-69de557f-edff-4c7f-acbf-069df1ba097f.png" width="600">
</div>

- RE（关系提取）
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb.jpg" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185540080-0431e006-9235-4b6d-b63d-0b3c6e1de48f.jpg" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094813-3a8e16cc-42e5-4982-b9f4-0134dfb5688d.png" width="600">
</div>

</details>

<a name="许可证书"></a>

## 许可证书
本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>许可认证。
