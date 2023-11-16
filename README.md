[<img src="https://img.shields.io/badge/Language-English-blue.svg">](README_en.md) | [<img src="https://img.shields.io/badge/Language-简体中文-red.svg">](README.md)

<p align="center">
 <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/PaddleOCR_logo.png" align="middle" width = "600"/>
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
    <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/demo.gif" width="800">
</div>

## 🚀 社区

## 📣 近期更新
- **🔨2023.11 发布 [PP-ChatOCRv2](https://aistudio.baidu.com/projectdetail/paddlex/7050167)**:覆盖20+高频应用场景，支持5种文本图像智能分析能力和部署，包括通用场景关键信息抽取（快递单、营业执照和机动车行驶证等）、复杂文档场景关键信息抽取（解决生僻字、特殊标点、多页pdf、表格等难点问题）、通用OCR、文档场景专用OCR、通用表格识别。针对垂类业务场景，也支持模型训练、微调和Prompt优化。
- **🔥2023.8.7 发布 PaddleOCR [release/2.7](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.7)**
    - 发布[PP-OCRv4](./doc/doc_ch/PP-OCRv4_introduction.md)，提供mobile和server两种模型
      - PP-OCRv4-mobile：速度可比情况下，中文场景效果相比于PP-OCRv3再提升4.5%，英文场景提升10%，80语种多语言模型平均识别准确率提升8%以上
      - PP-OCRv4-server：发布了目前精度最高的OCR模型，中英文场景上检测模型精度提升4.9%， 识别模型精度提升2%
        可参考[快速开始](./doc/doc_ch/quickstart.md) 一行命令快速使用，同时也可在飞桨AI套件(PaddleX)中的[通用OCR产业方案](https://aistudio.baidu.com/aistudio/modelsdetail?modelId=286)中低代码完成模型训练、推理、高性能部署全流程
    - 发布[PP-ChatOCR](https://aistudio.baidu.com/aistudio/modelsdetail?modelId=332) ,使用融合PP-OCR模型和文心大模型的通用场景关键信息抽取全新方案
- 🔨**2022.11 新增实现[4种前沿算法](doc/doc_ch/algorithm_overview.md)**：文本检测 [DRRG](doc/doc_ch/algorithm_det_drrg.md),  文本识别 [RFL](doc/doc_ch/algorithm_rec_rfl.md), 文本超分[Text Telescope](doc/doc_ch/algorithm_sr_telescope.md)，公式识别[CAN](doc/doc_ch/algorithm_rec_can.md)
- **2022.10 优化[JS版PP-OCRv3模型](./deploy/paddlejs/README_ch.md)**：模型大小仅4.3M，预测速度提升8倍，配套web demo开箱即用
- **💥 直播回放：PaddleOCR研发团队详解PP-StructureV2优化策略**。微信扫描[下方二维码](#开源社区)，关注公众号并填写问卷后进入官方交流群，获取直播回放链接与20G重磅OCR学习大礼包（内含PDF转Word应用程序、10种垂类模型、《动手学OCR》电子书等）
- **🔥2022.8.24 发布 PaddleOCR [release/2.6](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)**
  - 发布[PP-StructureV2](./ppstructure/README_ch.md)，系统功能性能全面升级，适配中文场景，新增支持[版面复原](./ppstructure/recovery/README_ch.md)，支持**一行命令完成PDF转Word**；
  - [版面分析](./ppstructure/layout/README_ch.md)模型优化：模型存储减少95%，速度提升11倍，平均CPU耗时仅需41ms；
  - [表格识别](./ppstructure/table/README_ch.md)模型优化：设计3大优化策略，预测耗时不变情况下，模型精度提升6%；
  - [关键信息抽取](./ppstructure/kie/README_ch.md)模型优化：设计视觉无关模型结构，语义实体识别精度提升2.8%，关系抽取精度提升9.1%。
- 🔥**2022.8 发布 [OCR场景应用集合](./applications)**：包含数码管、液晶屏、车牌、高精度SVTR模型、手写体识别等**9个垂类模型**，覆盖通用，制造、金融、交通行业的主要OCR垂类应用。

> [更多](./doc/doc_ch/update.md)

## 🌟 特性

支持多种 OCR 相关前沿算法，包括但不限于文本检测、文本识别、表格识别等。在此基础上打造产业级特色模型 PP-OCR、PP-Structure 和 PP-ChatOCR，并打通数据生产、模型训练、压缩、预测部署全流程，为开发者提供一站式解决方案。

<div align="center">
    <img src="./docs/images/ppocrv4.png">
</div>

## ⚡ [快速开始](https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html)

## 🔥 [低代码全流程开发](https://paddlepaddle.github.io/PaddleOCR/latest/paddlex/overview.html)

## 📝 文档

- 在线网站体验：
    - PP-OCRv4 在线体验地址：https://aistudio.baidu.com/application/detail/7658
    - PP-ChatOCR 在线体验地址：https://aistudio.baidu.com/application/detail/8519
- 一行命令快速使用：[快速开始（中英文/多语言/文档分析）](./doc/doc_ch/quickstart.md)
- 飞桨AI套件（PaddleX）中训练、推理、高性能部署全流程体验：
    - PP-OCRv4：https://aistudio.baidu.com/projectdetail/paddlex/6796224
    - PP-ChatOCR：https://aistudio.baidu.com/projectdetail/paddlex/6796372
    - PP-ChatOCRv2：https://aistudio.baidu.com/projectdetail/paddlex/7050167
- 移动端demo体验：[安装包DEMO下载地址](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)(基于EasyEdge和Paddle-Lite, 支持iOS和Android系统)

<a name="技术交流合作"></a>
## 📖 技术交流合作
- 飞桨AI套件（PaddleX）—— 精选产业实用模型的一站式开发平台。包含如下特点：
* 【优质的算法库】包含10大任务领域的36个精选模型，实现在一个平台中完成不同任务模型算法的开发。更多领域模型持续丰富中！PaddleX还提供完善的模型训练推理benchmark数据，服务开发者基于业务需求选择最合适的模型。
* 【简易的开发方式】工具箱/开发者双模式联动，无代码+低代码开发方式，四步完成数据、训练、验证、部署的全流程AI开发。
* 【高效的训练部署】沉淀百度算法团队的最佳调优策略，实现每个模型都能最快最优地收敛。完善的部署SDK支持，实现跨平台、跨硬件的快速产业级部署（服务化部署能力完善中）。
* 【丰富的国产硬件支持】PaddleX除了在AI Studio云端使用，还沉淀了windows本地端，正在丰富Linux版本、昆仑芯版本、昇腾版本、寒武纪版本。
* 【共赢的联创共建】除了便捷地开发AI应用外，PaddleX还为大家提供了获取商业收益的机会，为企业探索更多商业空间。

作为一款高效的开发神器，PaddleX值得每一位开发者拥有。

PaddleX官网地址：https://www.paddlepaddle.org.cn/paddle/paddleX

欢迎微信扫描下方二维码或者点击[链接](https://aistudio.baidu.com/community/channel/610) 进入AI Studio【PaddleX社区频道】获得更高效的技术答疑～

<div align="center">
<img src="https://user-images.githubusercontent.com/45199522/279737332-e9f960f7-f0e5-4b92-95fb-79313bee2d89.png"  width = "150" height = "150",caption='' />
<p>飞桨AI套件【PaddleX】社区频道二维码</p>
</div>

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>

## ⭐️ Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)

## 📄 许可证书

| 模型简介                              | 模型名称                | 推荐场景        | 检测模型                                                     | 方向分类器                                                   | 识别模型                                                     |
| ------------------------------------- | ----------------------- | --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 中英文超轻量PP-OCRv4模型（15.8M）     | ch_PP-OCRv4_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar) |
| 中英文超轻量PP-OCRv3模型（16.2M）     | ch_PP-OCRv3_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
| 英文超轻量PP-OCRv3模型（13.4M）     | en_PP-OCRv3_xx          | 移动端&服务器端 | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [推理模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [训练模型](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |

- 超轻量OCR系列更多模型下载（包括多语言），可以参考[PP-OCR系列模型下载](./doc/doc_ch/models_list.md)，文档分析相关模型参考[PP-Structure系列模型下载](./ppstructure/docs/models_list.md)

### PaddleOCR场景应用模型

| 行业 | 类别         | 亮点                               | 文档说明                                                     | 模型下载                                      |
| ---- | ------------ | ---------------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| 制造 | 数码管识别   | 数码管数据合成、漏识别调优         | [光功率计数码管字符识别](./applications/光功率计数码管字符识别/光功率计数码管字符识别.md) | [下载链接](./applications/README.md#模型下载) |
| 金融 | 通用表单识别 | 多模态通用表单结构化提取           | [多模态表单识别](./applications/多模态表单识别.md)           | [下载链接](./applications/README.md#模型下载) |
| 交通 | 车牌识别     | 多角度图像处理、轻量模型、端侧部署 | [轻量级车牌识别](./applications/轻量级车牌识别.md)           | [下载链接](./applications/README.md#模型下载) |

- 更多制造、金融、交通行业的主要OCR垂类应用模型（如电表、液晶屏、高精度SVTR模型等），可参考[场景应用模型下载](./applications)

<a name="文档教程"></a>

## 📖 文档教程

- [运行环境准备](./doc/doc_ch/environment.md)
- [PP-OCR文本检测识别🔥](./doc/doc_ch/ppocr_introduction.md)
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
        - [基于Python预测引擎推理](./doc/doc_ch/inference_ppocr.md)
        - [基于C++预测引擎推理](./deploy/cpp_infer/readme_ch.md)
        - [服务化部署](./deploy/pdserving/README_CN.md)
        - [端侧部署](./deploy/lite/readme.md)
        - [Paddle2ONNX模型转化与预测](./deploy/paddle2onnx/readme.md)
        - [云上飞桨部署工具](./deploy/paddlecloud/README.md)
        - [Benchmark](./doc/doc_ch/benchmark.md)
- [PP-Structure文档分析🔥](./ppstructure/README_ch.md)
    - [快速开始](./ppstructure/docs/quickstart.md)
    - [模型库](./ppstructure/docs/models_list.md)
    - [模型训练](./doc/doc_ch/training.md)
        - [版面分析](./ppstructure/layout/README_ch.md)
        - [表格识别](./ppstructure/table/README_ch.md)
        - [关键信息提取](./ppstructure/kie/README_ch.md)
    - [推理部署](./deploy/README_ch.md)
        - [基于Python预测引擎推理](./ppstructure/docs/inference.md)
        - [基于C++预测引擎推理](./deploy/cpp_infer/readme_ch.md)
        - [服务化部署](./deploy/hubserving/readme.md)
- [前沿算法与模型🚀](./doc/doc_ch/algorithm_overview.md)
    - [文本检测算法](./doc/doc_ch/algorithm_overview.md)
    - [文本识别算法](./doc/doc_ch/algorithm_overview.md)
    - [端到端OCR算法](./doc/doc_ch/algorithm_overview.md)
    - [表格识别算法](./doc/doc_ch/algorithm_overview.md)
    - [关键信息抽取算法](./doc/doc_ch/algorithm_overview.md)
    - [使用PaddleOCR架构添加新算法](./doc/doc_ch/add_new_algorithm.md)
- [场景应用](./applications)
- 数据标注与合成
    - [半自动标注工具PPOCRLabel](./PPOCRLabel/README_ch.md)
    - [数据合成工具Style-Text](./StyleText/README_ch.md)
    - [其它数据标注工具](./doc/doc_ch/data_annotation.md)
    - [其它数据合成工具](./doc/doc_ch/data_synthesis.md)
- 数据集
    - [通用中英文OCR数据集](doc/doc_ch/dataset/datasets.md)
    - [手写中文OCR数据集](doc/doc_ch/dataset/handwritten_datasets.md)
    - [垂类多语言OCR数据集](doc/doc_ch/dataset/vertical_and_multilingual_datasets.md)
    - [版面分析数据集](doc/doc_ch/dataset/layout_datasets.md)
    - [表格识别数据集](doc/doc_ch/dataset/table_datasets.md)
    - [关键信息提取数据集](doc/doc_ch/dataset/kie_datasets.md)
- [代码组织结构](./doc/doc_ch/tree.md)
- [效果展示](#效果展示)
- [《动手学OCR》电子书📚](./doc/doc_ch/ocr_book.md)
- [开源社区](#开源社区)
- FAQ
    - [通用问题](./doc/doc_ch/FAQ.md)
    - [PaddleOCR实战问题](./doc/doc_ch/FAQ.md)
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
