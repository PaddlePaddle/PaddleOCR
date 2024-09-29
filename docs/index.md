---
comments: true
typora-copy-images-to: images
hide:
  - navigation
  - toc
---

<div align="center">
 <img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/doc/PaddleOCR_log.png" align="middle" width = "600"/>
  <p align="center">
      <a href="https://discord.gg/z9xaRVjdbD"><img src="https://img.shields.io/badge/Chat-on%20discord-7289da.svg?sanitize=true" alt="Chat"></a>
      <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
      <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
      <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
      <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
      <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
      <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
  </p>
</div>

## 简介

PaddleOCR 旨在打造一套丰富、领先、且实用的 OCR 工具库，助力开发者训练出更好的模型，并应用落地。

## 🚀 社区

PaddleOCR 由 [PMC](https://github.com/PaddlePaddle/PaddleOCR/issues/12122) 监督。Issues 和 PRs 将在尽力的基础上进行审查。欲了解 PaddlePaddle 社区的完整概况，请访问 [community](https://github.com/PaddlePaddle/community)。

⚠️注意：[Issues](https://github.com/PaddlePaddle/PaddleOCR/issues)模块仅用来报告程序🐞Bug，其余提问请移步[Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions)模块提问。如所提Issue不是Bug，会被移到Discussions模块，敬请谅解。

## 📣 近期更新

- **🔥2024.9.30 发布PaddleOCR release/2.9**:
  
  * 发布文档图像智能分析[PP-ChatOCRv3](/docs/paddlex/pipeline_usage/document_scene_information_extraction.md)
  * 发布PaddleOCR 训练推理[全流程低代码开发方案](/docs/paddlex/quick_start.md)
    * 🎨 模型丰富一键调用：将**通用OCR、通用表格识别和文档场景信息抽取**的XXX模型整合为3条模型产线，通过极简的**Python API一键调用**，快速体验模型效果。同时覆盖目标检测、语义分割、时序预测等多个热门AI领域**20+单功能模块**，方便开发者进行模型组合使用。
    * 🚀提高效率降低门槛：实现基于图形界面和统一命令的模型**全流程开发**，支持**高性能部署、服务化部署和端侧部署**等多种部署方式，同时支持英伟达GPU、昆仑芯、昇腾、寒武纪和海光等**多种主流硬件的无缝切换**。
  * 新增4个高精度[版面分析模型](/docs/paddlex/module_usage/layout_detection.md)
  * 新增高精度表格结构识别模型[SLANet_Plus](/docs/paddlex/module_usage/table_structure_recognition.md)
  * 新增版面矫正预测模型[UVDoc](/docs/paddlex/module_usage/image_warping.md)
  * 新增公式识别模型[LatexOCR](/docs/paddlex/module_usage/formula_recognition.md)
  
- **🔥2024.7 添加 PaddleOCR 算法模型挑战赛冠军方案**：
    - 赛题一：OCR 端到端识别任务冠军方案——[场景文本识别算法-SVTRv2](https://paddlepaddle.github.io/PaddleOCR/algorithm/text_recognition/algorithm_rec_svtrv2.html)；
    - 赛题二：通用表格识别任务冠军方案——[表格识别算法-SLANet-LCNetV2](https://paddlepaddle.github.io/PaddleOCR/algorithm/table_recognition/algorithm_table_slanet.html)。

- **💥2024.6.27 飞桨低代码开发工具 [PaddleX 3.0](https://github.com/paddlepaddle/paddlex) 重磅更新,低门槛轻松玩转OCR 模型开发！**
    - 🎨 **模型丰富一键调用**：将**文本检测、表格识别、文档场景信息抽取**等OCR领域关键任务整合为3条模型产线，通过极简的Python API**一键调用**，快速体验模型效果。同时覆盖目标检测、语义分割、时序预测等多个热门AI领域的**10条模型产线**和**20+单功能模块**，方便开发者进行模型组合使用。
    - 🚀 **提高效率降低门槛**：实现基于图形界面和统一命令的模型**全流程开发**，打造大小模型结合、大模型半监督学习和多模型融合的8条特色模型产线，大幅度降低迭代模型的成本。
    - 🌐 **多种场景灵活部署**：支持**高性能部署**、**服务化部署**和**端侧部署**等多种部署方式，确保不同应用场景下模型的高效运行和快速响应。
    - 🔧 **主流硬件高效支持**：支持英伟达 GPU、昆仑芯、昇腾和寒武纪等**多种主流硬件**的无缝切换，确保高效运行。


> [更多](./update.md)

## 🌟 特性

支持多种 OCR 相关前沿算法，在此基础上打造产业级特色模型[PP-OCR](./ppocr/overview.md)、[PP-Structure](./ppstructure/overview.md)和[PP-ChatOCR](https://aistudio.baidu.com/community/app/70303)，并打通数据生产、模型训练、压缩、预测部署全流程。

<img src="./images/ppocrv4.png" width="600" />

## 效果展示

### 超轻量PP-OCRv3效果展示

#### PP-OCRv3中文模型

![img](./images/test_add_91.jpg)

<img src="./images/00006737.jpg" width="600" />

<img src="./images/PP-OCRv3-pic001.jpg" width="600" />

<img src="./images/PP-OCRv3-pic002.jpg" width="600" />

<img src="./images/PP-OCRv3-pic003.jpg" width="600" />

#### PP-OCRv3英文数字模型

<img src="./images/en_1.png" width="600" />

<img src="./images/en_2.png" width="600" />

<img src="./images/en_3-0398013.png" width="600" />

#### PP-OCRv3多语言模型

<img src="./images/japan_2.jpg" width="600" />

<img src="./images/korean_1.jpg" width="600" />

#### PP-Structure 文档分析

- 版面分析+表格识别

  <img src="./images/ppstructure-20240708082235651.gif" width="600" />

- SER（语义实体识别）

  <img src="./images/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808-20240708082238739.jpg" width="600" />

  <img src="./images/185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a-20240708082247529.png" width="600" />

  <img src="./images/197464552-69de557f-edff-4c7f-acbf-069df1ba097f-20240708082253634.png" width="600" />

- RE（关系提取）

  <img src="./images/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb-20240708082310650.jpg" width="600" />

  <img src="./images/185540080-0431e006-9235-4b6d-b63d-0b3c6e1de48f-20240708082316558.jpg" width="600" />

  <img src="./images/186094813-3a8e16cc-42e5-4982-b9f4-0134dfb5688d-20240708082323916.png" width="600" />

## 许可证书

本项目的发布受<a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>许可认证。
