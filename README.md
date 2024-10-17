[English](README_en.md) | 简体中文

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

PaddleOCR 由 [PMC](https://github.com/PaddlePaddle/PaddleOCR/issues/12122) 监督。Issues 和 PRs 将在尽力的基础上进行审查。欲了解 PaddlePaddle 社区的完整概况，请访问 [community](https://github.com/PaddlePaddle/community)。

⚠️注意：[Issues](https://github.com/PaddlePaddle/PaddleOCR/issues)模块仅用来报告程序🐞Bug，其余提问请移步[Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions)模块提问。如所提Issue不是Bug，会被移到Discussions模块，敬请谅解。

## 📣 近期更新([more](https://paddlepaddle.github.io/PaddleOCR/latest/update.html))

- 🔥🔥《PaddleX文档信息个性化抽取新升级》，PP-ChatOCRv3创新性提供了基于数据融合技术的OCR模型二次开发功能，具备更强的模型微调能力。百万级高质量通用OCR文本识别数据，按特定比例自动融入垂类模型训练数据，破解产业垂类模型训练导致通用文本识别能力减弱难题。适用自动化办公、金融风控、医疗健康、教育出版、法律党政等产业实际场景。**10月17日（周四）19：00**直播为您详细解读数据融合技术以及如何利用提示词工程实现更好的信息抽取效果。 [报名链接](https://www.wjx.top/vm/mFhGfwx.aspx?udsid=772552)

- **🔥2024.10.1 添加OCR领域低代码全流程开发能力**:
  * 飞桨低代码开发工具PaddleX，依托于PaddleOCR的先进技术，支持了OCR领域的低代码全流程开发能力：
     * 🎨 [**模型丰富一键调用**](https://paddlepaddle.github.io/PaddleOCR/latest/paddlex/quick_start.html)：将文本图像智能分析、通用OCR、通用版面解析、通用表格识别、公式识别、印章文本识别涉及的**17个模型**整合为6条模型产线，通过极简的**Python API一键调用**，快速体验模型效果。此外，同一套API，也支持图像分类、目标检测、图像分割、时序预测等共计**200+模型**，形成20+单功能模块，方便开发者进行**模型组合**使用。
     * 🚀[**提高效率降低门槛**](https://paddlepaddle.github.io/PaddleOCR/latest/paddlex/overview.html)：提供基于**统一命令**和**图形界面**两种方式，实现模型简洁高效的使用、组合与定制。支持**高性能推理、服务化部署和端侧部署**等多种部署方式。此外，对于各种主流硬件如**英伟达GPU、昆仑芯、昇腾、寒武纪和海光**等，进行模型开发时，都可以**无缝切换**。

  * 支持文档场景信息抽取v3([PP-ChatOCRv3-doc](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction.md))、基于RT-DETR的[高精度版面区域检测模型](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/layout_detection.md)和PicoDet的[高效率版面区域检测模型](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/layout_detection.md)、高精度表格结构识别模型[SLANet_Plus](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/table_structure_recognition.md)、文本图像矫正模型[UVDoc](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/text_image_unwarping.md)、公式识别模型[LatexOCR](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/formula_recognition.md)、基于PP-LCNet的[文档图像方向分类模型](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)
  
- **🔥2024.7 添加 PaddleOCR 算法模型挑战赛冠军方案**：
    - 赛题一：OCR 端到端识别任务冠军方案——[场景文本识别算法-SVTRv2](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)；
    - 赛题二：通用表格识别任务冠军方案——[表格识别算法-SLANet-LCNetV2](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html)。


## 🌟 特性

支持多种 OCR 相关前沿算法，在此基础上打造产业级特色模型PP-、PP-Structure和PP-ChatOCR，并打通数据生产、模型训练、压缩、预测部署全流程。

<div align="center">
    <img src="./docs/images/ppocrv4.png">
</div>

## ⚡ [快速开始](https://paddlepaddle.github.io/PaddleOCR/latest/quick_start.html)

## 🔥 [低代码全流程开发](https://paddlepaddle.github.io/PaddleOCR/latest/paddlex/overview.html)

## 📝 文档

完整文档请移步：[docs](https://paddlepaddle.github.io/PaddleOCR/latest/)

## 📚《动手学 OCR》电子书

- [《动手学 OCR》电子书](https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/blog/ocr_book.html)

## 🎖 贡献者

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>

## ⭐️ Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)

## 许可证书

本项目的发布受 [Apache License Version 2.0](./LICENSE) 许可认证。
