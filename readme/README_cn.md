<div align="center">
  <p>
      <img width="800" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Banner.png" alt="Star-history">
  </p>



<h3>全球领先的 OCR 工具包与文档 AI 引擎</h3>

[English](../README.md) | 简体中文| [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

<!-- icon -->

[![PyPI Downloads](https://static.pepy.tech/badge/paddleocr)](https://pepy.tech/projects/paddleocr)
[![Used by](https://img.shields.io/badge/Used%20by-6k%2B%20repositories-blue)](https://github.com/PaddlePaddle/PaddleOCR/network/dependents)
![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)

[![AI Studio](https://img.shields.io/badge/PaddleOCR-_Offiical_Website-1927BA?logo=data:image/png;base64,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&labelColor=white)](https://www.paddleocr.com)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](../LICENSE)

</div>






**PaddleOCR 以业界领先的精准度，将 PDF 文档和图像转换为结构化、LLM 友好的数据格式（JSON/Markdown）。凭借 70,000+ Stars 的成绩，PaddleOCR 已获得 Dify、RAGFlow、Cherry Studio 等顶级项目的广泛信赖，是构建智能 RAG 和 Agentic 应用的核心基础组件。**


## 🚀 核心特性

### 📄 智能文档解析（面向大模型）
> *为大模型时代将杂乱的文档视觉信息转化为结构化数据。*

* **SOTA 级文档视觉语言模型 (VLM)**: 业界领先的轻量级文档解析视觉语言模型 **PaddleOCR-VL-1.5 (0.9B)**。它在五大"真实场景"中表现卓越：**弯曲、扫描、屏幕拍照、复杂光照及倾斜文档**，并支持以 **Markdown** 和 **JSON** 格式输出结构化结果。
* **版面结构分析**：由 **PP-StructureV3** 驱动，无缝将复杂的 PDF文档 和图像转换为 **Markdown** 或 **JSON** 格式。与 PaddleOCR-VL 系列模型不同，它提供更细粒度的坐标信息，包括表格单元格坐标、文本坐标等。
* **生产级高效能**：以极小的模型体积实现商业级别的准确率。在公开基准测试中超越众多闭源解决方案，同时保持极高的资源利用率，完美适配边缘计算与云端部署。

### 🔍 通用文本识别（场景 OCR）
> *快速、精准的多语言文本检测与识别，被全球开发者广泛采用。*

* **支持 100+ 种语言**：原生支持庞大丰富的全球语种库。**PP-OCRv5** 模型解决方案能够优雅应对多语言混合排版文档（中文、英文、日文、拼音等）。
* **复杂场景支持**：除了标准的文本识别，还支持在各种广泛的环境下进行**自然场景文本检测与识别**，涵盖身份证件、街景、书籍以及工业零部件等。
* **性能提升**：PP-OCRv5 相比前代版本实现了 **13% 的准确率提升**，同时延续了 PaddleOCR 的“极致高效”特性。

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch_cn.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

### 🛠️ 以开发者为中心的生态系统
* **无缝集成**：AI 智能体生态系统的首选——与 **Dify、RAGFlow、Pathway和Cherry Studio** 深度集成。
* **大语言模型数据飞轮**：完整的数据流水线,用于构建高质量数据集,为微调大语言模型提供可持续的"数据引擎"。
* **一键部署**：支持多种硬件后端(NVIDIA GPU、Intel CPU、昆仑芯 XPU 和多种 AI 加速器)。


## 📣 最新动态

### 🔥 [2026.01.29] PaddleOCR v3.4.0 发布：异型文档解析新标杆
* **PaddleOCR-VL-1.5 (SOTA 0.9B VLM)**：
    * **OmniDocBench 94.5%准确率**：超越顶级通用大模型和专业文档解析模型。
    * **现实5大场景文档解析的SOTA性能**：首次引入**PP-DocLayoutV3**算法进行异形框定位，可以解决5种真实场景: 倾斜、弯曲、扫描、光线变化和屏幕拍照。
    * **能力拓展**：增加**印章识别**、**文本行检测/识别**，并扩展至**111种语言**(包括中国的藏文和孟加拉语)。
    * **长文档跨页解析**：支持自动跨页表格合并和分层标题识别。
    * **立即试用**：可在[HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)或[PaddleOCR 官方网站](https://www.paddleocr.com)使用。

<details>
<summary><strong>2025.10.16: PaddleOCR 3.3.0 发布</strong></summary>

- **发布PaddleOCR-VL**：
    - **模型介绍**：
        - **PaddleOCR-VL** 是一款先进、高效的文档解析模型，专为文档中的元素识别设计。其核心组件为 PaddleOCR-VL-0.9B，这是一种紧凑而强大的视觉语言模型（VLM），它由 NaViT 风格的动态分辨率视觉编码器与 ERNIE-4.5-0.3B 语言模型组成，能够实现精准的元素识别。**该模型支持 109 种语言，并在识别复杂元素（如文本、表格、公式和图表）方面表现出色，同时保持极低的资源消耗。通过在广泛使用的公开基准与内部基准上的全面评测，PaddleOCR-VL 在页级级文档解析与元素级识别均达到 SOTA 表现**。它显著优于现有的基于Pipeline方案和文档解析多模态方案以及先进的通用多模态大模型，并具备更快的推理速度。这些优势使其非常适合在真实场景中落地部署。模型已发布至[HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)，欢迎大家下载使用！更多介绍内容请点击[PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html)。

    - **特性**：
        - **紧凑而强大的视觉语言模型架构**：我们提出了一种新的视觉语言模型，专为资源高效的推理而设计，在元素识别方面表现出色。通过将NaViT风格的动态高分辨率视觉编码器与轻量级的ERNIE-4.5-0.3B语言模型结合，我们显著增强了模型的识别能力和解码效率。这种集成在保持高准确率的同时降低了计算需求，使其非常适合高效且实用的文档处理应用。
        - **文档解析的SOTA性能**：PaddleOCR-VL在页面级文档解析和元素级识别中达到了最先进的性能。它显著优于现有的基于流水线的解决方案，并在文档解析中展现出与领先的视觉语言模型（VLMs）竞争的强劲实力。此外，它在识别复杂的文档元素（如文本、表格、公式和图表）方面表现出色，使其适用于包括手写文本和历史文献在内的各种具有挑战性的内容类型。这使得它具有高度的多功能性，适用于广泛的文档类型和场景。
        - **多语言支持**：PaddleOCR-VL支持109种语言，覆盖了主要的全球语言，包括但不限于中文、英文、日文、拉丁文和韩文，以及使用不同文字和结构的语言，如俄语（西里尔字母）、阿拉伯语、印地语（天城文）和泰语。这种广泛的语言覆盖大大增强了我们系统在多语言和全球化文档处理场景中的适用性。

- **发布PP-OCRv5小语种识别模型**：
    - 优化拉丁文识别的准度和广度，新增西里尔文、阿拉伯文、天城文、泰卢固语、泰米尔语等语系，覆盖109种语言文字的识别。模型参数量仅为2M，部分模型精度较上一代提升40%以上。

</details>


<details>
<summary><strong>2025.08.21: PaddleOCR 3.2.0 发布</strong></summary>

- **重要模型新增：**
    - 新增 PP-OCRv5 英文、泰文、希腊文识别模型的训练、推理、部署。**其中 PP-OCRv5 英文模型较 PP-OCRv5 主模型在英文场景提升 11%，泰文识别模型精度 82.68%，希腊文识别模型精度 89.28%。**

- **部署能力升级：**
    - **全面支持飞桨框架 3.1.0 和 3.1.1 版本。**
    - **全面升级 PP-OCRv5 C++ 本地部署方案，支持 Linux、Windows，功能及精度效果与 Python 方案保持一致。**
    - **高性能推理支持 CUDA 12，可使用 Paddle Inference、ONNX Runtime 后端推理。**
    - **高稳定性服务化部署方案全面开源，支持用户根据需求对 Docker 镜像和 SDK 进行定制化修改。**
    - 高稳定性服务化部署方案支持通过手动构造HTTP请求的方式调用，该方式允许客户端代码使用任意编程语言编写。

- **Benchmark支持**：
    - **全部产线支持产线细粒度 benchmark，能够测量产线端到端推理时间以及逐层、逐模块的耗时数据，可用于辅助产线性能分析。可以参考[文档](../docs/version3.x/pipeline_usage/instructions/benchmark.md)来进行性能测试。**
    - **文档中补充各产线常用配置在主流硬件上的关键指标，包括推理耗时和内存占用等，为用户部署提供参考。**

- **Bug修复：**
    - 修复模型训练时训练日志保存失败的问题。
    - 对公式模型的数据增强部分进行了版本兼容性升级，以适应新版本的 albumentations 依赖，并修复了在多进程使用 tokenizers 依赖包时出现的死锁警告。
    - 修复 PP-StructureV3 配置文件中的 `use_chart_parsing` 等开关行为与其他产线不统一的问题。

- **其他升级：**
    - **分离必要依赖与可选依赖。使用基础文字识别功能时，仅需安装少量核心依赖；若需文档解析、信息抽取等功能，用户可按需选择安装额外依赖。**
    - **支持 Windows 用户使用英伟达 50 系显卡，可根据 [安装文档](../docs/version3.x/installation.md) 安装对应版本的 paddle 框架。**
    - **PP-OCR 系列模型支持返回单文字坐标。**
    - 模型新增 AIStudio、ModelScope 等下载源。可指定相关下载源下载对应的模型。
    - 支持图表转表 PP-Chart2Table 单功能模块推理能力。
    - 优化部分使用文档中的描述，提升易用性。
</details>


[历史日志](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>


## 🚀 快速开始

### 步骤 1: 在线体验
PaddleOCR官方网站提供交互式**体验中心**和**APIs**——无需设置，一键体验。

👉 [访问官方网站](https://www.paddleocr.com)

### 步骤 2: 本地部署
对于本地使用,请根据您的需求参考以下文档:

- **PP-OCR系列**：查看[PP-OCR文档](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html)
- **PaddleOCR-VL系列**：查看[PaddleOCR-VL文档](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html)
- **PP-StructureV3**：查看[PP-StructureV3文档](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- **更多能力**：查看[更多能力文档](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/pipeline_overview.html)


## 🧩 更多功能

- 将模型转换为ONNX格式: [获取ONNX模型](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/obtaining_onnx_models.html)。
- 使用OpenVINO、ONNX Runtime、TensorRT等引擎加速推理,或使用ONNX格式模型进行推理: [高性能推理](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/high_performance_inference.html)。
- 使用多GPU和多进程加速推理: [流水线并行推理](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/instructions/parallel_inference.html)。
- 将PaddleOCR集成到C++、C#、Java等语言编写的应用程序中: [服务化部署](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/serving.html)。

## 🔄 执行结果快速预览

### PP-OCRv5

<div align="center">
  <p>
       <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-OCRv5_demo.gif" alt="PP-OCRv5 Demo">
  </p>
</div>



### PP-StructureV3

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-StructureV3_demo.gif" alt="PP-StructureV3 Demo">
  </p>
</div>

### PaddleOCR-VL

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PaddleOCR-VL_demo.gif" alt="PP-StructureV3 Demo">
  </p>
</div>


## ✨ 保持关注

⭐ **收藏本仓库，持续关注最新动态与版本发布，包括强大的 OCR 及文档解析等新功能特性。** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr2.en.gif" alt="Star-Project">
  </p>
</div>


## 👩‍👩‍👧‍👦 社区

<div align="center">

| PaddlePaddle 微信公众号 |  加入技术讨论群 |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |
</div>


## 😃 使用 PaddleOCR 的优秀项目

<div align="center">

PaddleOCR 的发展离不开社区贡献！💗衷心感谢所有开发者、合作伙伴与贡献者！
| 项目名称 | 简介 |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|基于RAG的AI工作流引擎|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|用于流处理、实时分析、LLM流水线和RAG的Python ETL框架|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|多类型文档转换Markdown工具|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|开源批量离线OCR软件|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|一个支持多个LLM提供商的桌面客户端|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |基于纯视觉的GUI智能体屏幕解析工具|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |基于任意内容的问答系统|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|高效复杂PDF文档提取工具包|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |屏幕实时翻译工具|
| [更多项目](../awesome_projects.md) |  [更多基于PaddleOCR的项目](../awesome_projects.md) |
</div>


## 👩‍👩‍👧‍👦 贡献者

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>
</div>

## 🌟 Star历史

<div align="center">
  <p>
      <img width="800" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="Star-history">
  </p>
</div>


## 📄 许可证
本项目采用[Apache 2.0许可证](LICENSE)发布。

## 🎓 引用

```bibtex
@misc{cui2025paddleocr30technicalreport,
      title={PaddleOCR 3.0 Technical Report},
      author={Cheng Cui and Ting Sun and Manhui Lin and Tingquan Gao and Yubo Zhang and Jiaxuan Liu and Xueqing Wang and Zelun Zhang and Changda Zhou and Hongen Liu and Yue Zhang and Wenyu Lv and Kui Huang and Yichao Zhang and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2507.05595},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05595},
}

@misc{cui2025paddleocrvlboostingmultilingualdocument,
      title={PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model},
      author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Handong Zheng and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2510.14528},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14528},
}

@misc{cui2026paddleocrvl15multitask09bvlm,
      title={PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing},
      author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2026},
      eprint={2601.21957},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.21957},
}
```
