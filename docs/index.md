---
comments: true
hide:
  - navigation
  - toc
---

<div align="center">
  <p>
    <a href="https://paddlepaddle.github.io/PaddleOCR/latest/index.html" target="_blank">
      <img width="100%" src="./images/Banner_cn.png" alt="PaddleOCR Banner"></a>
  </p>
</div>


PaddleOCR自发布以来凭借学术前沿算法和产业落地实践，受到了产学研各方的喜爱，并被广泛应用于众多知名开源项目，例如：Umi-OCR、OmniParser、MinerU、RAGFlow等，已成为广大开发者心中的开源OCR领域的首选工具。2025年5月20日，飞桨团队发布**PaddleOCR 3.0**，全面适配[飞桨框架3.0](https://github.com/PaddlePaddle/Paddle)正式版，进一步**提升文字识别精度**，支持**多文字类型识别**和**手写体识别**，满足大模型应用对**复杂文档高精度解析**的旺盛需求，结合**文心大模型4.5**显著提升关键信息抽取精度，并新增**对昆仑芯、昇腾等国产硬件**的支持。

**2025 年 10 月 16 日，PaddleOCR 开源了先进、高效的文档解析模型 PaddleOCR-VL**，其核心组件为 PaddleOCR-VL-0.9B，这是一种紧凑而强大的视觉语言模型（VLM），它由 NaViT 风格的动态分辨率视觉编码器与 ERNIE-4.5-0.3B 语言模型组成，能够实现精准的元素识别。该模型支持 109 种语言，并在识别复杂元素（如文本、表格、公式和图表）方面表现出色，同时保持极低的资源消耗。通过在广泛使用的公开基准与内部基准上的全面评测，PaddleOCR-VL 在页级级文档解析与元素级识别均达到 SOTA 表现。它显著优于现有的基于 Pipeline 方案和文档解析多模态方案以及先进的通用多模态大模型，并具备更快的推理速度。这些优势使其非常适合在真实场景中落地部署。


**PaddleOCR 3.x 核心特色能力：**


- **PaddleOCR-VL - 通过 0.9B 超紧凑视觉语言模型增强多语种文档解析**  
  **面向文档解析的 SOTA 且资源高效的模型**, 支持 109 种语言，在复杂元素（如文本、表格、公式和图表）识别方面表现出色，同时资源消耗极低。

- **PP-OCRv5 — 全场景文字识别**  
  **单模型支持五种文字类型**（简中、繁中、英文、日文及拼音），精度提升**13个百分点**。解决多语言混合文档的识别难题。

- **PP-StructureV3 — 复杂文档解析**  
  将复杂PDF和文档图像智能转换为保留**原始结构的Markdown文件和JSON**文件，在公开评测中**领先**众多商业方案。**完美保持文档版式和层次结构**。

- **PP-ChatOCRv4 — 智能信息抽取**  
  原生集成ERNIE 4.5，从海量文档中**精准提取关键信息**，精度较上一代提升15个百分点。让文档"**听懂**"您的问题并给出准确答案。

> [!TIP]
> 
> 2025 年 10 月 24 日，PaddleOCR 官网 Beta 版现上线，支持更便捷的在线体验和大批量 PDF 文件解析，并提供免费 API 及 MCP 服务。更多详情请参见 [PaddleOCR 官网](https://www.paddleocr.com)。

PaddleOCR 3.0除了提供优秀的模型库外，还提供好学易用的工具，覆盖模型训练、推理和服务化部署，方便开发者快速落地AI应用。
<div align="center">
  <p>
    <a href="https://paddlepaddle.github.io/PaddleOCR/latest/index.html" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch_cn.jpg" alt="PaddleOCR Architecture"></a>
  </p>
</div>

您可直接[快速开始](./quick_start.md)，或查阅完整的 [PaddleOCR 文档](https://paddlepaddle.github.io/PaddleOCR/main/index.html)，或通过 [Github Issues](https://github.com/PaddlePaddle/PaddleOCR/issues) 获取支持，或在 [AIStudio 课程平台](https://aistudio.baidu.com/course/introduce/25207) 探索我们的 OCR 课程。

**特别说明**：PaddleOCR 3.x 引入了多项重要的接口变动，**基于 PaddleOCR 2.x 编写的旧代码很可能无法使用 PaddleOCR 3.x 运行**。请确保您阅读的文档与实际使用的 PaddleOCR 版本匹配。[此文档](./update/upgrade_notes.md) 阐述了升级原因及 PaddleOCR 2.x 到 PaddleOCR 3.x 的主要变更。

## 🔄 快速一览运行效果

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


## 👩‍👩‍👧‍👦 开发者社区
* [AI加码，PaddleOCR最佳实践场景项目征集等你参与！](https://aistudio.baidu.com/activitydetail/1503019405)   
📅 **August 5, 2025 – October 30, 2025**. 分享你的场景化 PaddleOCR 应用项目，与全球开发者共创精彩！
* 👫 加入 [PaddlePaddle 开发者社区](https://github.com/PaddlePaddle/community)，与全球开发者、研究人员互动交流
* 🎓 通过 AI Studio 的 [技术研讨会](https://aistudio.baidu.com/learn/center) 学习前沿技术
* 🏆 参与 [黑客马拉松](https://aistudio.baidu.com/competition) 展示才能，赢取奖励
* 📣 关注 [微信公众号](https://mp.weixin.qq.com/s/vYj1ZDcAfJ1lu_DzlOKgtQ) 获取最新动态
