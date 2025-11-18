---
comments: true
hide:
  - navigation
  - toc
---

<div align="center">
  <p>
    <a href="https://paddlepaddle.github.io/PaddleOCR/latest/index.html" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/docs/images/Banner.png" alt="PaddleOCR Banner">
    </a>
  </p>
</div>

Since its initial release, PaddleOCR has gained widespread acclaim across academia, industry, and research communities, thanks to its cutting-edge algorithms and proven performance in real-world applications. It’s already powering popular open-source projects like Umi-OCR, OmniParser, MinerU, and RAGFlow, making it the go-to OCR toolkit for developers worldwide.

On May 20, 2025, the PaddlePaddle team unveiled PaddleOCR 3.0, fully compatible with the official release of the [PaddlePaddle 3.0](https://github.com/PaddlePaddle/Paddle) framework. This update further **boosts text-recognition accuracy**, adds support for **multiple text-type recognition** and **handwriting recognition**, and meets the growing demand from large-model applications for **high-precision parsing of complex documents**. When combined with the **ERNIE 4.5**, it significantly enhances key-information extraction accuracy. PaddleOCR 3.0 also introduces support for domestic hardware platforms such as **KUNLUNXIN** and **Ascend**.


**Major Features in PaddleOCR 3.x:**

- **PaddleOCR-VL - Multilingual Document Parsing via a 0.9B VLM**  
  **The SOTA and resource-efficient model tailored for document parsing**, that supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption.

- **PP-OCRv5 — Universal Scene Text Recognition**  
  **Single model supports five text types** (Simplified Chinese, Traditional Chinese, English, Japanese, and Pinyin) with **13% accuracy improvement**. Solves multilingual mixed document recognition challenges.

- **PP-StructureV3 — Complex Document Parsing**  
  Intelligently converts complex PDFs and document images into **Markdown and JSON files that preserve original structure**. **Outperforms** numerous commercial solutions in public benchmarks. **Perfectly maintains document layout and hierarchical structure**.

- **PP-ChatOCRv4 — Intelligent Information Extraction**  
  Natively integrates ERNIE 4.5 to **precisely extract key information** from massive documents, with 15% accuracy improvement over previous generation. Makes documents "**understand**" your questions and provide accurate answers.

> [!TIP]
> 
> On October 24, 2025, the PaddleOCR official website Beta version was launched, offering a more convenient online experience and large-scale PDF file parsing, as well as free API and MCP services. For more details, please visit the [PaddleOCR official website](https://www.paddleocr.com).

In addition to providing an outstanding model library, PaddleOCR 3.0 also offers user-friendly tools covering model training, inference, and service deployment, so developers can rapidly bring AI applications to production.

<div align="center">
  <p>
    <a href="https://paddlepaddle.github.io/PaddleOCR/latest/index.html" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture"></a>
  </p>
</div>

You can [Quick Start](./quick_start.en.md) directly, find comprehensive documentation in the [PaddleOCR Docs](https://paddlepaddle.github.io/PaddleOCR/main/index.html), get support via [Github Issues](https://github.com/PaddlePaddle/PaddleOCR/issues), and explore our OCR courses on [OCR courses on AIStudio](https://aistudio.baidu.com/course/introduce/25207).

**Special Note**: PaddleOCR 3.x introduces several significant interface changes. **Old code written based on PaddleOCR 2.x is likely incompatible with PaddleOCR 3.x**. Please ensure that the documentation you are reading matches the version of PaddleOCR you are using. [This document](./update/upgrade_notes.en.md) explains the reasons for the upgrade and the major changes from PaddleOCR 2.x to 3.x.

## 🔄 Quick Overview of Execution Results


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

## 👩‍👩‍👧‍👦 Community
* The [PaddleOCR Best Practice Projects](https://aistudio.baidu.com/activitydetail/1503019405) call for submissions is now open!
📅 **August 5, 2025 – October 30, 2025**. Share your scenario-based PaddleOCR applications and shine in the global developer community! 
* 👫 Join the [PaddlePaddle Community](https://github.com/PaddlePaddle/community), where you can engage with [paddlepaddle developers](https://www.paddlepaddle.org.cn/developercommunity), researchers, and enthusiasts from around the world.
* 🎓 Learn from experts through workshops, tutorials, and Q&A sessions [hosted by the AI Studio](https://aistudio.baidu.com/learn/center).
* 🏆 Participate in [hackathons, challenges, and competitions](https://aistudio.baidu.com/competition) to showcase your skills and win exciting prizes.
* 📣 Stay updated with the latest news, announcements, and events by following our [Twitter](https://x.com/PaddlePaddle) and [WeChat](https://mp.weixin.qq.com/s/vYj1ZDcAfJ1lu_DzlOKgtQ)).
