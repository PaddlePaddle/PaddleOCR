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

On January 29, 2026, PaddleOCR open-sourced the advanced and efficient document parsing model PaddleOCR-VL-1.5. PaddleOCR-VL-1.5 is a new iterative version of the PaddleOCR-VL series. Based on comprehensive optimization of the core capabilities of version 1.0, the model achieves 94.5% accuracy on the authoritative document parsing benchmark OmniDocBench v1.5, surpassing top global general-purpose large models and document parsing–specific models. PaddleOCR-VL-1.5 innovatively supports irregular-shaped bounding box localization of document elements, enabling excellent performance in real-world application scenarios such as scanning, skew, warping, screen-photography, and complex illumination, achieving comprehensive SOTA performance. In addition, the model further integrates seal recognition and spotting tasks, with key metrics continuing to lead mainstream models.

**On May 28, 2026, PaddleOCR open-sourced PaddleOCR-VL-1.6, the latest version in the PaddleOCR-VL series.** With an accuracy of 96.3%, PaddleOCR-VL-1.6 once again set a new benchmark on OmniDocBench v1.6, while also achieving new state-of-the-art (SOTA) results on OmniDocBench v1.5 and Real5-OmniDocBench. It delivers industry-leading performance in text, formula, and table recognition across both open-source and proprietary solutions. In addition, the model shows substantial improvements in ancient document and rare character recognition, as well as significantly enhanced capabilities in multiple scenarios such as seal recognition, spotting, and chart understanding. PaddleOCR-VL-1.6 further upgrades the VLM component (PaddleOCR-VL-1.6-0.9B) on top of PaddleOCR-VL-1.5 and continues to use PP-DocLayoutV3 for layout analysis. The model architecture remains fully consistent with PaddleOCR-VL-1.5, enabling seamless migration at zero cost.

**On June 11, 2026, PaddleOCR released PP-OCRv6, the next-generation universal OCR system.** Built on the newly designed PPLCNetV4 unified backbone, PP-OCRv6 offers tiny/small/medium tiers (1.5M–34.5M parameters). In terms of accuracy, the medium tier achieves +5.1% recognition and +4.6% detection over PP-OCRv5_server, surpassing mainstream Vision-Language Models with only 34.5M parameters. For multilingual support, a single model covers 50 languages (Chinese, English, Japanese, and 46 Latin-script languages) without switching models. PP-OCRv6 significantly improves recognition in specialized scenarios such as digital displays, dot-matrix characters, tire prints, and industrial text that are traditionally underserved by VLMs. In terms of speed, PP-OCRv6_medium achieves 5.2× speedup over PP-OCRv5_server on Intel Xeon CPU with OpenVINO (1.40s vs 7.30s), the tiny tier reaches 6.1× speedup on Apple M4 (0.96s vs 5.82s), and only 0.13s on A100 GPU. See [PP-OCRv6 Technical Documentation](./version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md) for details.

You can use it online on the [PaddleOCR official website](https://www.paddleocr.com) or call the model API.


**Major Features in PaddleOCR 3.x:**

- **PaddleOCR-VL - Multilingual Document Parsing via a 0.9B VLM**  
  **The SOTA and resource-efficient model tailored for document parsing**, that supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption.

- **PP-OCRv6 — Universal Multi-Language Text Recognition**
  **Single model supports 50 languages** (Chinese, English, Japanese, and 46 Latin-script languages), with **+4.6% detection** and **+5.1% recognition** accuracy over PP-OCRv5, surpassing mainstream Vision-Language Models. Medium tier achieves 5.2× CPU speedup end-to-end, covering specialized scenarios like digital displays, dot-matrix, and industrial characters. Three tiers (tiny/small/medium, 1.5M–34.5M params) for edge-to-server deployment.

- **PP-StructureV3 — Complex Document Parsing**  
  Intelligently converts complex PDFs and document images into **Markdown and JSON files that preserve original structure**. **Outperforms** numerous commercial solutions in public benchmarks. **Perfectly maintains document layout and hierarchical structure**.

- **PP-ChatOCRv4 — Intelligent Information Extraction**  
  Natively integrates ERNIE 4.5 to **precisely extract key information** from massive documents, with 15% accuracy improvement over previous generation. Makes documents "**understand**" your questions and provide accurate answers.

> 💡 Tips
> 
> PaddleOCR's free API now supports up to **20,000** pages of document parsing per day, enabling large-scale PDF file processing, along with MCP and Skills services. For more details, please visit [PaddleOCR Official Website](https://www.paddleocr.com).

In addition to its strong model library, PaddleOCR 3.0 also provides easy-to-use tools covering model training, inference, and serving, helping developers bring AI applications into production more efficiently.

**In addition, PaddleOCR provides official [Agent Skills](./version3.x/integrations/skills.en.md) for invoking text recognition, document parsing, and related capabilities in Skills-enabled AI apps.**

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

## 👩‍👩‍👧‍👦 PaddleOCR OCEAN Ecosystem Alliance

The lead in single-point technology is just the beginning — the prosperity of the ecosystem is where long-term value truly lies. To better serve global developers and industry scenarios with OCR and document intelligence technologies, we are officially launching the **PaddleOCR OCEAN Ecosystem Alliance**.

The alliance name **OCEAN** embodies five core pillars:

* **O**pen Source – Open source as the foundation
* **C**ommunity – Community-driven
* **E**cosystem – Shared ecosystem success
* **A**pplication – Real-world application
* **N**etwork – Networked collaboration

**Positioning**: An ecosystem alliance centered on open-source co-building, open to global upstream and downstream partners in OCR and document intelligence. The alliance involves no commercial exclusivity and does not interfere with partners' independent business decisions. It focuses on technical collaboration, community engagement, and mutual influence expansion. Guided by the core principles of openness, symbiosis, and shared success, it brings together developers, platform providers, and application builders to jointly advance the full-chain application of OCR technology and ecosystem prosperity. The alliance is committed to achieving dual growth in full-chain application scale and the number of derivative projects, enabling developers and users worldwide to share in the dividends of OCR technology advancement.

**Join Us: Walk into the Deep End with Like-Minded Partners**

The PaddleOCR OCEAN Ecosystem Alliance is open to global partners across the OCR and document intelligence value chain. We firmly believe: **the value of an ecosystem lies not in quantity, but in quality.**

We look forward to welcoming partners who:

* **Genuinely embrace the open-source spirit** and are willing to co-build and share with an open mindset
* **Have the willingness and capability to contribute consistently** — whether through code, use case scenarios, or platform integration
* **Are committed to growing together with the alliance** — not chasing short-term traffic, but cultivating long-term value

**The alliance is not a hall of fame — it is a rallying call for those who take action.**

We will carefully evaluate every application, giving priority to partners who have already taken action within the PaddleOCR ecosystem or have a clear co-building plan in place. We do not pursue being "large and all-encompassing." Instead, we seek to work hand in hand with truly like-minded organizations and individuals, diving deep into the frontiers of OCR together.

If you resonate with the above vision, we welcome you to reach out through the following channel:

* Send an email to paddleocr@baidu.com with a brief introduction of your collaboration with PaddleOCR or your co-building plan.
