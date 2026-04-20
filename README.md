
<div align="center">
  <p>
      <img width="800" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Banner.png" alt="Star-history">
  </p>



<h3>Global Leading OCR Toolkit & Document AI Engine</h3>

English | [简体中文](./readme/README_cn.md) | [繁體中文](./readme/README_tcn.md) | [日本語](./readme/README_ja.md) | [한국어](./readme/README_ko.md) | [Français](./readme/README_fr.md) | [Русский](./readme/README_ru.md) | [Español](./readme/README_es.md) | [العربية](./readme/README_ar.md)

<!-- icon -->

[![PyPI Downloads](https://static.pepy.tech/badge/paddleocr)](https://pepy.tech/projects/paddleocr)
[![Used by](https://img.shields.io/badge/Used%20by-6k%2B%20repositories-blue)](https://github.com/PaddlePaddle/PaddleOCR/network/dependents)
![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)

[![AI Studio](https://img.shields.io/badge/PaddleOCR-_Offiical_Website-1927BA?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAMAAADDpiTIAAAABlBMVEU2P+X///+1KuUwAAAHKklEQVR42u3dS5bjOAwEwALvf2fMavZum6IAImI7b2yYSqU+1Zb//gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADKCR/+fzly7rD92yVg69xh8zeLwOa5w+ZvFYHtc4ft3ykB++cOm79PAp6YO2z/Ngl4ZO5l+9+yT4QAvLqS748VF33Ylzdvzpl72f6z53YIGJ6SZdPeNHcIwOycaADdLgCSIgAIgCOAACAAykIAEAAEAAFAABCAT+WQuQVgeBqXhXQIQAAYegowLQBpbg3gZGFyAC6vgBQAMREA2/YfDPxyaDQNyTNz+3Zwn5J4ZG7PB2h0kHhi7plPCImmJwkPzO0RMa3OET0i5uGlzHFze0xcu0vE2Dq3J4U2vEPgSaHbFzPNDQAAAAAAAMBNovdw+cP/ny+uaf7w/+eYADy8kE+F4Offdjn6zZXhAXgiA78G4MNNsmnu1Xr7b3mbOL8T5Ja5bw/A35EC2LiWpzt1y9jRugBy30fLg3NvHPvnuZcC2NsCUXA/aRmA89V07Fwgt37uH8deCmBr6N44pP4UgaUATpdA7v/cMbIB8okliY65/SW5HhJ1ehPmM+8edwXgpbu4R88FayR32Y/P7oZZbOx13/Zr//ZHx27bAPnkFoyewYlbAhD3TvBobr95gaUAtr1EdNx1lgI4OcTTuR3z6+FZMEDRcu9ZCuDgGCdyGxMa4EgBRMvcjrkM7NgBZw5c0TwAUWUhZwRXA2xaya65Xa3jO2qYZ8bu2AD5w38tG5V8aZpoGN6Tz0bOfa9bceyWAciTO0jWyO1Tc5cLwJmF/JfPnXVyu3/slgHIg1n79O2O5fZv+1cHV7sC2HYqmUdHysNzX3sVkMcjUK5Gc+dMs28E5bGtm0V3gloBOP9vgZv+4sYn3RUaYFMCol5uN77g6lUApc8pWs69Zn7snS9Z9Q8G0S0AUTVUUTG3A54R1KSvo/diLAv5fKzynZeN6xogC75u93+AtBTA47OlAFSv6qY/vp3DAjD8iv2ZdFYJwKynMhTK1rInPfzaxW81LnvSgFP9KxrATaCLA3DxHpbFX31ZyNm5XRZyXG5bNkAWfP0rcrsUwOgC6NIAzgBcBiqAWwPgLrAGuGBP6jr2sifdfiJ6QQM4Bbw4AK4B3129ZSFn53ZZyA/GyFty27IBFMDFAXAG8PbyLQv5xULGPRl0K3h2AbwcgCZPhs+LD1zLnjS6AN4NwMU/DVFh7LyhASreTbvqrxdr/J4XT4Swz4FrTS+AGJ7bNbwAYkxuWzZAVljHrJfbjb9wviYXwFO/FJ8Vli4vaICsEMFyBbA3tmtsAUS0zG1c/bj4YwsZH2/+Whd0+1Nb+S7IE2sfPw4RL0XmsR8Nqvz7qFngmPHF34EqjP15AAofAkosZKPC/K6FVoeP02Ehi540NG6AK/4pYP3cLgVwXwHkDQ1QcSGb/uF4WwCmfX8u/+4vgLINcMUlQIfcLgXwXAF0+BGkpQDuuJx7/hwgpu//cWVuO3wxJOz/z8297vgYBwaIO3O7Kn+c194578ltywbIgu8fl+Z2lS+APvnLjnOv8hsgSqxjgwL4Ln9LAezaj98tgPzy7ZcC+GQzxrWxXQpgx370dm6/H7v6jaBoso5dY1swAFlwHWvfBf5pxVa93fCtdx64+1dsgCy4joWvAfPX9VoKYMs6Zse9/8Mlvv7LILlhAfKFFdsSutJXAdFkL3qlADJPrXFcXAC5KYaH586jO9mtAch9S3T0GQJ726ZWAE49kjP3rlDJuetdaL/1zeqZY9c7CRz7s0wCUPxienQBnAuAAtAAlxaAAAxfyBQABSAACkAAFIAAKAABUAACMEkKwL170oh7V8ueNLoAjgTAXWAN4BRwcABcA2oABTA4AApAAyiAwQFQABpAAQwOgALQADMWUgCuEmNyu15fSIY3gFPAiwPgFFADKIDBAVAAGkABCIACmBqAUAAaQAHMDUCMWkgBuMWw3K43F5LhDeAU8OIAuAmkARTA4AAoAA2gAARAAUwNgLvAGkABDA6Au8AaoKOJuV0vLSTDG8Ap4MUBcBNIAyiAwQFQABpAAQwOgALQAApAABTA1AC4C6wBOhqb23V+IRneAE4BLw6Aa0ANoAAGB0ABaAAFMDgACkADKAABUABTA+AusAboKATAQs4trjV+IYcfuJYCcA6gAATAQk69dFkKQANYyLkFcLIBFIDLQAVwawDsSRrAEWBwAJwCagAFMDgACkADKIDBAVAAGkABCIACmBoAzwXWAApgcADsSRrg0iNACoACEADXgAIwdCFTACykALgGFIAfl0kBAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPBv/gN+IH8U6YveYgAAAABJRU5ErkJggg==&labelColor=white)](https://www.paddleocr.com)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](../LICENSE)

</div>






**PaddleOCR converts PDF documents and images into structured, LLM-ready data (JSON/Markdown) with industry-leading accuracy. With 70k+ Stars and trusted by top-tier projects like Dify, RAGFlow, and Cherry Studio, PaddleOCR is the bedrock for building intelligent RAG and Agentic applications.**


## 🚀 Key Features

### 📄 Intelligent Document Parsing (LLM-Ready)
> *Transforming messy visuals into structured data for the LLM era.*

* **SOTA Document VLM**: Featuring **PaddleOCR-VL-1.5 (0.9B)**, the industry's leading lightweight vision-language model for document parsing. It excels in parsing complex documents across 5 major "Real-World" challenges: **Warping, Scanning, Screen Photography, Illumination, and Skewed documents**, with structured outputs in **Markdown** and **JSON** formats.
* **Structure-Aware Conversion**: Powered by **PP-StructureV3**, seamlessly convert complex PDFs and images into **Markdown** or **JSON**. Unlike the PaddleOCR-VL series models, it provides more fine-grained coordinate information, including table cell coordinates, text coordinates, and more.
* **Production-Ready Efficiency**: Achieve commercial-grade accuracy with an ultra-small footprint. Outperforms numerous closed-source solutions in public benchmarks while remaining resource-efficient for edge/cloud deployment.

### 🔍 Universal Text Recognition (Scene OCR)
> *The global gold standard for high-speed, multilingual text spotting.*

* **100+ Languages Supported**: Native recognition for a vast global library. Our **PP-OCRv5** single-model solution elegantly handles multilingual mixed documents (Chinese, English, Japanese, Pinyin, etc.).
* **Complex Element Mastery**: Beyond standard text recognition, we support **natural scene text spotting** across a wide range of environments, including IDs, street views, books, and industrial components
* **Performance Leap**: PP-OCRv5 delivers a **13% accuracy boost** over previous versions, maintaining the "Extreme Efficiency" that PaddleOCR is famous for.

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

### 🛠️ Developer-Centric Ecosystem
* **Seamless Integration**: The premier choice for the AI Agent ecosystem—deeply integrated with **Dify, RAGFlow, Pathway, and Cherry Studio**.
* **LLM Data Flywheel**: A complete pipeline to build high-quality datasets, providing a sustainable "Data Engine" for fine-tuning Large Language Models.
* **One-Click Deployment**: Supports various hardware backends (NVIDIA GPU, Intel CPU, Kunlunxin XPU, and diverse AI Accelerators).


## 📣 Recent updates

### 🔥 PaddleOCR v3.5.0 Released: More Flexible Inference and Richer Document Output
* **Flexible inference backends**: Seamlessly switch between Paddle static graph, Paddle dynamic graph, or Transformers. PaddleOCR is now deeply integrated with the Hugging Face ecosystem, and 20 major models support Transformers as the inference backend.
* **Office documents to Markdown**: Convert common document formats such as Word, Excel, and PowerPoint into Markdown.
* **DOCX export for parsed results**: The `PaddleOCR-VL` series, `PP-StructureV3`, and `PP-DocTranslation` now support exporting parsed results to DOCX for convenient viewing and editing in Microsoft Word.
* **Official browser inference SDK**: Released `PaddleOCR.js`, the official browser inference SDK that supports running `PP-OCRv5` directly in the browser.

<details>
<summary><strong>2026.01.29: Release of PaddleOCR 3.4.0</strong></summary>

* **PaddleOCR-VL-1.5 (SOTA 0.9B VLM)**: Our latest flagship model for document parsing is now live!
    * **94.5% Accuracy on OmniDocBench**: Surpassing top-tier general large models and specialized document parsers.
    * **Real-World Robustness**: First to introduce the **PP-DocLayoutV3** algorithm for irregular shape positioning, mastering 5 tough scenarios: *Skew, Warping, Scanning, Illumination, and Screen Photography*.
    * **Capability Expansion**: Now supports **Seal Recognition**, **Text Spotting**, and expands to **111 languages** (including China’s Tibetan script and Bengali).
    * **Long Document Mastery**: Supports automatic cross-page table merging and hierarchical heading identification.
    * **Try it now**: Available on [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) or our [Official Website](https://www.paddleocr.com).

</details>

<details>
<summary><strong>2025.10.16: Release of PaddleOCR 3.3.0</strong></summary>

- Released PaddleOCR-VL:
    - **Model Introduction**:
        - **PaddleOCR-VL** is a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. **This innovative model efficiently supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption**. Through comprehensive evaluations on widely used public benchmarks and in-house benchmarks, PaddleOCR-VL achieves SOTA performance in both page-level document parsing and element-level recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference speeds. These strengths make it highly suitable for practical deployment in real-world scenarios. The model has been released on [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL). Everyone is welcome to download and use it! More introduction information can be found in [PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html).

    - **Core Features**:
        - **Compact yet Powerful VLM Architecture**: We present a novel vision-language model that is specifically designed for resource-efficient inference, achieving outstanding performance in element recognition. By integrating a NaViT-style dynamic high-resolution visual encoder with the lightweight ERNIE-4.5-0.3B language model, we significantly enhance the model’s recognition capabilities and decoding efficiency. This integration maintains high accuracy while reducing computational demands, making it well-suited for efficient and practical document processing applications.
        - **SOTA Performance on Document Parsing**: PaddleOCR-VL achieves state-of-the-art performance in both page-level document parsing and element-level recognition. It significantly outperforms existing pipeline-based solutions and exhibiting strong competitiveness against leading vision-language models (VLMs) in document parsing. Moreover, it excels in recognizing complex document elements, such as text, tables, formulas, and charts, making it suitable for a wide range of challenging content types, including handwritten text and historical documents. This makes it highly versatile and suitable for a wide range of document types and scenarios.
        - **Multilingual Support**: PaddleOCR-VL Supports 109 languages, covering major global languages, including but not limited to Chinese, English, Japanese, Latin, and Korean, as well as languages with different scripts and structures, such as Russian (Cyrillic script), Arabic, Hindi (Devanagari script), and Thai. This broad language coverage substantially enhances the applicability of our system to multilingual and globalized document processing scenarios.

- Released PP-OCRv5 Multilingual Recognition Model:
    - Improved the accuracy and coverage of Latin script recognition; added support for Cyrillic, Arabic, Devanagari, Telugu, Tamil, and other language systems, covering recognition of 109 languages. The model has only 2M parameters, and the accuracy of some models has increased by over 40% compared to the previous generation.

</details>


<details>
<summary><strong>2025.08.21: Release of PaddleOCR 3.2.0</strong></summary>

- **Significant Model Additions:**
    - Introduced training, inference, and deployment for PP-OCRv5 recognition models in English, Thai, and Greek. **The PP-OCRv5 English model delivers an 11% improvement in English scenarios compared to the main PP-OCRv5 model, with the Thai and Greek recognition models achieving accuracies of 82.68% and 89.28%, respectively.**

- **Deployment Capability Upgrades:**
    - **Full support for PaddlePaddle framework versions 3.1.0 and 3.1.1.**
    - **Comprehensive upgrade of the PP-OCRv5 C++ local deployment solution, now supporting both Linux and Windows, with feature parity and identical accuracy to the Python implementation.**
    - **High-performance inference now supports CUDA 12, and inference can be performed using either the Paddle Inference or ONNX Runtime backends.**
    - **The high-stability service-oriented deployment solution is now fully open-sourced, allowing users to customize Docker images and SDKs as required.**
    - The high-stability service-oriented deployment solution also supports invocation via manually constructed HTTP requests, enabling client-side code development in any programming language.

- **Benchmark Support:**
    - **All production lines now support fine-grained benchmarking, enabling measurement of end-to-end inference time as well as per-layer and per-module latency data to assist with performance analysis. [Here's](docs/version3.x/pipeline_usage/instructions/benchmark.en.md) how to set up and use the benchmark feature.**
    - **Documentation has been updated to include key metrics for commonly used configurations on mainstream hardware, such as inference latency and memory usage, providing deployment references for users.**

- **Bug Fixes:**
    - Resolved the issue of failed log saving during model training.
    - Upgraded the data augmentation component for formula models for compatibility with newer versions of the albumentations dependency, and fixed deadlock warnings when using the tokenizers package in multi-process scenarios.
    - Fixed inconsistencies in switch behaviors (e.g., `use_chart_parsing`) in the PP-StructureV3 configuration files compared to other pipelines.

- **Other Enhancements:**
    - **Separated core and optional dependencies. Only minimal core dependencies are required for basic text recognition; additional dependencies for document parsing and information extraction can be installed as needed.**
    - **Enabled support for NVIDIA RTX 50 series graphics cards on Windows; users can refer to the [installation guide](docs/version3.x/installation.en.md) for the corresponding PaddlePaddle framework versions.**
    - **PP-OCR series models now support returning single-character coordinates.**
    - Added AIStudio, ModelScope, and other model download sources, allowing users to specify the source for model downloads.
    - Added support for chart-to-table conversion via the PP-Chart2Table module.
    - Optimized documentation descriptions to improve usability.
</details>


[History Log](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>


## 🚀 Quick Start

### Step 1: Try Online
PaddleOCR official website provides interactive **Experience Center** and **APIs**—no setup required, just one click to experience.

👉 [Visit Official Website](https://www.paddleocr.com)

### Step 2: Local Deployment
For local usage, please refer to the following documentation based on your needs:

- **PP-OCR Series**: See [PP-OCR Documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)
- **PaddleOCR-VL Series**: See [PaddleOCR-VL Documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- **PP-StructureV3**: See [PP-StructureV3 Documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html)
- **More Capabilities**: See [More Capabilities Documentation](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/pipeline_overview.html)


## 🧩 More Features

- Convert models to ONNX format: [Obtaining ONNX Models](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- Accelerate inference using engines like OpenVINO, ONNX Runtime, TensorRT, or perform inference using ONNX format models: [High-Performance Inference](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- Accelerate inference using multi-GPU and multi-process: [Parallel Inference for Pipelines](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- Integrate PaddleOCR into applications written in C++, C#, Java, etc.: [Serving](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

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


## ✨ Stay Tuned

⭐ **Star this repository to keep up with exciting updates and new releases, including powerful OCR and document parsing capabilities!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr2.en.gif" alt="Star-Project">
  </p>
</div>


## 👩‍👩‍👧‍👦 Community

<div align="center">

| PaddlePaddle WeChat official account |  Join the tech discussion group |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |
</div>


## 😃 Awesome Projects Leveraging PaddleOCR
PaddleOCR wouldn't be where it is today without its incredible community! 💗 A massive thank you to all our longtime partners, new collaborators, and everyone who's poured their passion into PaddleOCR — whether we've named you or not. Your support fuels our fire!

<div align="center">

| Project Name | Description |
| ------------ | ----------- |
| [Dify](https://github.com/langgenius/dify) <a href="https://github.com/langgenius/dify"><img src="https://img.shields.io/github/stars/langgenius/dify"></a>|Production-ready platform for agentic workflow development.|
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|RAG engine based on deep document understanding.|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|Python ETL framework for stream processing, real-time analytics, LLM pipelines, and RAG.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Multi-type Document to Markdown Conversion Tool|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Free, Open-source, Batch Offline OCR Software.|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|A desktop client that supports for multiple LLM providers.|
| [haystack](https://github.com/deepset-ai/haystack)<a href="https://github.com/deepset-ai/haystack"><img src="https://img.shields.io/github/stars/deepset-ai/haystack"></a> |AI orchestration framework to build customizable, production-ready LLM applications.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |Question and Answer based on Anything.|
| [Learn more projects](./awesome_projects.md) | [More projects based on PaddleOCR](./awesome_projects.md)|
</div>

## 👩‍👩‍👧‍👦 Contributors

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>
</div>

## 🌟 Star

<div align="center">
  <p>
      <img width="800" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="Star-history">
  </p>
</div>


## 📄 License
This project is released under the [Apache 2.0 license](LICENSE).

## 🎓 Citation

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
