<div align="center">
  <p>
      <img width="100%" src="./docs/images/Banner.png" alt="PaddleOCR Banner">
  </p>

English | [简体中文](./readme/README_cn.md) | [繁體中文](./readme/README_tcn.md) | [日本語](./readme/README_ja.md) | [한국어](./readme/README_ko.md) | [Français](./readme/README_fr.md) | [Русский](./readme/README_ru.md) | [Español](./readme/README_es.md) | [العربية](./readme/README_ar.md)

[![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![arXiv](https://img.shields.io/badge/arXiv-2507.05595-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2507.05595)
[![PyPI Downloads](https://static.pepy.tech/badge/paddleocr/month)](https://pepy.tech/project/paddleocr)
[![PyPI Downloads](https://static.pepy.tech/badge/paddleocr)](https://pepy.tech/project/paddleocr)
[![Used by](https://img.shields.io/badge/Used%20by-5.9k%2B%20repositories-blue)](https://github.com/PaddlePaddle/PaddleOCR/network/dependents)

![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)
[![License](https://img.shields.io/badge/license-Apache_2.0-green)](./LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PaddlePaddle/PaddleOCR)


**PaddleOCR is an industry-leading, production-ready OCR and document AI engine, offering end-to-end solutions from text extraction to intelligent document understanding**

</div>

# PaddleOCR
[![Framework](https://img.shields.io/badge/PaddlePaddle-3.0-orange)](https://www.paddlepaddle.org.cn/en)
[![Accuracy](https://img.shields.io/badge/Recognition%20Accuracy-🏆-green)](#)
[![Multi-Language](https://img.shields.io/badge/Support_Languages-80+-brightgreen)](#)
[![Handwriting](https://img.shields.io/badge/Handwriting-✓-success)](#)
[![Hardware](https://img.shields.io/badge/Heterogeneous%20Hardware-Kunlunxin%20%7C%20Ascend_NPU-red)](#)

> [!TIP]
> PaddleOCR now provides an MCP server that supports integration with Agent applications like Claude Desktop. For details, please refer to [PaddleOCR MCP Server](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/mcp_server.html).
>
> The PaddleOCR 3.0 Technical Report is now available. See details at: [PaddleOCR 3.0 Technical Report](https://arxiv.org/abs/2507.05595)


**PaddleOCR** converts documents and images into **structured, AI-friendly data** (like JSON and Markdown) with **industry-leading accuracy**—powering AI applications for everyone from indie developers and startups to large enterprises worldwide. With over **50,000 stars** and deep integration into leading projects like **MinerU, RAGFlow, and OmniParser**, PaddleOCR has become the **premier solution** for developers building intelligent document applications in the **AI era**.

### PaddleOCR 3.0 Core Features

[![AI Studio](https://img.shields.io/badge/PP_OCRv5-Demo_on_AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-Demo_on_AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-Demo_on_AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)
[![ModelScope](https://img.shields.io/badge/🤖_Demo_on_ModelScope-purple)](https://www.modelscope.cn/organization/PaddlePaddle)
[![HuggingFace](https://img.shields.io/badge/Demo_on_HuggingFace-purple.svg?logo=huggingface)](https://huggingface.co/PaddlePaddle)

- **PP-OCRv5 — Universal Scene Text Recognition**  
  **Single model supports five text types** (Simplified Chinese, Traditional Chinese, English, Japanese, and Pinyin) with **13% accuracy improvement**. Solves multilingual mixed document recognition challenges.

- **PP-StructureV3 — Complex Document Parsing**  
  Intelligently converts complex PDFs and document images into **Markdown and JSON files that preserve original structure**. **Outperforms** numerous commercial solutions in public benchmarks. **Perfectly maintains document layout and hierarchical structure**.

- **PP-ChatOCRv4 — Intelligent Information Extraction**  
  Natively integrates ERNIE 4.5 to **precisely extract key information** from massive documents, with 15% accuracy improvement over previous generation. Makes documents "**understand**" your questions and provide accurate answers.

In addition to providing an outstanding model library, PaddleOCR 3.0 also offers user-friendly tools covering model training, inference, and service deployment, so developers can rapidly bring AI applications to production.
<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

**Special Note**: PaddleOCR 3.x introduces several significant interface changes. **Old code written based on PaddleOCR 2.x is likely incompatible with PaddleOCR 3.x**. Please ensure that the documentation you are reading matches the version of PaddleOCR you are using. [This document](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/upgrade_notes.html) explains the reasons for the upgrade and the major changes from PaddleOCR 2.x to 3.x.

## 📣 Recent updates

### 🔥🔥2025.08.21: Release of PaddleOCR 3.2.0, includes:


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


<details>
<summary><strong>2025.08.15: PaddleOCR 3.1.1 Released</strong></summary>

- **Bug Fixes:**
  - Added the missing methods `save_vector`, `save_visual_info_list`, `load_vector`, and `load_visual_info_list` in the `PP-ChatOCRv4` class.
  - Added the missing parameters `glossary` and `llm_request_interval` to the `translate` method in the `PPDocTranslation` class.

- **Documentation Improvements:**
  - Added a demo to the MCP documentation.
  - Added information about the PaddlePaddle and PaddleOCR version used for performance metrics testing in the documentation.
  - Fixed errors and omissions in the production line document translation.

- **Others:**
  - Changed the MCP server dependency to use the pure Python library `puremagic` instead of `python-magic` to reduce installation issues.
  - Retested PP-OCRv5 performance metrics with PaddleOCR version 3.1.0 and updated the documentation.

</details>

<details>
<summary><strong>2025.06.29: PaddleOCR 3.1.0 Released</strong></summary>

- **Key Models and Pipelines:**
  - **Added PP-OCRv5 Multilingual Text Recognition Model**, which supports the training and inference process for text recognition models in 37 languages, including French, Spanish, Portuguese, Russian, Korean, etc. **Average accuracy improved by over 30%.** [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
  - Upgraded the **PP-Chart2Table model** in PP-StructureV3, further enhancing the capability of converting charts to tables. On internal custom evaluation sets, the metric (RMS-F1) **increased by 9.36 percentage points (71.24% -> 80.60%).**
  - Newly launched **document translation pipeline, PP-DocTranslation, based on PP-StructureV3 and ERNIE 4.5**, which supports the translation of Markdown format documents, various complex-layout PDF documents, and document images, with the results saved as Markdown format documents. [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-DocTranslation.html)


- **New MCP server:** [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/mcp_server.html)
  - **Supports both OCR and PP-StructureV3 pipelines.**
  - Supports three working modes: local Python library, AIStudio Community Cloud Service, and self-hosted service.
  - Supports invoking local services via stdio and remote services via Streamable HTTP.

- **Documentation Optimization:** Improved the descriptions in some user guides for a smoother reading experience.

</details>

<details>
    <summary><strong>2025.06.26: PaddleOCR 3.0.3 Released</strong></summary>
- Bug Fix: Resolved the issue where the `enable_mkldnn` parameter was not effective, restoring the default behavior of using MKL-DNN for CPU inference.
</details>

<details>
    <summary><strong>2025.06.19: PaddleOCR 3.0.2 Released</strong></summary>
- **New Features:**

  - The default download source has been changed from `BOS` to `HuggingFace`. Users can also change the environment variable `PADDLE_PDX_MODEL_SOURCE` to `BOS` to set the model download source back to Baidu Object Storage (BOS).
  - Added service invocation examples for six languages—C++, Java, Go, C#, Node.js, and PHP—for pipelines like PP-OCRv5, PP-StructureV3, and PP-ChatOCRv4.
  - Improved the layout partition sorting algorithm in the PP-StructureV3 pipeline, enhancing the sorting logic for complex vertical layouts to deliver better results.
  - Enhanced model selection logic: when a language is specified but a model version is not, the system will automatically select the latest model version supporting that language. 
  - Set a default upper limit for MKL-DNN cache size to prevent unlimited growth, while also allowing users to configure cache capacity.
  - Updated default configurations for high-performance inference to support Paddle MKL-DNN acceleration and optimized the logic for automatic configuration selection for smarter choices.
  - Adjusted the logic for obtaining the default device to consider the actual support for computing devices by the installed Paddle framework, making program behavior more intuitive.
  - Added Android example for PP-OCRv5. [Details](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/on_device_deployment.html).

- **Bug Fixes:**
  - Fixed an issue with some CLI parameters in PP-StructureV3 not taking effect.
  - Resolved an issue where `export_paddlex_config_to_yaml` would not function correctly in certain cases.
  - Corrected the discrepancy between the actual behavior of `save_path` and its documentation description.
  - Fixed potential multithreading errors when using MKL-DNN in basic service deployment.
  - Corrected channel order errors in image preprocessing for the Latex-OCR model.
  - Fixed channel order errors in saving visualized images within the text recognition module.
  - Resolved channel order errors in visualized table results within PP-StructureV3 pipeline.
  - Fixed an overflow issue in the calculation of `overlap_ratio` under extremely special circumstances in the PP-StructureV3 pipeline.

- **Documentation Improvements:**
  - Updated the description of the `enable_mkldnn` parameter in the documentation to accurately reflect the program's actual behavior.
  - Fixed errors in the documentation regarding the `lang` and `ocr_version` parameters.
  - Added instructions for exporting pipeline configuration files via CLI.
  - Fixed missing columns in the performance data table for PP-OCRv5.
  - Refined benchmark metrics for PP-StructureV3 across different configurations.

- **Others:**

  - Relaxed version restrictions on dependencies like numpy and pandas, restoring support for Python 3.12.
</details>

<details>
    <summary><strong>History Log</strong></summary>

2025.06.05: **PaddleOCR 3.0.1 Released**, includes:

- **Optimisation of certain models and model configurations:**
  - Updated the default model configuration for PP-OCRv5, changing both detection and recognition from mobile to server models. To improve default performance in most scenarios, the parameter `limit_side_len` in the configuration has been changed from 736 to 64.
  - Added a new text line orientation classification model `PP-LCNet_x1_0_textline_ori` with an accuracy of 99.42%. The default text line orientation classifier for OCR, PP-StructureV3, and PP-ChatOCRv4 pipelines has been updated to this model.
  - Optimized the text line orientation classification model `PP-LCNet_x0_25_textline_ori`, improving accuracy by 3.3 percentage points to a current accuracy of 98.85%.
- **Optimizations and fixes for some issues in version 3.0.0, [details](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)**

🔥🔥2025.05.20: Official Release of **PaddleOCR v3.0**, including:
- **PP-OCRv5**: High-Accuracy Text Recognition Model for All Scenarios - Instant Text from Images/PDFs.
   1. 🌐 Single-model support for **five** text types - Seamlessly process **Simplified Chinese, Traditional Chinese, Simplified Chinese Pinyin, English** and **Japanese** within a single model.
   2. ✍️ Improved **handwriting recognition**: Significantly better at complex cursive scripts and non-standard handwriting.
   3. 🎯 **13-point accuracy gain** over PP-OCRv4, achieving state-of-the-art performance across a variety of real-world scenarios.

- **PP-StructureV3**: General-Purpose Document Parsing – Unleash SOTA Images/PDFs Parsing for Real-World Scenarios! 
   1. 🧮 **High-Accuracy multi-scene PDF parsing**, leading both open- and closed-source solutions on the OmniDocBench benchmark.
   2. 🧠 Specialized capabilities include **seal recognition**, **chart-to-table conversion**, **table recognition with nested formulas/images**, **vertical text document parsing**, and **complex table structure analysis**.

- **PP-ChatOCRv4**: Intelligent Document Understanding – Extract Key Information, not just text from Images/PDFs.
   1. 🔥 **15-point accuracy gain** in key-information extraction on PDF/PNG/JPG files over the previous generation.
   2. 💻 Native support for **ERNIE 4.5**, with compatibility for large-model deployments via PaddleNLP, Ollama, vLLM, and more.
   3. 🤝 Integrated [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2), enabling extraction and understanding of printed text, handwriting, seals, tables, charts, and other common elements in complex documents.

[History Log](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>

## ⚡ Quick Start
### 1. Run online demo 
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. Installation

Install PaddlePaddle refer to [Installation Guide](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html), after then, install the PaddleOCR toolkit.

```bash
# If you only want to use the basic text recognition feature (returns text position coordinates and content), including the PP-OCR series
python -m pip install paddleocr
# If you want to use all features such as document parsing, document understanding, document translation, key information extraction, etc.
# python -m pip install "paddleocr[all]"
```

Starting from version 3.2.0, in addition to the `all` dependency group demonstrated above, PaddleOCR also supports installing partial optional features by specifying other dependency groups. All dependency groups provided by PaddleOCR are as follows:

| Dependency Group Name | Corresponding Functionality |
| - | - |
| `doc-parser` | Document parsing: can be used to extract layout elements such as tables, formulas, stamps, images, etc. from documents; includes models like PP-StructureV3 |
| `ie` | Information extraction: can be used to extract key information from documents, such as names, dates, addresses, amounts, etc.; includes models like PP-ChatOCRv4 |
| `trans` | Document translation: can be used to translate documents from one language to another; includes models like PP-DocTranslation |
| `all` | Complete functionality |

### 3. Run inference by CLI
```bash
# Run PP-OCRv5 inference
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False  

# Run PP-StructureV3 inference
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# Get the Qianfan API Key at first, and then run PP-ChatOCRv4 inference
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# Get more information about "paddleocr ocr"
paddleocr ocr --help
```

### 4. Run inference by API
**4.1 PP-OCRv5 Example**
```python
# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Run OCR inference on a sample image 
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary><strong>4.2 PP-StructureV3 Example</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

# For Image
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
)

# Visualize the results and save the JSON results
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output")           
```

</details>

<details>
   <summary><strong>4.3 PP-ChatOCRv4 Example</strong></summary>

```python
from paddleocr import PPChatOCRv4Doc

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # your api_key
}

pipeline = PPChatOCRv4Doc(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

visual_predict_res = pipeline.visual_predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png",
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

mllm_predict_info = None
use_mllm = False
# If a multimodal large model is used, the local mllm service needs to be started. You can refer to the documentation: https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.en.md performs deployment and updates the mllm_chat_bot_config configuration.
if use_mllm:
    mllm_chat_bot_config = {
        "module_name": "chat_bot",
        "model_name": "PP-DocBee",
        "base_url": "http://127.0.0.1:8080/",  # your local mllm service url
        "api_type": "openai",
        "api_key": "api_key",  # your api_key
    }

    mllm_predict_res = pipeline.mllm_pred(
        input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png",
        key_list=["驾驶室准乘人数"],
        mllm_chat_bot_config=mllm_chat_bot_config,
    )
    mllm_predict_info = mllm_predict_res["mllm_res"]

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])
    layout_parsing_result = res["layout_parsing_result"]

vector_info = pipeline.build_vector(
    visual_info_list, flag_save_bytes_vector=True, retriever_config=retriever_config
)
chat_result = pipeline.chat(
    key_list=["驾驶室准乘人数"],
    visual_info=visual_info_list,
    vector_info=vector_info,
    mllm_predict_info=mllm_predict_info,
    chat_bot_config=chat_bot_config,
    retriever_config=retriever_config,
)
print(chat_result)
```

</details>

### 5. Chinese Heterogeneous AI Accelerators
- [Huawei Ascend](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_NPU.html)
- [KUNLUNXIN](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_XPU.html)

## 🧩 More Features

- Convert models to ONNX format: [Obtaining ONNX Models](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- Accelerate inference using engines like OpenVINO, ONNX Runtime, TensorRT, or perform inference using ONNX format models: [High-Performance Inference](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- Accelerate inference using multi-GPU and multi-process: [Parallel Inference for Pipelines](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- Integrate PaddleOCR into applications written in C++, C#, Java, etc.: [Serving](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## ⛰️ Advanced Tutorials
- [PP-OCRv5 Tutorial](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [PP-StructureV3 Tutorial](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [PP-ChatOCRv4 Tutorial](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 Quick Overview of Execution Results

<div align="center">
  <p>
     <img width="100%" src="./docs/images/demo.gif" alt="PP-OCRv5 Demo">
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="./docs/images/blue_v3.gif" alt="PP-StructureV3 Demo">
  </p>
</div>


## ✨ Stay Tuned

⭐ **Star this repository to keep up with exciting updates and new releases, including powerful OCR and document parsing capabilities!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr.en.gif" alt="Star-Project">
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
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|RAG engine based on deep document understanding.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|Multi-type Document to Markdown Conversion Tool|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|Free, Open-source, Batch Offline OCR Software.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |OmniParser: Screen Parsing tool for Pure Vision Based GUI Agent.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |Question and Answer based on Anything.|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|A powerful open-source toolkit designed to efficiently extract high-quality content from complex and diverse PDF documents.|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |Recognize text on the screen, translate it and show the translation results in real time.|
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

```
@misc{cui2025paddleocr30technicalreport,
      title={PaddleOCR 3.0 Technical Report}, 
      author={Cheng Cui and Ting Sun and Manhui Lin and Tingquan Gao and Yubo Zhang and Jiaxuan Liu and Xueqing Wang and Zelun Zhang and Changda Zhou and Hongen Liu and Yue Zhang and Wenyu Lv and Kui Huang and Yichao Zhang and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2507.05595},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05595}, 
}
```
