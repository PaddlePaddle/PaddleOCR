<div align="center">
  <p>
      <img width="800" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Banner.png" alt="Star-history">
  </p>



<h3>全球領先的 OCR 工具包與文檔 AI 引擎</h3>


[English](../README.md) | [简体中文](./README_cn.md)| 繁體中文 | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

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




**PaddleOCR 以業界領先的精準度，將 PDF 文件和圖像轉換為結構化、LLM 友好的資料格式（JSON/Markdown）。憑藉 70,000+ Stars 的成績，PaddleOCR 已獲得 Dify、RAGFlow、Cherry Studio 等頂級專案的廣泛信賴，是建構智慧 RAG 和 Agentic 應用的核心基礎元件。**


## 🚀 核心特性

### 📄 智能文檔解析（面向大模型）
> *為大模型時代將雜亂的文檔視覺信息轉化為結構化數據。*

* **SOTA 級文檔視覺語言模型 (VLM)**: 業界領先的輕量級文檔解析視覺語言模型 **PaddleOCR-VL-1.5 (0.9B)**。它在五大"真實場景"中表現卓越：**彎曲、掃描、屏幕拍照、複雜光照及傾斜文檔**，並支持以 **Markdown** 和 **JSON** 格式輸出結構化結果。
* **版面結構分析**：由**PP-StructureV3**驅動，無縫將複雜的PDF和圖像轉換為**Markdown**或**JSON**格式。與PaddleOCR-VL系列模型不同，它提供更細粒度的坐標信息,包括表格單元格坐標、文本坐標等，
* **生產級高效能**：以極小的模型體積實現商業級別的準確率。在公開基準測試中超越眾多閉源解決方案，同時保持極高的資源利用率，完美適配邊緣計算與雲端部署。

### 🔍 通用文本識別（場景 OCR）
> *快速、多語言文本檢測與識別的全球黃金標準。*

* **支持 100+ 種語言**：原生支持龐大豐富的全球語種庫。我們的 **PP-OCRv5** 模型解決方案能夠優雅應對多語言混合排版文檔（中文、英文、日文、拼音等）。
* **複雜場景支持**：除了標準的文本識別，我們還支持在各種廣泛的環境下進行**自然場景文本檢測與識別**，涵蓋身份證件、街景、書籍以及工業零部件等。
* **性能提升**：PP-OCRv5 相比前代版本實現了 **13% 的準確率提升**，同時延續了 PaddleOCR 的"極致高效"特性。

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch_cn.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

### 🛠️ 以開發者為中心的生態系統
* **無縫集成**：AI智能體生態系統的首選——與**Dify、RAGFlow、Pathway和Cherry Studio**深度集成。
* **大語言模型數據飛輪**：完整的數據流水線,用於構建高質量數據集，為微調大語言模型提供可持續的"數據引擎"。
* **一鍵部署**：支持多種硬件後端(NVIDIA GPU、Intel CPU、昆侖芯XPU和多種AI加速器)。


## 📣 最新動態

### 🔥 PaddleOCR v3.5.0 發布：推理後端更靈活，文檔輸出更豐富
* **推理後端靈活切換**：支持在飛槳靜態圖、飛槳動態圖和 Transformers 之間無縫切換。深度適配 Hugging Face 生態，20 個主要模型支持以 Transformers 作為推理後端。
* **常見文檔格式轉 Markdown**：支持將 Word、Excel、PowerPoint 等常見文檔格式轉換為 Markdown。
* **解析結果導出 DOCX**：`PaddleOCR-VL` 系列、`PP-StructureV3` 和 `PP-DocTranslation` 現已支持將解析結果導出為 DOCX，便於在 Microsoft Word 中查看和編輯。
* **官方瀏覽器推理 SDK**：發布官方瀏覽器推理 SDK `PaddleOCR.js`，支持在瀏覽器中運行 `PP-OCRv5`。

<details>
<summary><strong>2026.01.29: PaddleOCR 3.4.0 發布</strong></summary>
* **PaddleOCR-VL-1.5 (SOTA 0.9B VLM)**：我們最新的旗艦文檔解析模型現已上線!
    * **OmniDocBench 94.5%準確率**：超越頂級通用大模型和專業文檔解析模型。
    * **現實5大場景文檔解析的SOTA性能**：首次引入**PP-DocLayoutV3**算法進行不規則形狀定位，掌控5種艱難場景:傾斜、彎曲、掃描、光照和屏幕拍照。
    * **能力拓展**：現已支持**印章識別**、**文本識別**，並擴展至**111種語言**(包括中國的藏文和孟加拉語)。
    * **長文檔跨頁解析**：支持自動跨頁表格合併和分層標題識別。
    * **立即試用**：可在[HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)或我們的[官方網站](https://www.paddleocr.com)使用。

</details>

<details>
<summary><strong>2025.10.16: PaddleOCR 3.3.0 發布</strong></summary>

- **發布PaddleOCR-VL**：
    - **模型介紹**：
        - **PaddleOCR-VL** 是一款先進、高效的文檔解析模型，專為文檔中的元素識別設計。其核心組件為 PaddleOCR-VL-0.9B，這是一種緊湊而強大的視覺語言模型（VLM），它由 NaViT 風格的動態分辨率視覺編碼器與 ERNIE-4.5-0.3B 語言模型組成，能夠實現精準的元素識別。**該模型支持 109 種語言，並在識別複雜元素（如文本、表格、公式和圖表）方面表現出色，同時保持極低的資源消耗。通過在廣泛使用的公開基準與內部基準上的全面評測，PaddleOCR-VL 在頁級級文檔解析與元素級識別均達到 SOTA 表現**。它顯著優於現有的基於Pipeline方案和文檔解析多模態方案以及先進的通用多模態大模型，並具備更快的推理速度。這些優勢使其非常適合在真實場景中落地部署。模型已發布至[HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)，歡迎大家下載使用！更多介紹內容請點擊[PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html)。

    - **特性**：
        - **緊湊而強大的視覺語言模型架構**：我們提出了一種新的視覺語言模型，專為資源高效的推理而設計，在元素識別方面表現出色。通過將NaViT風格的動態高分辨率視覺編碼器與輕量級的ERNIE-4.5-0.3B語言模型結合，我們顯著增強了模型的識別能力和解碼效率。這種集成在保持高準確率的同時降低了計算需求，使其非常適合高效且實用的文檔處理應用。
        - **文檔解析的SOTA性能**：PaddleOCR-VL在頁面級文檔解析和元素級識別中達到了最先進的性能。它顯著優於現有的基於流水線的解決方案，並在文檔解析中展現出與領先的視覺語言模型（VLMs）競爭的強勁實力。此外，它在識別複雜的文檔元素（如文本、表格、公式和圖表）方面表現出色，使其適用於包括手寫文本和歷史文獻在內的各種具有挑戰性的內容類型。這使得它具有高度的多功能性，適用於廣泛的文檔類型和場景。
        - **多語言支持**：PaddleOCR-VL支持109種語言，覆蓋了主要的全球語言，包括但不限於中文、英文、日文、拉丁文和韓文，以及使用不同文字和結構的語言，如俄語（西里爾字母）、阿拉伯語、印地語（天城文）和泰語。這種廣泛的語言覆蓋大大增強了我們系統在多語言和全球化文檔處理場景中的適用性。

- **發布PP-OCRv5小語種識別模型**：
    - 優化拉丁文識別的準度和廣度，新增西里爾文、阿拉伯文、天城文、泰盧固語、泰米爾語等語系，覆蓋109種語言文字的識別。模型參數量僅為2M，部分模型精度較上一代提升40%以上。

</details>


<details>
<summary><strong>2025.08.21: PaddleOCR 3.2.0 發布</strong></summary>

- **重要模型新增：**
    - 新增 PP-OCRv5 英文、泰文、希臘文識別模型的訓練、推理、部署。**其中 PP-OCRv5 英文模型較 PP-OCRv5 主模型在英文場景提升 11%，泰文識別模型精度 82.68%，希臘文識別模型精度 89.28%。**

- **部署能力升級：**
    - **全面支持飛槳框架 3.1.0 和 3.1.1 版本。**
    - **全面升級 PP-OCRv5 C++ 本地部署方案，支持 Linux、Windows，功能及精度效果與 Python 方案保持一致。**
    - **高性能推理支持 CUDA 12，可使用 Paddle Inference、ONNX Runtime 後端推理。**
    - **高穩定性服務化部署方案全面開源，支持用戶根據需求對 Docker 鏡像和 SDK 進行定制化修改。**
    - 高穩定性服務化部署方案支持通過手動構造HTTP請求的方式調用，該方式允許客戶端代碼使用任意編程語言編寫。

- **Benchmark支持**：
    - **全部產線支持產線細粒度 benchmark，能夠測量產線端到端推理時間以及逐層、逐模塊的耗時數據，可用於輔助產線性能分析。可以參考[文檔](../docs/version3.x/pipeline_usage/instructions/benchmark.md)來進行性能測試。**
    - **文檔中補充各產線常用配置在主流硬件上的關鍵指標，包括推理耗時和內存佔用等，為用戶部署提供參考。**

- **Bug修復：**
    - 修復模型訓練時訓練日誌保存失敗的問題。
    - 對公式模型的數據增強部分進行了版本兼容性升級，以適應新版本的 albumentations 依賴，並修復了在多進程使用 tokenizers 依賴包時出現的死鎖警告。
    - 修復 PP-StructureV3 配置文件中的 `use_chart_parsing` 等開關行為與其他產線不統一的問題。

- **其他升級：**
    - **分離必要依賴與可選依賴。使用基礎文字識別功能時，僅需安裝少量核心依賴；若需文檔解析、信息抽取等功能，用戶可按需選擇安裝額外依賴。**
    - **支持 Windows 用戶使用英偉達 50 系顯卡，可根據 [安裝文檔](../docs/version3.x/installation.md) 安裝對應版本的 paddle 框架。**
    - **PP-OCR 系列模型支持返回單文字坐標。**
    - 模型新增 AIStudio、ModelScope 等下載源。可指定相關下載源下載對應的模型。
    - 支持圖表轉表 PP-Chart2Table 單功能模塊推理能力。
    - 優化部分使用文檔中的描述，提升易用性。
</details>


[歷史日誌](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>


## 🚀 快速開始

### 步驟 1: 在線體驗
PaddleOCR官方網站提供交互式**體驗中心**和**APIs**——無需設置,一鍵體驗。

👉 [訪問官方網站](https://www.paddleocr.com)

### 步驟 2: 本地部署
對於本地使用,請根據您的需求參考以下文檔:

- **PP-OCR系列**：查看[PP-OCR文檔](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/OCR.html)
- **PaddleOCR-VL系列**：查看[PaddleOCR-VL文檔](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html)
- **PP-StructureV3**：查看[PP-StructureV3文檔](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- **更多能力**：查看[更多能力文檔](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/pipeline_overview.html)


## 🧩 更多功能

- 將模型轉換為ONNX格式: [獲取ONNX模型](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/obtaining_onnx_models.html)。
- 使用OpenVINO、ONNX Runtime、TensorRT等引擎加速推理,或使用ONNX格式模型進行推理: [高性能推理](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/high_performance_inference.html)。
- 使用多GPU和多進程加速推理: [流水線並行推理](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/instructions/parallel_inference.html)。
- 將PaddleOCR集成到C++、C#、Java等語言編寫的應用程序中: [服務化部署](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/serving.html)。

## 🔄 執行結果快速預覽

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


## ✨ 保持關注

⭐ **收藏本倉庫，持續關注最新動態與版本發布，包括強大的 OCR 及文檔解析等新功能特性。** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr2.en.gif" alt="Star-Project">
  </p>
</div>


## 👩‍👩‍👧‍👦 社區

<div align="center">

| PaddlePaddle 微信公眾號 |  加入技術討論群 |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |
</div>


## 😃 使用 PaddleOCR 的優秀項目

<div align="center">

PaddleOCR 的發展離不開社區貢獻！💗衷心感謝所有開發者、合作夥伴與貢獻者！
| 項目名稱 | 簡介 |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|基於RAG的AI工作流引擎|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|用於流處理、實時分析、LLM流水線和RAG的Python ETL框架|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|多類型文檔轉換Markdown工具|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|開源批量離線OCR軟件|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|一個支持多個LLM提供商的桌面客戶端|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |基於純視覺的GUI智能體屏幕解析工具|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |基於任意內容的問答系統|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|高效複雜PDF文檔提取工具包|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |屏幕實時翻譯工具|
| [更多項目](../awesome_projects.md) |  [更多基於PaddleOCR的項目](../awesome_projects.md) |
</div>


## 👩‍👩‍👧‍👦 貢獻者

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>
</div>

## 🌟 Star歷史

<div align="center">
  <p>
      <img width="800" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="Star-history">
  </p>
</div>


## 📄 許可證
本項目採用[Apache 2.0許可證](LICENSE)發布。

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
