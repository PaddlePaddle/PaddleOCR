<div align="center">
  <p>
      <img width="100%" src="./docs/images/Banner_cn.png" alt="PaddleOCR 橫幅">
  </p>

<!-- language -->
[English](./README.md) | [简体中文](./README_cn.md) | 繁體中文 | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

<!-- icon -->

[![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![Downloads](https://img.shields.io/pypi/dm/paddleocr)](https://pypi.org/project/PaddleOCR/)
![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)


[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

</div>

## 🚀 簡介
PaddleOCR 自發布以來，憑藉其學術前沿的演算法與產業落地實踐，深受產學研各界的喜愛，並廣泛應用於眾多知名開源專案，如 Umi-OCR、OmniParser、MinerU、RAGFlow 等，已成為廣大開發者心中開源 OCR 領域的首選工具。2025 年 5 月 20 日，飛槳團隊發布 **PaddleOCR 3.0**，全面適配**飛槳框架 3.0 正式版**，進一步**提升文字辨識精度**，支援**多種文字類型辨識**和**手寫體辨識**，滿足大型模型應用對**複雜文件高精度解析**的旺盛需求。結合**ERNIE 4.5 Turbo**，顯著提升了關鍵資訊擷取的精度，並新增**對崑崙芯、昇騰等國產硬體**的支援。完整使用說明請參閱 [PaddleOCR 3.0 文檔](https://paddlepaddle.github.io/PaddleOCR/latest/)。

PaddleOCR 3.0 **新增**三大特色功能：
- 全場景文字辨識模型 [PP-OCRv5](docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)：單一模型支援五種文字類型和複雜手寫體辨識；整體辨識精度相較前一代**提升 13 個百分點**。[線上體驗](https://aistudio.baidu.com/community/app/91660/webUI)
- 通用文件解析方案 [PP-StructureV3](docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.md)：支援多場景、多版式的 PDF 高精度解析，在公開評測集中**領先眾多開源與閉源方案**。[線上體驗](https://aistudio.baidu.com/community/app/518494/webUI)
- 智慧文件理解方案 [PP-ChatOCRv4](docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.md)：原生支援ERNIE 4.5 Turbo，精度相較前一代**提升 15 個百分點**。[線上體驗](https://aistudio.baidu.com/community/app/518493/webUI)

除了提供優秀的模型庫，PaddleOCR 3.0 還提供好學易用的工具，涵蓋模型訓練、推論及服務化部署，方便開發者快速將 AI 應用落地。
<div align="center">
  <p>
      <img width="100%" src="./docs/images/Arch_cn.png" alt="PaddleOCR 架構">
  </p>
</div>


## 📣 最新動態
🔥🔥2025.06.05: **PaddleOCR 3.0.1** 發布，包含：

- **優化部分模型和模型設定：**
  - 更新 PP-OCRv5 預設模型設定，偵測和辨識模型均由 mobile 改為 server 模型。為改善多數場景下的預設效果，設定中的參數 `limit_side_len` 由 736 改為 64。
  - 新增文字行方向分類模型 `PP-LCNet_x1_0_textline_ori`，精度達 99.42%。OCR、PP-StructureV3、PP-ChatOCRv4 流程的預設文字行方向分類器已更新為此模型。
  - 優化文字行方向分類模型 `PP-LCNet_x0_25_textline_ori`，精度提升 3.3 個百分點，目前精度為 98.85%。
- **優化及修復 3.0.0 版本的部分問題，[詳情](https://paddlepaddle.github.io/PaddleOCR/latest/update/update.html)**

🔥🔥2025.05.20: **PaddleOCR 3.0** 正式發布，包含：
- **PP-OCRv5**: 全場景高精度文字辨識

   1. 🌐 單一模型支援**五種**文字類型（**簡體中文**、**繁體中文**、**中文拼音**、**英文**和**日文**）。
   2. ✍️ 支援複雜**手寫體**辨識：顯著提升對複雜連筆、非標準字跡的辨識效能。
   3. 🎯 整體辨識精度提升：在多種應用場景達到 SOTA 精度，相較於上一版 PP-OCRv4，辨識精度**提升 13 個百分點**！

- **PP-StructureV3**: 通用文件解析方案

   1. 🧮 支援多場景 PDF 高精度解析，在 OmniDocBench 基準測試中**領先眾多開源與閉源方案**。
   2. 🧠 多項專業功能：**印章辨識**、**圖表轉表格**、**含嵌套公式/圖片的表格辨識**、**直書文字解析**及**複雜表格結構分析**等。


- **PP-ChatOCRv4**: 智慧文件理解方案
   1. 🔥 文件影像（PDF/PNG/JPG）關鍵資訊擷取精度相較前一代**提升 15 個百分點**！
   2. 💻 原生支援**ERNIE 4.5 Turbo**，並相容 PaddleNLP、Ollama、vLLM 等工具部署的大型模型。
   3. 🤝 整合 [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2)，支援印刷體、手寫體、印章、表格、圖表等複雜文件元素的資訊擷取與理解。


## ⚡ 快速入門
### 1. 線上體驗
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. 本機安裝

請參考[安裝指南](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)完成 **PaddlePaddle 3.0** 的安裝，然後安裝 paddleocr。

```bash
# 安裝 paddleocr
pip install paddleocr
```

### 3. 命令列推論
```bash
# 執行 PP-OCRv5 推論
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False 

# 執行 PP-StructureV3 推論
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# 執行 PP-ChatOCRv4 推論前，需先取得千帆 API Key
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 駕駛室准乘人數 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# 查看 "paddleocr ocr" 詳細參數
paddleocr ocr --help
```
### 4. API 推論

**4.1 PP-OCRv5 範例**
```python
from paddleocr import PaddleOCR
# 初始化 PaddleOCR 執行個體
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# 對範例圖片執行 OCR 推論
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")
    
# 將結果視覺化並儲存為 JSON
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary><strong>4.2 PP-StructureV3 範例</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

# 針對圖片
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
    )

# 將結果視覺化並儲存為 JSON
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output") 
```

</details>


<details>
   <summary><strong>4.3 PP-ChatOCRv4 範例</strong></summary>

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
# 若使用多模態大型模型，需啟動本機 mllm 服務，可參考文件：https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.md 進行部署，並更新 mllm_chat_bot_config 設定。
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


### 5. **國產硬體支援**
- [崑崙芯安裝指南](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_XPU.html)
- [昇騰安裝指南](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_NPU.html)
  
## ⛰️ 進階指南
- [PP-OCRv5 使用教學](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [PP-StructureV3 使用教學](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [PP-ChatOCRv4 使用教學](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 效果展示

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

## 👩‍👩‍👧‍👦 開發者社群

| 掃描 QR Code 關注飛槳官方帳號 | 掃描 QR Code 加入技術交流群組 |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |

## 🏆 採用 PaddleOCR 的優秀專案
PaddleOCR 的發展離不開社群的貢獻！💗 衷心感謝所有的開發者、合作夥伴與貢獻者！
| 專案名稱 | 簡介 |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|基於 RAG 的 AI 工作流引擎|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|多類型文件轉 Markdown 工具|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|開源批次離線 OCR 軟體|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |基於純視覺的 GUI Agent 螢幕解析工具|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |基於任意內容的問答系統|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|高效複雜 PDF 文件擷取工具套件|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |螢幕即時翻譯工具|
| [更多專案](./awesome_projects.md) | |

## 👩‍👩‍👧‍👦 貢獻者

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>


## 🌟 Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)


## 📄 授權條款
本專案的發布受 [Apache 2.0 license](LICENSE) 授權條款認證。

## 🎓 學術引用

```
@misc{paddleocr2020,
title={PaddleOCR, Awesome multilingual OCR toolkits based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleOCR}},
year={2020}
}
``` 
