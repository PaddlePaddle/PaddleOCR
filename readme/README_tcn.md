<div align="center">
  <p>
      <img width="100%" src="../docs/images/Banner_cn.png" alt="PaddleOCR 橫幅">
  </p>

<!-- language -->
[English](../README.md) | [简体中文](./README_cn.md) | 繁體中文 | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

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
PaddleOCR 自發布以來，憑藉其學術前沿的演算法與產業落地實踐，深受產學研各界的喜愛，並廣泛應用於眾多知名開源專案，如 Umi-OCR、OmniParser、MinerU、RAGFlow 等，已成為廣大開發者心中開源 OCR 領域的首選工具。2025 年 5 月 20 日，飛槳團隊發布 **PaddleOCR 3.0**，全面適配**飛槳框架 3.0 正式版**，進一步**提升文字辨識精度**，支援**多種文字類型辨識**和**手寫體辨識**，滿足大型模型應用對**複雜文件高精度解析**的旺盛需求。結合**ERNIE 4.5**，顯著提升了關鍵資訊擷取的精度，並新增**對崑崙芯、昇騰等國產硬體**的支援。完整使用說明請參閱 [PaddleOCR 3.0 文檔](https://paddlepaddle.github.io/PaddleOCR/latest/)。

PaddleOCR 3.0 **新增**三大特色功能：
- 全場景文字辨識模型 [PP-OCRv5](../docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)：單一模型支援五種文字類型和複雜手寫體辨識；整體辨識精度相較前一代**提升 13 個百分點**。[線上體驗](https://aistudio.baidu.com/community/app/91660/webUI)
- 通用文件解析方案 [PP-StructureV3](../docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.md)：支援多場景、多版式的 PDF 高精度解析，在公開評測集中**領先眾多開源與閉源方案**。[線上體驗](https://aistudio.baidu.com/community/app/518494/webUI)
- 智慧文件理解方案 [PP-ChatOCRv4](../docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.md)：原生支援ERNIE 4.5，精度相較前一代**提升 15 個百分點**。[線上體驗](https://aistudio.baidu.com/community/app/518493/webUI)

除了提供優秀的模型庫，PaddleOCR 3.0 還提供好學易用的工具，涵蓋模型訓練、推論及服務化部署，方便開發者快速將 AI 應用落地。
<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch_cn.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

**特別注意**：PaddleOCR 3.x 引入了多項重大的介面變更。**基於 PaddleOCR 2.x 撰寫的舊程式碼很可能與 PaddleOCR 3.x 不相容**。請確保您閱讀的文件與您實際使用的 PaddleOCR 版本相符。[本文件](https://paddlepaddle.github.io/PaddleOCR/latest/update/upgrade_notes.html) 說明了升級的原因以及從 PaddleOCR 2.x 到 3.x 的主要變更。

## 📣 最新動態

 **🔥🔥2025.08.21: 發布 PaddleOCR 3.2.0**，內容包括：


- **重要模型新增：**
    - 新增 PP-OCRv5 英文、泰文、希臘文識別模型的訓練、推理、部署。**其中 PP-OCRv5 英文模型在英文場景下較 PP-OCRv5 主模型提升 11%，泰文識別模型準確率達 82.68%，希臘文識別模型準確率達 89.28%。**

- **部署能力升級：**
    - **全面支援飛槳（PaddlePaddle）框架 3.1.0 與 3.1.1 版本。**
    - **全面升級 PP-OCRv5 C++ 本地部署方案，支援 Linux、Windows，功能及精度與 Python 方案保持一致。**
    - **高效能推理支援 CUDA 12，可使用 Paddle Inference、ONNX Runtime 後端進行推理。**
    - **高穩定性服務化部署方案全面開源，支援用戶根據需求自訂 Docker 映像檔與 SDK。**
    - 高穩定性服務化部署方案支援通過手動構造 HTTP 請求的方式調用，允許客戶端程式可用任意程式語言編寫。

- **Benchmark 支援：**
    - **所有產線全面支援細粒度 benchmark，能測量產線端到端推理時間及逐層、逐模組耗時，用於協助產線效能分析。可以參考[文件](../docs/version3.x/pipeline_usage/instructions/benchmark.md)來進行效能測試。**
    - **文件中補充各產線常用配置於主流硬體上的關鍵指標，包括推理耗時、記憶體佔用等，為用戶部署提供參考。**

- **Bug 修復：**
    - 修復模型訓練時訓練日誌無法儲存的問題。
    - 對公式模型的資料增強部分進行版本相容性升級，以適應新版本 albumentations 依賴，並修復多進程使用 tokenizers 套件時出現的死鎖警告。
    - 修復 PP-StructureV3 配置檔案中 `use_chart_parsing` 等開關行為與其他產線不一致的問題。

- **其他升級：**
    - **區分必要依賴與可選依賴，使用基礎文字識別功能時僅需安裝少量核心依賴；如需文檔解析、資訊抽取等功能，使用者可按需安裝額外依賴。**
    - **支援 Windows 用戶使用 NVIDIA 50 系顯示卡，可依照[安裝文件](../docs/version3.x/installation.md)安裝對應版本的 Paddle 框架。**
    - **PP-OCR 系列模型支援返回單字座標。**
    - 模型新增 AIStudio、ModelScope 等下載來源，可指定相關來源下載對應模型。
    - 支援圖表轉表單一模組（PP-Chart2Table）推理能力。
    - 優化部分使用文件中的描述，提升易用性。

 **2025.08.15: 發布 PaddleOCR 3.1.1**，內容包括：

- **Bug修復：**
  - 補充 `PP-ChatOCRv4` 類缺失的 `save_vector`、`save_visual_info_list`、`load_vector`、`load_visual_info_list` 方法。
  - 補充 `PPDocTranslation` 類的 `translate` 方法缺失的 `glossary` 和 `llm_request_interval` 參數。

- **文件優化：**
  - 補充 MCP 文件中的 demo。
  - 補充文件中測試性能指標所使用的飛槳框架與 PaddleOCR 版本。
  - 修正文件翻譯產線文件中的錯漏。

- **其他：**
  - 修改 MCP 伺服器依賴，使用純 Python 函式庫 `puremagic` 取代 `python-magic`，以減少安裝問題。
  - 使用 3.1.0 版本 PaddleOCR 重新測試 PP-OCRv5 性能指標，並更新文件。


 **🔥🔥2025.06.29：發布 PaddleOCR 3.1.0**，內容包括：

- **主要模型與流程：**
  - **新增 PP-OCRv5 多語言文字識別模型**，支援包括法語、西班牙語、葡萄牙語、俄語、韓語等在內的 37 種語言的文字識別模型訓練與推理。**平均準確率提升超過 30%。** [詳情](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html)
  - 升級了 PP-StructureV3 的 **PP-Chart2Table 模型**，進一步提升圖表轉表格能力。在內部自訂評測集上，指標（RMS-F1）**提升了 9.36 個百分點（71.24% -> 80.60%）。**
  - 新增基於 PP-StructureV3 和 ERNIE 4.5 的**文件翻譯流程 PP-DocTranslation**，支援 Markdown 格式文件、各種複雜版面 PDF 文件及文件圖片翻譯，結果可儲存為 Markdown 格式文件。[詳情](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-DocTranslation.html)

- **新增 MCP server：**[Details](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/mcp_server.html)
  - **支援 OCR 及 PP-StructureV3 流程。**
  - 支援三種工作模式：本地 Python 函式庫、AIStudio 社群雲端服務、自主託管服務。
  - 支援通過 stdio 調用本地服務，通過 Streamable HTTP 調用遠端服務。

- **文件優化：** 優化了部分使用說明文件描述，提升閱讀體驗。

<details>
    <summary><strong>歷史日誌</strong></summary>
    
2025.06.26: **PaddleOCR 3.0.3** 發布，包含：

- 錯誤修復：修復`enable_mkldnn`參數不生效的問題，恢復CPU默認使用MKL-DNN推理的行為。


2025.06.19: **PaddleOCR 3.0.2** 發布，包含：

- **功能新增：**
  - 模型預設下載來源從`BOS`改為`HuggingFace`，同時也支援使用者透過更改環境變數`PADDLE_PDX_MODEL_SOURCE`為`BOS`，將模型下載來源設定為百度雲端物件儲存 BOS。
  - PP-OCRv5、PP-StructureV3、PP-ChatOCRv4 等 pipeline 新增 C++、Java、Go、C#、Node.js、PHP 6 種語言的服務呼叫範例。
  - 優化 PP-StructureV3 產線中版面分區排序演算法，對複雜直書版面排序邏輯進行完善，進一步提升了複雜版面排序效果。
  - 優化模型選擇邏輯，當指定語言、未指定模型版本時，自動選擇支援該語言的最新版本的模型。
  - 為 MKL-DNN 快取大小設定預設上限，防止快取無限增長。同時，支援使用者設定快取容量。
  - 更新高效能推論預設設定，支援 Paddle MKL-DNN 加速。優化高效能推論自動設定邏輯，支援更智慧的設定選擇。
  - 調整預設裝置取得邏輯，考量環境中安裝的 Paddle 框架對運算裝置的實際支援情況，使程式行為更符合直覺。
  - 新增 PP-OCRv5 的 Android 端範例，[詳情](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/on_device_deployment.html)。

- **錯誤修復：**
  - 修復 PP-StructureV3 部分 CLI 參數不生效的問題。
  - 修復部分情況下 `export_paddlex_config_to_yaml` 無法正常運作的問題。
  - 修復 save_path 實際行為與文件描述不符的問題。
  - 修復基礎服務化部署在使用 MKL-DNN 時可能出現的多執行緒錯誤。
  - 修復 Latex-OCR 模型的影像預處理通道順序錯誤。
  - 修復文字辨識模組儲存視覺化影像的通道順序錯誤。
  - 修復 PP-StructureV3 中表格視覺化結果通道順序錯誤。
  - 修復 PP-StructureV3 產線中極特殊情況下，計算 overlap_ratio 時，變數溢位問題。

- **文件優化：**
  - 更新文件中對 `enable_mkldnn` 參數的說明，使其更準確地描述程式的實際行為。
  - 修復文件中對 `lang` 和 `ocr_version` 參數描述的錯誤。
  - 補充透過 CLI 匯出產線設定檔案的說明。
  - 修復 PP-OCRv5 效能資料表格中的欄位缺失問題。
  - 潤飾 PP-StructureV3 在不同設定下的 benchmark 指標。

- **其他：**
  - 放寬 numpy、pandas 等依賴項的版本限制，恢復對 Python 3.12 的支援。

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
   2. 💻 原生支援**ERNIE 4.5**，並相容 PaddleNLP、Ollama、vLLM 等工具部署的大型模型。
   3. 🤝 整合 [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2)，支援印刷體、手寫體、印章、表格、圖表等複雜文件元素的資訊擷取與理解。

[更多日誌](https://paddlepaddle.github.io/PaddleOCR/latest/update/update.html)

</details>

## ⚡ 快速入門
### 1. 線上體驗
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. 本機安裝

請參考[安裝指南](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)完成 **PaddlePaddle 3.0** 的安裝，然後安裝 paddleocr。

```bash
# 如果你只想使用基本的文字識別功能（返回文字的座標和內容，包括 PP-OCR 系列）
python -m pip install paddleocr
# 如果你想使用所有功能，如文檔解析、文檔理解、文檔翻譯、關鍵資訊提取等
# python -m pip install "paddleocr[all]"
```

自 3.2.0 版本起，除了上述的 all 依賴組合外，PaddleOCR 也支持指定其他依賴組合來安裝部分附加功能。PaddleOCR 提供的所有依賴組合如下表所示：

| 依賴組合名稱 | 對應功能 |
| - | - |
| `doc-parser` | 文檔解析：可以從文檔中抽取表格、公式、印章、圖片等版面元素，包括 PP-StructureV3 等模型 |
| `ie` | 資訊抽取：可以從文檔中抽取姓名、日期、地址、金額等關鍵資訊，包括 PP-ChatOCRv4 等模型 |
| `trans` | 文檔翻譯：可以將文檔翻譯成其他語言，包括 PP-DocTranslation 等模型 |
| `all` | 所有功能 |

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

## 🧩 更多特性

- 將模型轉換為 ONNX 格式：[取得 ONNX 模型](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/obtaining_onnx_models.html)
- 使用 OpenVINO、ONNX Runtime、TensorRT 等引擎加速推論，或使用 ONNX 格式模型執行推論：[高效能推論](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/high_performance_inference.html)
- 使用多卡、多進程加速推論：[產線平行推論](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/instructions/parallel_inference.html)
- 在 C++、C#、Java 等語言編寫的應用中整合 PaddleOCR：[服務化部署](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/deployment/serving.html)。

## ⛰️ 進階指南
- [PP-OCRv5 使用教學](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [PP-StructureV3 使用教學](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [PP-ChatOCRv4 使用教學](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 效果展示

<div align="center">
  <p>
       <img width="100%" src="../docs/images/demo.gif" alt="PP-OCRv5 Demo">
  </p>
</div>

<div align="center">
  <p>
      <img width="100%" src="../docs/images/blue_v3.gif" alt="PP-StructureV3 Demo">
  </p>
</div>

## 🌟 不要錯過最新資訊

⭐ **請給這個倉庫加星，以便第一時間獲取包含強大 OCR 及文件分析功能的精彩更新和新版本發布！** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr.en.gif" alt="Star-Project">
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
| [更多專案](../awesome_projects.md) | [更多基於PaddleOCR的項目](../awesome_projects.md) |

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
