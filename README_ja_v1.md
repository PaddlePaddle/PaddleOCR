<div align="center">
  <p>
      <img width="100%" src="./docs/images/Banner.png" alt="PaddleOCR Banner"></a>
  </p>

<!-- language -->
[中文](./readme_c.md)| [English](./README_en.md) | 日本語

<!-- icon -->

[![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![Downloads](https://img.shields.io/pypi/dm/paddleocr)](https://pypi.org/project/PaddleOCR/)
![python](https://img.shields.io/badge/python-3.8～3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)

[![Website](https://img.shields.io/badge/Website-PaddleOCR-blue?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmmRkdj0AAAAASUVORK5CYII=)](https://www.paddleocr.ai/)
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

</div>

## 🚀 はじめに
PaddleOCRは、その最先端のアルゴリズムと実世界での応用実績により、初版リリース以来、学界、産業界、研究コミュニティから広く称賛を得ています。Umi-OCR、OmniParser、MinerU、RAGFlowなどの人気オープンソースプロジェクトに既に採用されており、世界中の開発者にとって頼れるOCR（光学文字認識）ツールキットとなっています。

2025年5月20日、PaddlePaddleチームは**PaddlePaddle 3.0**フレームワークの公式リリースと完全互換のPaddleOCR 3.0を発表しました。このアップデートでは、**テキスト認識精度**がさらに向上し、**複数テキストタイプ認識**と**手書き文字認識**のサポートが追加され、**複雑なドキュメントの高精度解析**に対する大規模モデル応用の高まる需要に応えています。**ERNIE 4.5T**と組み合わせることで、キー情報抽出の精度が大幅に向上します。PaddleOCR 3.0は、**KUNLUNXIN**や**Ascend**といった国産ハードウェアプラットフォームのサポートも導入しています。完全な使用方法については、[PaddleOCR 3.0ドキュメント](https://paddlepaddle.github.io/PaddleOCR/latest/ja/index.html)を参照してください。

PaddleOCR 3.0の3つの主要な新機能：
- ユニバーサルシーンテキスト認識モデル [PP-OCRv5](./docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.ja.md): 1つのモデルで5つの異なるテキストタイプと複雑な手書き文字を処理します。全体の認識精度は前世代に比べて13パーセントポイント向上しました。[オンラインデモ](https://aistudio.baidu.com/community/app/91660/webUI)

- 汎用ドキュメント解析ソリューション [PP-StructureV3](./docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.ja.md): 多様なレイアウト、多シーンのPDFを高精度で解析し、公開ベンチマークで多くのオープンソースおよびクローズドソースソリューションを上回ります。[オンラインデモ](https://aistudio.baidu.com/community/app/518494/webUI)

- インテリジェントドキュメント理解ソリューション [PP-ChatOCRv4](./docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.ja.md): WenXin大規模モデル4.5Tをネイティブで活用し、前世代より15パーセントポイント高い精度を達成。[オンラインデモ](https://aistudio.baidu.com/community/app/518493/webUI)

優れたモデルライブラリの提供に加えて、PaddleOCR 3.0はモデルのトレーニング、推論、サービス展開をカバーする使いやすいツールも提供しており、開発者はAIアプリケーションを迅速に本番環境に導入できます。
<div align="center">
  <p>
      <img width="100%" src="./docs/images/Arch.png" alt="PaddleOCR Architecture"></a>
  </p>
</div>



## 📣 最近の更新

#### **🔥🔥 2025.06.05: PaddleOCR 3.0.1リリース、内容:**

- **一部のモデルとモデル設定の最適化:**
  - PP-OCRv5のデフォルトモデル設定を更新し、検出と認識の両方をmobileからserverモデルに変更しました。ほとんどのシナリオでデフォルトのパフォーマンスを向上させるため、設定のパラメータ`limit_side_len`を736から64に変更しました。
  - 新しいテキスト行方向分類モデル`PP-LCNet_x1_0_textline_ori`を追加し、精度は99.42%です。OCR、PP-StructureV3、PP-ChatOCRv4パイプラインのデフォルトのテキスト行方向分類器をこのモデルに更新しました。
  - テキスト行方向分類モデル`PP-LCNet_x0_25_textline_ori`を最適化し、精度を3.3パーセントポイント向上させ、現在の精度は98.85%です。

- **バージョン3.0.0の一部の問題の最適化と修正、[詳細](https://paddlepaddle.github.io/PaddleOCR/latest/ja/update/update.html)**

🔥🔥2025.05.20: **PaddleOCR v3.0**公式リリース、内容:
- **PP-OCRv5**: 全シーン対応高精度テキスト認識モデル - 画像/PDFから瞬時にテキスト化。
   1. 🌐 単一モデルで**5つ**のテキストタイプをサポート - **簡体字中国語、繁体字中国語、簡体字中国語ピンイン、英語**、**日本語**を単一モデル内でシームレスに処理。
   2. ✍️ **手書き認識**の改善: 複雑な筆記体や非標準的な手書き文字の認識が大幅に向上。
   3. 🎯 PP-OCRv4に比べて**13ポイントの精度向上**、様々な実世界のシナリオで最先端のパフォーマンスを達成。

- **PP-StructureV3**: 汎用ドキュメント解析 – 実世界のシナリオでSOTAの画像/PDF解析を解放！ 
   1. 🧮 **高精度なマルチシーンPDF解析**、OmniDocBenchベンチマークで多くのオープンソースおよびクローズドソースソリューションをリード。
   2. 🧠 **印章認識**、**図から表への変換**、**ネストされた数式/画像を含む表の認識**、**縦書き文書の解析**、**複雑な表構造の分析**などの専門的な機能。

- **PP-ChatOCRv4**: インテリジェントな文書理解 – 画像/PDFからテキストだけでなく、キー情報を抽出。
   1. 🔥 PDF/PNG/JPGファイルからのキー情報抽出において、前世代に比べて**15ポイントの精度向上**。
   2. 💻 **ERNIE4.5 Turbo**をネイティブサポートし、PaddleNLP、Ollama、vLLMなどを介した大規模モデル展開との互換性あり。
   3. 🤝 [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2)を統合し、印刷テキスト、手書き文字、印章、表、図、その他の複雑な文書における一般的な要素の抽出と理解を可能に。

<details>
   <summary><strong>更新履歴</strong></summary>


- 🔥🔥2025.03.07: **PaddleOCR v2.10**リリース、内容:

  - **12種類の自社開発新モデル:**
    - **[レイアウト検出シリーズ](https://paddlepaddle.github.io/PaddleX/latest/ja/module_usage/tutorials/ocr_modules/layout_detection.html)**(3モデル): PP-DocLayout-L、M、S -- 多様な文書形式（論文、レポート、試験、書籍、雑誌、契約書など）にわたる23種類の一般的なレイアウトタイプを英語と中国語で検出可能。最大**90.4%のmAP@0.5**を達成し、軽量な機能で毎秒100ページ以上を処理可能。
    - **[数式認識シリーズ](https://paddlepaddle.github.io/PaddleX/latest/ja/module_usage/tutorials/ocr_modules/formula_recognition.html)**(2モデル): PP-FormulaNet-L、S -- 50,000以上のLaTeX表現の認識をサポートし、印刷された数式と手書きの数式の両方に対応。PP-FormulaNet-Lは同等のモデルより**6%高い精度**を提供し、PP-FormulaNet-Sは同等の精度を維持しつつ16倍高速。
    - **[表構造認識シリーズ](https://paddlepaddle.github.io/PaddleX/latest/ja/module_usage/tutorials/ocr_modules/table_structure_recognition.html)**(2モデル): SLANeXt_wired、SLANeXt_wireless -- 複雑な表認識においてSLANet_plusより**6%の精度向上**を実現した新開発モデル。
    - **[表分類](https://paddlepaddle.github.io/PaddleX/latest/ja/module_usage/tutorials/ocr_modules/table_classification.html)**(1モデル): 
PP-LCNet_x1_0_table_cls -- 有線および無線テーブル用の超軽量分類器。

[詳細はこちら](https://paddlepaddle.github.io/PaddleOCR/latest/ja/update.html)

</details>

## ⚡ クイックスタート
### 1. オンラインデモの実行
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. インストール

[インストールガイド](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)を参照してPaddlePaddleをインストールした後、PaddleOCRツールキットをインストールします。

```bash
# paddleocrをインストール
pip install paddleocr
```

### 3. CLIによる推論の実行
```bash
# PP-OCRv5推論の実行
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False  

# PP-StructureV3推論の実行
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# まずQianfan APIキーを取得し、次にPP-ChatOCRv4推論を実行
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# "paddleocr ocr"に関する詳細情報を取得
paddleocr ocr --help
```

### 4. APIによる推論の実行
**4.1 PP-OCRv5の例**
```python
# PaddleOCRインスタンスの初期化
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# サンプル画像でOCR推論を実行
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# 結果を可視化し、JSON形式で保存
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary><strong>4.2 PP-StructureV3の例</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3()

# 画像の場合
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
    )

# 結果を可視化し、JSON形式とMarkdown形式で保存
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output")           
```

</details>

<details>
   <summary><strong>4.3 PP-ChatOCRv4の例</strong></summary>

```python
from paddleocr import PPChatOCRv4Doc

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # あなたのapi_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # あなたのapi_key
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
# マルチモーダル大規模モデルを使用する場合、ローカルmllmサービスを開始する必要があります。ドキュメントを参照してください: https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.md を参照してデプロイメントを行い、mllm_chat_bot_config設定を更新します。
if use_mllm:
    mllm_chat_bot_config = {
        "module_name": "chat_bot",
        "model_name": "PP-DocBee",
        "base_url": "http://127.0.0.1:8080/",  # あなたのローカルmllmサービスURL
        "api_type": "openai",
        "api_key": "api_key",  # あなたのapi_key
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

### 5. 国産AIアクセラレータ
- [Huawei Ascend](https://paddlepaddle.github.io/PaddleOCR/latest/ja/version3.x/other_devices_support/paddlepaddle_install_NPU.html)
- [KUNLUNXIN](https://paddlepaddle.github.io/PaddleOCR/latest/ja/version3.x/other_devices_support/paddlepaddle_install_XPU.html)

## ⛰️ 上級チュートリアル
- [PP-OCRv5 チュートリアル](https://paddlepaddle.github.io/PaddleOCR/latest/ja/version3.x/pipeline_usage/OCR.html)
- [PP-StructureV3 チュートリアル](https://paddlepaddle.github.io/PaddleOCR/latest/ja/version3.x/pipeline_usage/PP-StructureV3.html)
- [PP-ChatOCRv4 チュートリアル](https://paddlepaddle.github.io/PaddleOCR/latest/ja/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 実行結果のクイック概要

<div align="center">
  <p>
     <img width="100%" src="./docs/images/demo.gif" alt="PP-OCRv5 Demo"></a>
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="./docs/images/blue_v3.gif" alt="PP-StructureV3 Demo"></a>
  </p>
</div>

## 👩‍👩‍👧‍👦 コミュニティ

| PaddlePaddle WeChat公式アカウント |  技術交流グループに参加 |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |


## 😃 PaddleOCRを活用した素晴らしいプロジェクト
PaddleOCRは、その素晴らしいコミュニティなしには今日の姿はありませんでした！💗 長年のパートナー、新しい協力者、そして情熱を注いでくださったすべての方々に心から感謝します—名前を挙げているかどうかにかかわらず。あなたのサポートが私たちの原動力です！

| プロジェクト名 | 説明 |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|深層文書理解に基づくRAGエンジン。|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|複数タイプの文書をMarkdownに変換するツール|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|無料、オープンソース、バッチ処理対応のオフラインOCRソフトウェア。|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |純粋なビジョンベースのGUIエージェント向け画面解析ツール。|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |任意の内容に基づく質問応答システム。|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|複雑で多様なPDF文書から高品質なコンテンツを効率的に抽出するために設計された強力なオープンソースツールキット。|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |画面上のテキストを認識し、翻訳してリアルタイムで翻訳結果を表示します。|
| [その他のプロジェクト](./awesome_projects.md) | [PaddleOCRをベースにしたその他のプロジェクト](./awesome_projects.md)|

## 👩‍👩‍👧‍👦 貢献者

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>


## 🌟 Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)


## 📄 ライセンス
このプロジェクトは[Apache 2.0ライセンス](LICENSE)の下で公開されています。

## 🎓 引用

```
@misc{paddleocr2020,
title={PaddleOCR, Awesome multilingual OCR toolkits based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleOCR}},
year={2020}
}
```

</rewritten_file>
