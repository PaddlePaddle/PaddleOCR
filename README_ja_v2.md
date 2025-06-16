<div align="center">
  <p>
      <img width="100%" src="./docs/images/Banner_cn.png" alt="PaddleOCR Banner"></a>
  </p>

<!-- language -->
日本語 | [中文](./README.md) | [English](./README_en.md)

<!-- icon -->

[![stars](https.img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![Downloads](https://img.shields.io/pypi/dm/paddleocr)](https://pypi.org/project/PaddleOCR/)
![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)


[![Website](https://img.shields.io/badge/Website-PaddleOCR-blue?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmmRkdj0AAAAASUVORK5CYII=)](https://www.paddleocr.ai/)
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

</div>

## 🚀 概要
PaddleOCRはリリース以来、最先端のアルゴリズムと産業界での実践的な応用により、学術界および産業界から広く支持されています。Umi-OCR、OmniParser、MinerU、RAGFlowなど、多くの著名なオープンソースプロジェクトで採用されており、オープンソースOCR分野における開発者の第一選択肢となっています。2025年5月20日、PaddlePaddleチームは**PaddleOCR 3.0**をリリースしました。これは**PaddlePaddleフレームワーク3.0正式版**に完全対応しており、文字認識精度をさらに向上させ、複数文字タイプ認識と手書き文字認識をサポートし、大規模モデル応用における複雑なドキュメントの高精度解析という高まる需要に応えます。**ERNIE 4.5 Turbo**と組み合わせることで、キー情報抽出の精度が大幅に向上します。また、**KUNLUNXIN**や**Ascend**などのハードウェアプラットフォームのサポートも追加されました。完全なドキュメントについては、[PaddleOCR 3.0 ドキュメント](https://paddlepaddle.github.io/PaddleOCR/latest/)をご参照ください。

PaddleOCR 3.0は**3つの新しい主要機能**を追加しました：
- 全シーン対応文字認識モデル[PP-OCRv5](docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)：単一モデルで5種類の文字タイプと複雑な手書き文字の認識をサポート。全体的な認識精度は前世代に比べて**13パーセントポイント向上**。[オンライン体験](https://aistudio.baidu.com/community/app/91660/webUI)
- 汎用ドキュメント解析ソリューション[PP-StructureV3](docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.md)：複数シーン、複数レイアウトのPDFの高精度解析をサポートし、公開ベンチマークで**多くのオープンソースおよびクローズドソースソリューションをリード**。[オンライン体験](https://aistudio.baidu.com/community/app/518494/webUI)
- インテリジェントドキュメント理解ソリューション[PP-ChatOCRv4](docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.md)：ERNIE 4.5 Turboをネイティブサポートし、精度は前世代に比べて**15パーセントポイント向上**。[オンライン体験](https://aistudio.baidu.com/community/app/518493/webUI)

PaddleOCR 3.0は、優れたモデルライブラリを提供するだけでなく、モデルのトレーニング、推論、サービス展開をカバーする使いやすいツールも提供し、開発者が迅速にAIアプリケーションを実用化できるようにサポートします。
<div align="center">
  <p>
      <img width="100%" src="./docs/images/Arch_cn.png" alt="PaddleOCR Architecture"></a>
  </p>
</div>


## 📣 最新情報
🔥🔥2025.06.05: **PaddleOCR 3.0.1** リリース。内容は次の通りです：

- **一部のモデルとモデル設定の最適化：**
  - PP-OCRv5のデフォルトモデル設定を更新。検出と認識モデルをmobileからserverモデルに変更。ほとんどのシーンでのデフォルト性能を向上させるため、設定パラメータ`limit_side_len`を736から64に変更。
  - 新しいテキスト行方向分類モデル`PP-LCNet_x1_0_textline_ori`（精度99.42%）を追加。OCR、PP-StructureV3、PP-ChatOCRv4パイプラインのデフォルトテキスト行方向分類器をこのモデルに変更。
  - テキスト行方向分類モデル`PP-LCNet_x0_25_textline_ori`を最適化し、精度が3.3パーセントポイント向上し、現在の精度は98.85%。
- **3.0.0バージョンの一部の問題を最適化・修正。[詳細](https://paddlepaddle.github.io/PaddleOCR/latest/update/update.html)**

🔥🔥2025.05.20: **PaddleOCR 3.0** 公式リリース。内容は次の通りです：
- **PP-OCRv5**: 全シーン対応高精度文字認識

   1. 🌐 単一モデルで**5種類**の文字タイプ（**簡体字中国語**、**繁体字中国語**、**中国語ピンイン**、**英語**、**日本語**）をサポート。
   2. ✍️ 複雑な**手書き文字**の認識をサポート：複雑な筆記体や非標準的な手書き文字の認識性能が大幅に向上。
   3. 🎯 全体的な認識精度の向上 - 様々な応用シーンでSOTAの精度を達成。前バージョンのPP-OCRv4と比較して、認識精度が**13パーセントポイント向上**！

- **PP-StructureV3**: 汎用ドキュメント解析ソリューション

   1. 🧮 複数シーンのPDFの高精度解析をサポートし、OmniDocBenchベンチマークで**多くのオープンソースおよびクローズドソースソリューションをリード**。
   2. 🧠 多岐にわたる専門能力：**印鑑認識**、**グラフから表への変換**、**ネストされた数式/画像を含む表の認識**、**縦書きテキストの解析**、**複雑な表構造の分析**など。


- **PP-ChatOCRv4**: インテリジェントドキュメント理解ソリューション
   1. 🔥 ドキュメント画像（PDF/PNG/JPG）からのキー情報抽出精度が前世代に比べて**15パーセントポイント向上**！
   2. 💻 **ERNIE 4.5 Turbo**をネイティブサポートし、PaddleNLP、Ollama、vLLMなどのツールでデプロイされた大規模モデルとも互換性あり。
   3. 🤝 [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2)を統合し、印刷文字、手書き文字、印鑑情報、表、グラフなど、複雑なドキュメントに含まれる一般的な情報の抽出と理解をサポート。


## ⚡ クイックスタート
### 1. オンライン体験
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. ローカルインストール

[インストールガイド](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html)を参考に**PaddlePaddle 3.0**をインストールした後、paddleocrをインストールしてください。

```bash
# paddleocrをインストール
pip install paddleocr
```

### 3. コマンドラインによる推論
```bash
# PP-OCRv5 推論の実行
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False 

# PP-StructureV3 推論の実行
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# PP-ChatOCRv4 推論の実行前に、Qianfan APIキーを取得する必要があります
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# "paddleocr ocr" の詳細なパラメータを確認
paddleocr ocr --help
```
### 4. APIによる推論

**4.1 PP-OCRv5 の例**
```python
from paddleocr import PaddleOCR
# PaddleOCRインスタンスを初期化
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)
# サンプル画像に対してOCR推論を実行 
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")
# 結果を可視化し、JSON形式で保存
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary><strong>4.2 PP-StructureV3 の例</strong></summary>

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

# 結果を可視化し、JSON形式で保存
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output") 
```

</details>


<details>
   <summary><strong>4.3 PP-ChatOCRv4 の例</strong></summary>

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
# マルチモーダル大規模モデルを使用する場合、ローカルでmllmサービスを起動する必要があります。ドキュメント：https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.md を参照してデプロイし、mllm_chat_bot_configを設定してください。
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


### 5. **特定ハードウェアのサポート**
- [KUNLUNXIN インストールガイド](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_XPU.html)
- [Ascend インストールガイド](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_NPU.html)
  
## ⛰️ 上級チュートリアル
- [PP-OCRv5 チュートリアル](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [PP-StructureV3 チュートリアル](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [PP-ChatOCRv4 チュートリアル](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 実行結果のプレビュー

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

## 👩‍👩‍👧‍👦 開発者コミュニティ

| QRコードをスキャンしてPaddlePaddle公式アカウントをフォロー | QRコードをスキャンして技術交流コミュニティに参加 |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |

## 🏆 PaddleOCRを活用した素晴らしいプロジェクト
PaddleOCRの発展はコミュニティの貢献なしにはありえません！💗すべての開発者、パートナー、貢献者の皆様に心より感謝申し上げます。
| プロジェクト名 | 概要 |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|RAGベースのAIワークフローエンジン|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|複数タイプのドキュメントからMarkdownへの変換ツール|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|オープンソースのバッチ処理対応オフラインOCRソフトウェア|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |純粋なビジョンベースのGUIエージェント向け画面解析ツール|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |あらゆるコンテンツに基づいた質問応答システム|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|効率的に複雑なPDFドキュメントからコンテンツを抽出するツールキット|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |画面上のテキストをリアルタイムで翻訳するツール|
| [その他のプロジェクト](./awesome_projects.md) | |

## 👩‍👩‍👧‍👦 貢献者

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>


## 🌟 Star

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)


## 📄 ライセンス
このプロジェクトは[Apache 2.0 license](LICENSE)の下で公開されています。

## 🎓 学術引用

```
@misc{paddleocr2020,
title={PaddleOCR, Awesome multilingual OCR toolkits based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleOCR}},
year={2020}
}
``` 
