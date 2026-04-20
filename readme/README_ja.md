
<div align="center">
  <p>
      <img width="800" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Banner.png" alt="Star履歴">
  </p>



<h3>世界をリードするOCRツールキット & ドキュメントAIエンジン</h3>

[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | 日本語 | [한국어](./README_ko.md) | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | [العربية](./README_ar.md)

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






**PaddleOCRは、ドキュメントや画像を業界最高水準の精度で構造化されたLLM対応データ（JSON/Markdown）に変換します。70,000以上のStarを獲得し、Dify、RAGFlow、Cherry Studioなどの一流プロジェクトで採用されているPaddleOCRは、インテリジェントなRAGおよびエージェントアプリケーション構築の基盤です。**


## 🚀 主な機能

### 📄 インテリジェントドキュメント解析（LLM対応）
> *LLM時代に向けて、雑然とした視覚データを構造化データに変換*

* **最先端のドキュメントVLM**: 業界をリードする軽量視覚言語モデル **PaddleOCR-VL-1.5（0.9B）** を搭載。**歪み、スキャン、画面撮影、照明、傾き**という5つの主要な「実環境」課題にわたる複雑なドキュメント解析に優れ、**Markdown**および**JSON**形式の構造化出力に対応しています。
* **構造認識型変換**: **PP-StructureV3**を活用し、複雑なPDFや画像を**Markdown**または**JSON**にシームレスに変換します。PaddleOCR-VLシリーズモデルとは異なり、テーブルセル座標、テキスト座標などのより詳細な座標情報を提供します。
* **本番環境対応の効率性**: 超小型フットプリントで商用レベルの精度を実現。公開ベンチマークで多くのクローズドソースソリューションを凌駕しつつ、エッジ/クラウドデプロイメントに対してリソース効率を維持します。

### 🔍 汎用テキスト認識（シーンOCR）
> *高速・多言語テキスト検出のグローバルスタンダード*

* **100以上の言語をサポート**: 広範なグローバル言語ライブラリのネイティブ認識。**PP-OCRv5**の単一モデルソリューションは、多言語混在ドキュメント（中国語、英語、日本語、ピンインなど）をエレガントに処理します。
* **複雑な要素への対応力**: 標準的なテキスト認識を超え、身分証明書、街頭風景、書籍、産業部品など、幅広い環境での**自然シーンテキスト検出**をサポートします。
* **性能の飛躍的向上**: PP-OCRv5は前バージョンと比較して**13%の精度向上**を達成し、PaddleOCRの代名詞である「極限の効率性」を維持しています。

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR アーキテクチャ">
  </p>
</div>

### 🛠️ 開発者中心のエコシステム
* **シームレスな統合**: AIエージェントエコシステムの最良の選択肢 ── **Dify、RAGFlow、Pathway、Cherry Studio**と深く統合されています。
* **LLMデータフライホイール**: 高品質データセットを構築する完全なパイプラインを提供し、大規模言語モデルのファインチューニングのための持続可能な「データエンジン」を実現します。
* **ワンクリックデプロイ**: さまざまなハードウェアバックエンド（NVIDIA GPU、Intel CPU、Kunlunxin XPU、各種AIアクセラレータ）をサポートします。


## 📣 最新情報

### 🔥 PaddleOCR v3.5.0 リリース：より柔軟な推論バックエンドと、より充実したドキュメント出力
* **柔軟な推論バックエンド**: Paddleの静的グラフ、動的グラフ、Transformersをシームレスに切り替え可能。Hugging Face エコシステムに深く対応し、主要20モデルがTransformersを推論バックエンドとしてサポート。
* **Office文書をMarkdownに変換**: Word、Excel、PowerPoint などの一般的な文書形式を Markdown に変換可能。
* **解析結果の DOCX 出力**: `PaddleOCR-VL` シリーズ、`PP-StructureV3`、`PP-DocTranslation` で、解析結果を DOCX として出力できるようになり、Microsoft Word での閲覧・編集が容易に。
* **公式ブラウザ推論 SDK**: 公式ブラウザ推論 SDK `PaddleOCR.js` を公開し、ブラウザ上で `PP-OCRv5` を実行可能。

<details>
<summary><strong>2026.01.29: PaddleOCR 3.4.0リリース</strong></summary>
* **PaddleOCR-VL-1.5（最先端の0.9B VLM）**: ドキュメント解析のための最新フラッグシップモデルが公開されました！
    * **OmniDocBenchで94.5%の精度**: トップクラスの汎用大規模モデルや専門ドキュメントパーサーを凌駕。
    * **実環境でのロバスト性**: 非定型形状位置決定のための**PP-DocLayoutV3**アルゴリズムを初めて導入し、*傾き、歪み、スキャン、照明、画面撮影*の5つの困難なシナリオに対応。
    * **機能拡張**: **印鑑認識**、**テキスト検出**をサポートし、**111言語**（中国のチベット文字やベンガル文字を含む）に対応拡大。
    * **長文ドキュメントへの対応**: ページをまたがるテーブルの自動結合および階層的な見出し識別をサポート。
    * **今すぐ試す**: [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)または[公式ウェブサイト](https://www.paddleocr.com)で利用可能です。

</details>

<details>
<summary><strong>2025.10.16: PaddleOCR 3.3.0リリース</strong></summary>

- PaddleOCR-VLをリリース:
    - **モデル紹介**:
        - **PaddleOCR-VL**はドキュメント解析に特化した最先端かつリソース効率の高いモデルです。コアコンポーネントであるPaddleOCR-VL-0.9Bは、NaViTスタイルの動的解像度ビジュアルエンコーダとERNIE-4.5-0.3B言語モデルを統合したコンパクトながらも強力な視覚言語モデル（VLM）であり、正確な要素認識を実現します。**この革新的なモデルは109言語を効率的にサポートし、複雑な要素（テキスト、テーブル、数式、チャートなど）の認識に優れつつ、リソース消費を最小限に抑えます**。広く使用されている公開ベンチマークおよび社内ベンチマークでの包括的な評価を通じて、PaddleOCR-VLはページレベルのドキュメント解析と要素レベルの認識の両方で最先端の性能を達成しています。既存のソリューションを大幅に上回り、トップクラスのVLMに対して高い競争力を示し、高速な推論速度を提供します。これらの強みにより、実世界のシナリオへの実践的なデプロイメントに非常に適しています。モデルは[HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL)で公開されています。ぜひダウンロードしてお使いください！詳細情報は[PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html)をご覧ください。

    - **主要機能**:
        - **コンパクトかつ強力なVLMアーキテクチャ**: リソース効率の高い推論に特化した新しい視覚言語モデルを提案し、要素認識において卓越した性能を実現しました。NaViTスタイルの動的高解像度ビジュアルエンコーダと軽量なERNIE-4.5-0.3B言語モデルを統合することで、モデルの認識能力とデコード効率を大幅に向上させました。この統合により、計算負荷を削減しつつ高い精度を維持し、効率的で実用的なドキュメント処理アプリケーションに適しています。
        - **ドキュメント解析における最先端性能**: PaddleOCR-VLはページレベルのドキュメント解析と要素レベルの認識の両方で最先端の性能を達成しています。既存のパイプラインベースのソリューションを大幅に上回り、ドキュメント解析における主要な視覚言語モデル（VLM）に対して高い競争力を示しています。さらに、テキスト、テーブル、数式、チャートなどの複雑なドキュメント要素の認識に優れ、手書きテキストや歴史文書を含む幅広い種類のコンテンツに対応可能です。これにより、多様なドキュメントタイプやシナリオに対して高い汎用性を発揮します。
        - **多言語サポート**: PaddleOCR-VLは109言語をサポートし、中国語、英語、日本語、ラテン語、韓国語をはじめ、ロシア語（キリル文字）、アラビア語、ヒンディー語（デーヴァナーガリー文字）、タイ語など、異なる文字体系や構造を持つ言語を含む主要なグローバル言語をカバーしています。この幅広い言語対応により、多言語およびグローバルなドキュメント処理シナリオへの適用性が大幅に向上しています。

- PP-OCRv5多言語認識モデルをリリース:
    - ラテン文字認識の精度とカバレッジを改善。キリル文字、アラビア文字、デーヴァナーガリー文字、テルグ文字、タミル文字など他の文字体系のサポートを追加し、109言語の認識をカバー。モデルのパラメータ数はわずか2Mで、一部のモデルの精度は前世代と比較して40%以上向上しています。

</details>


<details>
<summary><strong>2025.08.21: PaddleOCR 3.2.0リリース</strong></summary>

- **モデルの大幅な追加:**
    - PP-OCRv5認識モデルの英語、タイ語、ギリシャ語の学習、推論、デプロイメントを導入。**PP-OCRv5英語モデルは、主要なPP-OCRv5モデルと比較して英語シナリオで11%の改善を達成し、タイ語およびギリシャ語の認識モデルはそれぞれ82.68%および89.28%の精度を実現しています。**

- **デプロイメント機能のアップグレード:**
    - **PaddlePaddleフレームワークバージョン3.1.0および3.1.1を完全サポート。**
    - **PP-OCRv5 C++ローカルデプロイメントソリューションを全面アップグレードし、LinuxとWindowsの両方をサポート。Python実装と同等の機能および精度を実現。**
    - **高性能推論がCUDA 12をサポートし、Paddle InferenceまたはONNX Runtimeバックエンドを使用した推論が可能。**
    - **高安定性サービス指向デプロイメントソリューションが完全にオープンソース化され、ユーザーが必要に応じてDockerイメージやSDKをカスタマイズ可能。**
    - 高安定性サービス指向デプロイメントソリューションは、手動でHTTPリクエストを構築して呼び出すこともサポートしており、任意のプログラミング言語でクライアント側コードの開発が可能です。

- **ベンチマークサポート:**
    - **全プロダクションラインがきめ細かいベンチマークをサポートし、エンドツーエンドの推論時間だけでなく、レイヤーごとおよびモジュールごとのレイテンシデータの測定が可能となり、性能分析を支援します。ベンチマーク機能のセットアップと使用方法は[こちら](docs/version3.x/pipeline_usage/instructions/benchmark.en.md)をご覧ください。**
    - **主要なハードウェアでよく使用される構成の推論レイテンシやメモリ使用量などの重要な指標を含むようドキュメントを更新し、ユーザーにデプロイメントの参考情報を提供。**

- **バグ修正:**
    - モデル学習時のログ保存失敗の問題を解決。
    - 数式モデルのデータ拡張コンポーネントをアップグレードし、albumentations依存関係の新しいバージョンとの互換性を確保。マルチプロセスシナリオでtokenizersパッケージ使用時のデッドロック警告を修正。
    - PP-StructureV3設定ファイルにおけるスイッチ動作（例：`use_chart_parsing`）の他のパイプラインとの不整合を修正。

- **その他の改善:**
    - **コア依存関係とオプション依存関係を分離。基本的なテキスト認識には最小限のコア依存関係のみが必要で、ドキュメント解析や情報抽出のための追加依存関係は必要に応じてインストール可能。**
    - **WindowsでNVIDIA RTX 50シリーズグラフィックスカードのサポートを有効化。対応するPaddlePaddleフレームワークバージョンについては[インストールガイド](docs/version3.x/installation.en.md)を参照してください。**
    - **PP-OCRシリーズモデルが単一文字座標の返却をサポート。**
    - AIStudio、ModelScopeなどのモデルダウンロードソースを追加し、ユーザーがモデルダウンロードのソースを指定可能に。
    - PP-Chart2Tableモジュールによるチャートからテーブルへの変換サポートを追加。
    - ドキュメントの説明を最適化し、使いやすさを向上。
</details>


[更新履歴](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>


## 🚀 クイックスタート

### ステップ1：オンラインで試す
PaddleOCR公式ウェブサイトでは、インタラクティブな**体験センター**と**API**を提供しています。セットアップ不要、ワンクリックで体験できます。

👉 [公式ウェブサイトへアクセス](https://www.paddleocr.com)

### ステップ2：ローカルデプロイメント
ローカルでの使用については、ニーズに応じて以下のドキュメントを参照してください：

- **PP-OCRシリーズ**: [PP-OCRドキュメント](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)を参照
- **PaddleOCR-VLシリーズ**: [PaddleOCR-VLドキュメント](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)を参照
- **PP-StructureV3**: [PP-StructureV3ドキュメント](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html)を参照
- **その他の機能**: [その他の機能ドキュメント](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/pipeline_overview.html)を参照


## 🧩 その他の機能

- モデルをONNX形式に変換: [ONNXモデルの取得](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html)
- OpenVINO、ONNX Runtime、TensorRTなどのエンジンを使用した推論高速化、またはONNX形式モデルによる推論: [高性能推論](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html)
- マルチGPUおよびマルチプロセスによる推論高速化: [パイプラインの並列推論](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html)
- PaddleOCRをC++、C#、Javaなどで書かれたアプリケーションに統合: [サービスデプロイメント](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html)

## 🔄 実行結果の概要

### PP-OCRv5

<div align="center">
  <p>
       <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-OCRv5_demo.gif" alt="PP-OCRv5 デモ">
  </p>
</div>



### PP-StructureV3

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-StructureV3_demo.gif" alt="PP-StructureV3 デモ">
  </p>
</div>

### PaddleOCR-VL

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PaddleOCR-VL_demo.gif" alt="PaddleOCR-VL デモ">
  </p>
</div>


## ✨ 最新情報をチェック

⭐ **このリポジトリにStarを付けて、強力なOCRおよびドキュメント解析機能を含むエキサイティングなアップデートや新リリースを見逃さないようにしましょう！** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr2.en.gif" alt="プロジェクトにStarを付ける">
  </p>
</div>


## 👩‍👩‍👧‍👦 コミュニティ

<div align="center">

| PaddlePaddle WeChat公式アカウント | 技術ディスカッショングループに参加 |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |
</div>


## 😃 PaddleOCRを活用した素晴らしいプロジェクト
PaddleOCRが今日あるのは、素晴らしいコミュニティのおかげです！💗 長年のパートナー、新たな協力者、そしてPaddleOCRに情熱を注いでくださったすべての皆様に心から感謝いたします。名前を挙げきれなかった方も含めて、皆様のサポートが私たちの原動力です！

<div align="center">

| プロジェクト名 | 説明 |
| ------------ | ----------- |
| [Dify](https://github.com/langgenius/dify) <a href="https://github.com/langgenius/dify"><img src="https://img.shields.io/github/stars/langgenius/dify"></a>|エージェントワークフロー開発のためのプロダクション対応プラットフォーム。|
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|深いドキュメント理解に基づくRAGエンジン。|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|ストリーム処理、リアルタイム分析、LLMパイプライン、RAG向けのPython ETLフレームワーク。|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|マルチタイプドキュメントからMarkdownへの変換ツール。|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|無料・オープンソースのバッチオフラインOCRソフトウェア。|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|複数のLLMプロバイダーをサポートするデスクトップクライアント。|
| [haystack](https://github.com/deepset-ai/haystack)<a href="https://github.com/deepset-ai/haystack"><img src="https://img.shields.io/github/stars/deepset-ai/haystack"></a> |カスタマイズ可能なプロダクション対応LLMアプリケーションを構築するためのAIオーケストレーションフレームワーク。|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |純粋なビジョンベースのGUIエージェント向け画面解析ツール。|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |あらゆるものに基づく質問応答。|
| [その他のプロジェクトを見る](./awesome_projects.md) | [PaddleOCRに基づくその他のプロジェクト](./awesome_projects.md)|
</div>

## 👩‍👩‍👧‍👦 コントリビューター

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>
</div>

## 🌟 Star

<div align="center">
  <p>
      <img width="800" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="Star履歴">
  </p>
</div>


## 📄 ライセンス
このプロジェクトは[Apache 2.0ライセンス](LICENSE)の下で公開されています。

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
