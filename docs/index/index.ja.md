---
comments: true
hide:
  - navigation
  - toc
---

<div align="center">
 <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.9.1/PaddleOCR_log.png" align="middle" width = "600"/>
  <p align="center">
      <a href="https://discord.gg/z9xaRVjdbD"><img src="https://img.shields.io/badge/Chat-on%20discord-7289da.svg?sanitize=true" alt="Chat"></a>
      <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
      <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
      <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
      <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
      <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
      <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
  </p>
</div>

## 紹介

PaddleOCR は、さまざまな言語で、優れた最先端かつ実用的な OCR ツールを作成することを目的とし、ユーザーがより優れたモデルをトレーニングし、実践的に対応できるようになるために役立つAIOCRです。

<div align="center">
    <img src="https://github.com/PaddlePaddle/PaddleOCR/releases/download/v2.8.0/demo.gif" width="800">
</div>

## 📣 最新アップデート

- **🔥2022.8.24 リリース PaddleOCR [release/2.6](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)**
    - [PP-Structurev2](../../ppstructure/)がリリース。機能と使いやすさがアップグレード、中国語のさまざまな文字に適応、 [レイアウトの復旧](../../ppstructure/recovery)  さらに**1 行のコマンドをPDFへ転換、そして Word**に変換可能。
    - [レイアウト分析](../../ppstructure/layout) の最適化：モデルのストレージが 95% 削減、速度が 11 倍向上、平均 CPU 時間コストはわずか 41 ミリ秒です。
    - [表認識](../../ppstructure/table) 最適化：3つの最適化戦略設計、モデルの精度が従来より同時間比が 6% 向上。
    - [キー情報抽出](../../ppstructure/kie) 最適化:視覚に依存しないモデル構造設計、語彙の実態識別精度が 2.8% 向上、関係抽出の精度が 9.1% 向上。

- **🔥2022.7 リリース [OCR scene application collection](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.8/applications/README_en.md)**
    - デジタルチューブ、液晶画面、ナンバー プレート、手書き認識モデル、高精度 SVTR モデルなど、**9つの垂直モデル**をリリース、一般、製造、金融、運輸業界の主要な OCR 垂直アプリケーションをカバー。

- **🔥2022.5.9 リリース PaddleOCR [release/2.5](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5)**
    - [PP-OCRv3](../version2.x/ppocr/overview.en.md#pp-ocrv3)リリース: 同等の速度で、中国語の識別効果は PP-OCRv2 より 5% 向上、英語の識別効果は 11% 向上し、80 言語の多言語モデルの平均認識精度は 5% 以上向上。
    - [PPOCRLabelv2](https://github.com/PFCCLab/PPOCRLabel)リリース: 表認識タスク、キー情報抽出タスク、イレギュラーテキスト画像のアノテーション機能を追加。
    - インタラクティブな電子書籍 [*"OCR に没入"*](../version2.x/ppocr/blog/ocr_book.en.md)、 をリリース。 OCRフルスタック技術の最先端の理論とコードの実践をカバー。

- [もっと](../update/update.en.md)

## 🌟 PaddleOCRとは？

PaddleOCRは、OCRに関連するさまざまな最先端のアルゴリズムに対応する、産業用の機能モデル/ソリューション [PP-OCR](../version2.x/ppocr/overview.en.md) や [PP-Structure](../../ppstructure/README.md) を開発。これに基づき、データの生成、モデルのトレーニング、圧縮、推論、展開の全プロセスを実行可能。

<div align="center">
    <img src="https://user-images.githubusercontent.com/50011306/195771471-fad5eb1d-190d-4a7b-8b0c-0433fb32445f.png">
</div>

## ⚡ 今すぐトライアル

```bash
pip3 install paddlepaddle # for gpu user please install paddlepaddle-gpu
pip3 install paddleocr
paddleocr --image_dir /your/test/image.jpg --lang=japan # change for i18n abbr
```

>Python環境がない場合は [環境の準備](../version2.x/ppocr/environment.en.md)に従ってください。[チュートリアル](#Tutorials) から始めることをお勧めします。

<a name="本"></a>

## 📚 電子書籍：*OCRに入る*

- [OCRに没入](../version2.x/ppocr/blog/ocr_book.en.md)

<a name="コミュニティ"></a>

## 👫コミュニティー

他国の開発者の方は [PaddleOCR Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions) を国際的なコミュニティ プラットフォームとして使用します。みなさんのアイデアや質問がある場合、ここで英語で話し合うことができます。

<a name="対応中国機種一覧"></a>

## 🛠️ シリーズ モデル式一覧

| モデル紹介                                           | モデル名                   | 推奨のシーン | 検出モデル                                             | 認識モデル                                           |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 日本語超軽量 PP-OCRv3 モデル(14.8M) | japan_PP-OCRv3_xx | モバイル & サーバー |[推論モデル](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_infer.tar)/[トレーニングモデル](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar) |[推論モデル](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_infer.tar)/[トレーニングモデル](https://paddleocr.bj.bcebos.com/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar) |
| 英語超軽量PP-OCRv3モデル（13.4M） | en_PP-OCRv3_xx | モバイル & サーバー | [推論モデル](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [トレーニングモデル](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [推論モデル](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv3_mobile_rec_infer.tar) / [トレーニングモデル](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
| 中国語と英語の超軽量 PP-OCRv3 モデル（16.2M）    | ch_PP-OCRv3_xx          | モバイル & サーバー | [推論モデル](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_det_infer.tar) / [トレーニングモデル](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams) | [推論モデル](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_rec_infer.tar) / [トレーニングモデル](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |

- その他のモデルのダウンロード (多言語を含む) については、[PP-OCR シリーズ モデルのダウンロード](../version2.x/ppocr/model_list.en.md)をご参照ください。
- 新しい言語のリクエストについては、 [新しい言語_リクエストのガイドライン](#language_requests)を参照してください。
- 構造文書分析モデルについては、[PP-Structure models](../version2.x/ppstructure/models_list.en.md)をご参照ください。

<a name="チュートリアル"></a>

## 📖 チュートリアル

- [環境の準備](../version2.x/ppocr/environment.en.md)
- [PP-OCR 🔥](../version2.x/ppocr/overview.en.md)
    - [クイックスタート](../version2.x/ppocr/quick_start.en.md)
    - [Model Zoo](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/models_en.md)
    - [トレーニング モデル](../version2.x/ppocr/model_train/training.en.md)
        - [テキスト検出](../version2.x/ppocr/model_train/detection.en.md)
        - [テキスト認識](../version2.x/ppocr/model_train/recognition.en.md)
        - [テキスト方向の分類](../version2.x/ppocr/model_train/angle_class.en.md)
    - モデル圧縮
        - [モデルの量子化](../../deploy/slim/quantization/README_en.md)
        - [モデルの剪裁](../../deploy/slim/prune/README_en.md)
        - [知識の蒸留](../version2.x/ppocr/model_compress/knowledge_distillation.en.md)
    - [推論と展開](../../deploy/README.md)
        - [Python 推論](../version2.x/legacy/python_infer.en.md)
        - [C++ 推論](../version2.x/legacy/cpp_infer.en.md)
        - [サービング](../version2.x/legacy/paddle_server.en.md)
        - [モバイル](../../deploy/lite/readme.md)
        - [Paddle2ONNX](../../deploy/paddle2onnx/readme.md)
        - [PaddleCloud](../../deploy/paddlecloud/README.md)
        - [Benchmark](../version2.x/legacy/benchmark.en.md)
- [PP-Structure 🔥](../../ppstructure/README.md)
    - [クイックスタート](../version2.x/ppstructure/quick_start.en.md)
    - [Model Zoo](../version2.x/ppstructure/models_list.en.md)
    - [トレーニング モデル](../version2.x/ppocr/model_train/training.en.md)
        - [レイアウト分析](../../ppstructure/layout/README.md)
        - [表認識](../../ppstructure/table/README.md)
        - [キー情報抽出](../../ppstructure/kie/README.md)
    - [推論と展開](../../deploy/README.md)
        - [Python 推論](../version2.x/ppstructure/infer_deploy/python_infer.en.md)
        - [C++ 推論](../version2.x/legacy/cpp_infer.en.md)
        - [サービング](../../deploy/hubserving/readme_en.md)
- [アカデミックアリゴリズム](../version2.x/algorithm/overview.en.md)
    - [テキスト検出](../version2.x/algorithm/overview.en.md)
    - [テキスト認識](../version2.x/algorithm/overview.en.md)
    - [エンド・ツー・エンド OCR](../version2.x/algorithm/overview.en.md)
    - [表認識](../version2.x/algorithm/overview.en.md)
    - [キー情報抽出](../version2.x/algorithm/overview.en.md)
    - [PaddleOCR に新しいアルゴリズムを追加する](../version2.x/algorithm/add_new_algorithm.en.md)
- データの注釈と合成
    - [半自動注釈ツール: PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md)
    - [データ合成ツール: Style-Text](https://github.com/PFCCLab/StyleText/blob/main/README.md)
    - [その他のデータ注釈ツール](../data_anno_synth/data_annotation.en.md)
    - [その他のデータ合成ツール](../data_anno_synth/data_synthesis.en.md)
- データセット
    - [一般OCRデータセット(中国語/英語)](../datasets/datasets.en.md)
    - [HandWritten_OCR_Datasets(中国語)](../datasets/handwritten_datasets.en.md)
    - [各種OCRデータセット(多言語対応)](../datasets/vertical_and_multilingual_datasets.en.md)
    - [レイアウト分析](../datasets/layout_datasets.en.md)
    - [表認識](../datasets/table_datasets.en.md)
    - [キー情報抽出](../datasets/kie_datasets.en.md)
- [コード構造](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/tree_en.md)
- [視覚化](#Visualization)
- [コミュニティ](#Community)
- [新言語のリクエスト](#language_requests)
- [よくある質問](../FAQ.en.md)
- [参考文献](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.9/doc/doc_en/reference_en.md)
- [ライセンス](#LICENSE)

<a name="language_requests"></a>

## 🇺🇳 新しい言語リクエストのガイドライン

**新言語モデルをリクエスト**したい場合、[多言語モデルのアップグレードへの投票](https://github.com/PaddlePaddle/PaddleOCR/discussions/7253)で投票してください。投票結果に応じて定期的にモデルがアップグレードされます。**友達を招待して一緒に投票しましょう!**

シナリオに基づいて**新しい言語モデルをトレーニング** する必要がある場合は、[多言語モデル トレーニング プロジェクト](https://github.com/PaddlePaddle/PaddleOCR/discussions/7252) のチュートリアルがデータセットの準備にご利用でき、 プロセス全体を段階的に表示することができます。

元の[多言語 OCR 開発計画](https://github.com/PaddlePaddle/PaddleOCR/issues/1048) には、まだ多くの有用なコーパスと辞書が表示されています

<a name="ビジュアリゼーション"></a>

## 👀 ビジュアリゼーション [more](../version2.x/ppocr/visualization.en.md)

<details open>
<summary>PP-OCRv3 多言語モデル</summary>
<div align="center">
    <img src="../version2.x/ppocr/images/PP-OCRv3/multi_lang/japan_2.jpg" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/multi_lang/korean_1.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-OCRv3 英語 モデル</summary>
<div align="center">
    <img src="../version2.x/ppocr/images/PP-OCRv3/en/en_1.png" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/en/en_2.png" width="800">
</div>
</details>
<details open>
<summary>PP-OCRv3 中国語 モデル</summary>
<div align="center">
    <img src="../version2.x/ppocr/images/PP-OCRv3/ch/PP-OCRv3-pic001.jpg" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/ch/PP-OCRv3-pic002.jpg" width="800">
    <img src="../version2.x/ppocr/images/PP-OCRv3/ch/PP-OCRv3-pic003.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-Structurev2</summary>
1. レイアウト分析＋テーブル認識
<div align="center">
    <img src="../version2.x/ppstructure/images/ppstructure.gif" width="800">
</div>
2. SER (セマンティックエンティティ認識)
<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094456-01a1dd11-1433-4437-9ab2-6480ac94ec0a.png" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808.jpg" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a.png" width="600">
</div>
3. RE (関係抽出)
<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094813-3a8e16cc-42e5-4982-b9f4-0134dfb5688d.png" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb.jpg" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185540080-0431e006-9235-4b6d-b63d-0b3c6e1de48f.jpg" width="600">
</div>
</details>

<a name="ライセンス"></a>

## 📄 ライセンス

このプロジェクトは以下の場所でリリースされています <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/{{PADDLEOCR_GITHUB_REF}}/LICENSE">Apache 2.0 license</a>
