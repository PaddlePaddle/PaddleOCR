English | [简体中文](README.md) | [हिन्दी](./doc/doc_i18n/README_हिन्द.md) | [日本語](./doc/doc_i18n/README_日本語.md) | [한국인](./doc/doc_i18n/README_한국어.md) | [Pу́сский язы́к](./doc/doc_i18n/README_Ру́сский_язы́к.md)

<p align="center">
 <img src="./doc/PaddleOCR_log.png" align="middle" width = "600"/>
<p align="center">
<p align="center">
    <a href="https://discord.gg/z9xaRVjdbD"><img src="https://img.shields.io/badge/Chat-on%20discord-7289da.svg?sanitize=true" alt="Chat"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
</p>

## Introduction

PaddleOCR aims to create multilingual, awesome, leading, and practical OCR tools that help users train better models and apply them into practice.

<div align="center">
    <img src="./doc/imgs_results/PP-OCRv3/en/en_4.png" width="800">
</div>

<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/00006737.jpg" width="800">
</div>

## 🚀 Community
PaddleOCR is being oversight by a [PMC](https://github.com/PaddlePaddle/PaddleOCR/issues/12122). Issues and PRs will be reviewed on a best-effort basis. For a complete overview of PaddlePaddle community, please visit [community](https://github.com/PaddlePaddle/community).

⚠️ Note: The [Issues](https://github.com/PaddlePaddle/PaddleOCR/issues) module is only for reporting program 🐞 bugs, for the rest of the questions, please move to the [Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions). Please note that if the Issue mentioned is not a bug, it will be moved to the Discussions module.

## 📣 Recent updates
- **🔥2023.8.7 Release PaddleOCR[release/2.7](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.7)**
  - Release [PP-OCRv4](./doc/doc_ch/PP-OCRv4_introduction.md), support mobile version and server version
    - PP-OCRv4-mobile：When the speed is comparable, the effect of the Chinese scene is improved by 4.5% compared with PP-OCRv3, the English scene is improved by 10%, and the average recognition accuracy of the 80-language multilingual model is increased by more than 8%.
    - PP-OCRv4-server：Release the OCR model with the highest accuracy at present, the detection model accuracy increased by 4.9% in the Chinese and English scenes, and the recognition model accuracy increased by 2%
        refer [quickstart](./doc/doc_en/quickstart_en.md) quick use by one line command, At the same time, the whole process of model training, reasoning, and high-performance deployment can also be completed with few code in the [General OCR Industry Solution](https://aistudio.baidu.com/aistudio/modelsdetail?modelId=286) in PaddleX.
  - Release[PP-ChatOCR](https://aistudio.baidu.com/aistudio/modelsdetail?modelId=332), a new scheme for extracting key information of general scenes using PP-OCR model and ERNIE LLM.
- 🔨**2022.11 Add implementation of [4 cutting-edge algorithms](doc/doc_ch/algorithm_overview_en.md)**：Text Detection [DRRG](doc/doc_en/algorithm_det_drrg_en.md),  Text Recognition [RFL](./doc/doc_en/algorithm_rec_rfl_en.md), Image Super-Resolution [Text Telescope](doc/doc_en/algorithm_sr_telescope_en.md)，Handwritten Mathematical Expression Recognition [CAN](doc/doc_en/algorithm_rec_can_en.md)
- **2022.10 release [optimized JS version PP-OCRv3 model](./deploy/paddlejs/README.md)** with 4.3M model size, 8x faster inference time, and a ready-to-use web demo
- 💥 **Live Playback: Introduction to PP-StructureV2 optimization strategy**. Scan [the QR code below](#Community) using WeChat, follow the PaddlePaddle official account and fill out the questionnaire to join the WeChat group, get the live link and 20G OCR learning materials (including PDF2Word application, 10 models in vertical scenarios, etc.)


- **🔥2022.8.24 Release PaddleOCR [release/2.6](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)**
  - Release [PP-StructureV2](./ppstructure/)，with functions and performance fully upgraded, adapted to Chinese scenes, and new support for [Layout Recovery](./ppstructure/recovery) and **one line command to convert PDF to Word**;
  - [Layout Analysis](./ppstructure/layout) optimization: model storage reduced by 95%, while speed increased by 11 times, and the average CPU time-cost is only 41ms;
  - [Table Recognition](./ppstructure/table) optimization: 3 optimization strategies are designed, and the model accuracy is improved by 6% under comparable time consumption;
  - [Key Information Extraction](./ppstructure/kie) optimization：a visual-independent model structure is designed, the accuracy of semantic entity recognition is increased by 2.8%, and the accuracy of relation extraction is increased by 9.1%.
- **🔥2022.8 Release [OCR scene application collection](./applications/README_en.md)**
    - Release **9 vertical models** such as digital tube, LCD screen, license plate, handwriting recognition model, high-precision SVTR model, etc, covering the main OCR vertical applications in general, manufacturing, finance, and transportation industries.
- **2022.8 Add implementation of [8 cutting-edge algorithms](doc/doc_en/algorithm_overview_en.md)**
  - Text Detection: [FCENet](doc/doc_en/algorithm_det_fcenet_en.md), [DB++](doc/doc_en/algorithm_det_db_en.md)
  - Text Recognition: [ViTSTR](doc/doc_en/algorithm_rec_vitstr_en.md), [ABINet](doc/doc_en/algorithm_rec_abinet_en.md), [VisionLAN](doc/doc_en/algorithm_rec_visionlan_en.md), [SPIN](doc/doc_en/algorithm_rec_spin_en.md), [RobustScanner](doc/doc_en/algorithm_rec_robustscanner_en.md)
  - Table Recognition: [TableMaster](doc/doc_en/algorithm_table_master_en.md)
- **2022.5.9 Release PaddleOCR [release/2.5](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5)**
    - Release [PP-OCRv3](./doc/doc_en/ppocr_introduction_en.md#pp-ocrv3): With comparable speed, the effect of Chinese scene is further improved by 5% compared with PP-OCRv2, the effect of English scene is improved by 11%, and the average recognition accuracy of 80 language multilingual models is improved by more than 5%.
    - Release [PPOCRLabelv2](https://github.com/PFCCLab/PPOCRLabel): Add the annotation function for table recognition task, key information extraction task and irregular text image.
    - Release interactive e-book [*"Dive into OCR"*](./doc/doc_en/ocr_book_en.md), covers the cutting-edge theory and code practice of OCR full stack technology.
- [more](./doc/doc_en/update_en.md)


## 🌟 Features
PaddleOCR support a variety of cutting-edge algorithms related to OCR, and developed industrial featured models/solution [PP-OCR](./doc/doc_en/ppocr_introduction_en.md)、 [PP-Structure](./ppstructure/README.md) and [PP-ChatOCR](https://aistudio.baidu.com/aistudio/projectdetail/6488689) on this basis, and get through the whole process of data production, model training, compression, inference and deployment.

<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186171245-40abc4d7-904f-4949-ade1-250f86ed3a90.png">
</div>

> It is recommended to start with the “quick experience” in the document tutorial


## ⚡ Quick Experience
- Web online experience
    - PP-OCRv4 online experience：https://aistudio.baidu.com/aistudio/projectdetail/6611435
    - PP-ChatOCR online experience：https://aistudio.baidu.com/aistudio/projectdetail/6488689
- One line of code quick use: [Quick Start（Chinese/English/Multilingual/Document Analysis](./doc/doc_en/quickstart_en.md)
- Full-process experience of training, inference, and high-performance deployment in the Paddle AI suite (PaddleX)：
    - PP-OCRv4：https://aistudio.baidu.com/aistudio/modelsdetail?modelId=286
    - PP-ChatOCR：https://aistudio.baidu.com/aistudio/modelsdetail?modelId=332
- Mobile demo experience：[Installation DEMO](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)(Based on EasyEdge and Paddle-Lite, support iOS and Android systems)

## 📖 Technical exchange and cooperation
- PaddleX provides a one-stop full-process high-efficiency development platform for flying paddle ecological model training, pressure, and push. Its mission is to help AI technology quickly land, and its vision is to make everyone an AI Developer!
   - PaddleX currently covers areas such as image classification, object detection, image segmentation, 3D, OCR, and time series prediction, and has built-in 36 basic single models, such as RP-DETR, PP-YOLOE, PP-HGNet, PP-LCNet, PP- LiteSeg, etc.; integrated 12 practical industrial solutions, such as PP-OCRv4, PP-ChatOCR, PP-ShiTu, PP-TS, vehicle-mounted road waste detection, identification of prohibited wildlife products, etc.
   - PaddleX provides two AI development modes: "Toolbox" and "Developer". The toolbox mode can tune key hyperparameters without code, and the developer mode can perform single-model training, push and multi-model serial inference with low code, and supports both cloud and local terminals.
   - PaddleX also supports joint innovation and development, profit sharing! At present, PaddleX is rapidly iterating, and welcomes the participation of individual developers and enterprise developers to create a prosperous AI technology ecosystem!

## 📚 E-book: *Dive Into OCR*
- [Dive Into OCR](./doc/doc_en/ocr_book_en.md)

## 🛠️ PP-OCR Series Model List（Update on September 8th）

| Model introduction                                           | Model name                   | Recommended scene | Detection model                                              | Direction classifier                                         | Recognition model                                            |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Chinese and English ultra-lightweight PP-OCRv4 model（16.2M）     | ch_PP-OCRv4_xx          | Mobile & Server | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_distill_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar) |
| Chinese and English ultra-lightweight PP-OCRv3 model（16.2M）     | ch_PP-OCRv3_xx          | Mobile & Server | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
| English ultra-lightweight PP-OCRv3 model（13.4M）     | en_PP-OCRv3_xx          | Mobile & Server | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |

- For more model downloads (including multiple languages), please refer to [PP-OCR series model downloads](./doc/doc_en/models_list_en.md).
- For a new language request, please refer to [Guideline for new language_requests](#language_requests).
- For structural document analysis models, please refer to [PP-Structure models](./ppstructure/docs/models_list_en.md).

## 📖 Tutorials
- [Environment Preparation](./doc/doc_en/environment_en.md)
- [PP-OCR 🔥](./doc/doc_en/ppocr_introduction_en.md)
    - [Quick Start](./doc/doc_en/quickstart_en.md)
    - [Model Zoo](./doc/doc_en/models_en.md)
    - [Model training](./doc/doc_en/training_en.md)
        - [Text Detection](./doc/doc_en/detection_en.md)
        - [Text Recognition](./doc/doc_en/recognition_en.md)
        - [Text Direction Classification](./doc/doc_en/angle_class_en.md)
    - Model Compression
        - [Model Quantization](./deploy/slim/quantization/README_en.md)
        - [Model Pruning](./deploy/slim/prune/README_en.md)
        - [Knowledge Distillation](./doc/doc_en/knowledge_distillation_en.md)
    - [Inference and Deployment](./deploy/README.md)
        - [Python Inference](./doc/doc_en/inference_ppocr_en.md)
        - [C++ Inference](./deploy/cpp_infer/readme.md)
        - [Serving](./deploy/pdserving/README.md)
        - [Mobile](./deploy/lite/readme.md)
        - [Paddle2ONNX](./deploy/paddle2onnx/readme.md)
        - [PaddleCloud](./deploy/paddlecloud/README.md)
        - [Benchmark](./doc/doc_en/benchmark_en.md)
- [PP-Structure 🔥](./ppstructure/README.md)
    - [Quick Start](./ppstructure/docs/quickstart_en.md)
    - [Model Zoo](./ppstructure/docs/models_list_en.md)
    - [Model training](./doc/doc_en/training_en.md)
        - [Layout Analysis](./ppstructure/layout/README.md)
        - [Table Recognition](./ppstructure/table/README.md)
        - [Key Information Extraction](./ppstructure/kie/README.md)
    - [Inference and Deployment](./deploy/README.md)
        - [Python Inference](./ppstructure/docs/inference_en.md)
        - [C++ Inference](./deploy/cpp_infer/readme.md)
        - [Serving](./deploy/hubserving/readme_en.md)
- [Academic Algorithms](./doc/doc_en/algorithm_overview_en.md)
    - [Text detection](./doc/doc_en/algorithm_overview_en.md)
    - [Text recognition](./doc/doc_en/algorithm_overview_en.md)
    - [End-to-end OCR](./doc/doc_en/algorithm_overview_en.md)
    - [Table Recognition](./doc/doc_en/algorithm_overview_en.md)
    - [Key Information Extraction](./doc/doc_en/algorithm_overview_en.md)
    - [Add New Algorithms to PaddleOCR](./doc/doc_en/add_new_algorithm_en.md)
- Data Annotation and Synthesis
    - [Semi-automatic Annotation Tool: PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md)
    - [Data Synthesis Tool: Style-Text](https://github.com/PFCCLab/StyleText/blob/main/README.md)
    - [Other Data Annotation Tools](./doc/doc_en/data_annotation_en.md)
    - [Other Data Synthesis Tools](./doc/doc_en/data_synthesis_en.md)
- Datasets
    - [General OCR Datasets(Chinese/English)](doc/doc_en/dataset/datasets_en.md)
    - [HandWritten_OCR_Datasets(Chinese)](doc/doc_en/dataset/handwritten_datasets_en.md)
    - [Various OCR Datasets(multilingual)](doc/doc_en/dataset/vertical_and_multilingual_datasets_en.md)
    - [Layout Analysis](doc/doc_en/dataset/layout_datasets_en.md)
    - [Table Recognition](doc/doc_en/dataset/table_datasets_en.md)
    - [Key Information Extraction](doc/doc_en/dataset/kie_datasets_en.md)
- [Code Structure](./doc/doc_en/tree_en.md)
- [Visualization](#Visualization)
- [Community](#Community)
- [New language requests](#language_requests)
- [FAQ](./doc/doc_en/FAQ_en.md)
- [References](./doc/doc_en/reference_en.md)
- [License](#LICENSE)

## 👀 Visualization [more](./doc/doc_en/visualization_en.md)

<details open>
<summary>PP-OCRv3 Chinese model</summary>
<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic001.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic002.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic003.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-OCRv3 English model</summary>
<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/en/en_1.png" width="800">
    <img src="doc/imgs_results/PP-OCRv3/en/en_2.png" width="800">
</div>
</details>

<details open>
<summary>PP-OCRv3 Multilingual model</summary>
<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/multi_lang/japan_2.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/multi_lang/korean_1.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-StructureV2</summary>

- layout analysis + table recognition
<div align="center">
    <img src="./ppstructure/docs/table/ppstructure.GIF" width="800">
</div>

- SER (Semantic entity recognition)
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/197464552-69de557f-edff-4c7f-acbf-069df1ba097f.png" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808.jpg" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a.png" width="600">
</div>

- RE (Relation Extraction)
<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094813-3a8e16cc-42e5-4982-b9f4-0134dfb5688d.png" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb.jpg" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185540080-0431e006-9235-4b6d-b63d-0b3c6e1de48f.jpg" width="600">
</div>

## 🇺🇳 Guideline for New Language Requests

If you want to request a new language support, a PR with 1 following files are needed：

- In folder [ppocr/utils/dict](./ppocr/utils/dict),
it is necessary to submit the dict text to this path and name it with `{language}_dict.txt` that contains a list of all characters. Please see the format example from other files in that folder.

If your language has unique elements, please tell me in advance within any way, such as useful links, wikipedia and so on.

More details, please refer to [Multilingual OCR Development Plan](https://github.com/PaddlePaddle/PaddleOCR/issues/1048).

## 📄 License
This project is released under <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>
