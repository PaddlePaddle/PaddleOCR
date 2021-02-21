English | [简体中文](README_ch.md)

<a name="Intro"></a>
## Intro
PaddleOCR aims to create multilingual, leading and practical OCR tools that help users train better models and apply them into practice.

<a name="PP-OCR-Pipeline"></a>
## System Architecture
<div align="center">
    <img src="./doc/ppocr_framework.png" width="800">
</div>

PaddleOCR is a practical ultra-lightweight OCR system. It is mainly composed of 3 parts: the DB Text Detection, the Detection Frame Correction and the CRNN Text Recognition. The system adopts 19 effective strategies from 8 aspects including the Backbone Network Selection and Adjustment, Prediction Head Design, Data Augmentation, Learning Rate Transformation Strategy, Regularization Parameter Selection, Pre-training Model Use, Automatic Model Tailoring and Quantization to optimize and slim down each module. The final results are an ultra-lightweight Chinese & English OCR model with an overall size being 3.5M and an English digital OCR model being 2.8M. For more details, please refer to our [technical report](https://arxiv.org/abs/2009.09941). Besides, the Implementation of FPGM Pruner and PACT Quantization is based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim).

<a name="Attention"></a>
## Attention
PaddleOCR supports both dynamic graph and static graph programming paradigm
- Dynamic graph: dygraph branch (default), **supported by paddle 2.0.0 ([installation](./doc/doc_en/installation_en.md))**
- Static graph: develop branch

**Recent updates**

- 2021.02.08 Release PaddleOCRv2.0(branch release/2.0) and set as default branch. Check release note here: https://github.com/PaddlePaddle/PaddleOCR/releases/tag/v2.0.0 .

- 2021.01.21 ADD more than 25+ multilingual recognition models [model list](./doc/doc_en/models_list_en.md) including：English, Chinese, German, French, Japanese，Spanish，Portuguese, Russia, Arabic et. al. Models for many MORE languages will be added soon. [Develop Plan](https://github.com/PaddlePaddle/PaddleOCR/issues/1048).

- 2020.12.15 Update Data synthesis tool, i.e., [Style-Text](./StyleText/README.md), easy to synthesize a large number of images which are similar to the target scene image.

- 2020.11.25 Update a new data annotation tool, i.e., [PPOCRLabel](./PPOCRLabel/README.md), which improves the data labeling efficiency. More, the labeling results can be used for training via the PP-OCR system directly.

- 2020.09.22 Update the PP-OCR [technical article](https://arxiv.org/abs/2009.09941)

- [See More](./doc/doc_en/update_en.md)

<a name="Features"></a>
## Features
- High Quality pre-trained industry strength models
    - Ultra lightweight ppocr_mobile series models: detection (3.0M) + direction classifier (1.4M) + recognition (5.0M) = 9.4M
    - General ppocr_server series models: detection (47.1M) + direction classifier (1.4M) + recognition (94.9M) = 143.4M
    - Support Chinese, English, and digit recognition, vertical text recognition, and long text recognition
    - Support multi-language recognition: Korean, Japanese, German, French
- Rich Toolkits for the whole OCR pipeline
    - Semi-automatic data annotation tool, i.e., PPOCRLabel: support fast and efficient data annotation
    - Data synthesis tool, i.e., Style-Text: easy to synthesize a large number of images which are similar to the target scene image
- Support user-defined training, provides rich predictive inference deployment solutions
- Support PIP installation, easy to use
- Support Linux, Windows, MacOS and other systems

<a name="Visualization"></a>
## Visualization

<div align="center">
    <img src="doc/imgs_results/ch_ppocr_mobile_v2.0/test_add_91.jpg" width="800">
    <img src="doc/imgs_results/ch_ppocr_mobile_v2.0/00018069.jpg" width="800">
</div>

The above image is the visualizations of the general ppocr_server model. For more, please see [More visualizations](./doc/doc_en/visualization_en.md).

<a name="Community"></a>
## Community
- Scan the QR code below using your Wechat, you can join the official technical group. We are looking forward to your participation. For users who don't have an Wechat account, please use github for now.

<div align="center">
<img src="./doc/joinus.PNG"  width = "200" height = "200" />
</div>

<a name="OnlineDemo"></a>
## Online Demo

Quickly experience our ultra-lightweight OCR System: [Online Demo](https://www.paddlepaddle.org.cn/hub/scene/ocr)

[Mobile DEMO](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite) (Note: Sign in to the website to obtain the QR code for installing the App. This demo is based on Baidu EasyEdge and Paddle-Lite, supports iOS and Android systems)

Also, you can scan the QR code below to install the App (Note: ONLY available on **Android** for current)

<div align="center">
<img src="./doc/ocr-android-easyedge.png"  width = "200" height = "200" />
</div>

<a name="Supported-Chinese-model-list"></a>
## PP-OCR 2.0 series model list（Update on Dec 15）
**Note** : Compared with [models 1.1](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/models_list_en.md), which are trained with static graph programming paradigm, models 2.0 are the dynamic graph trained version and achieve close performance.

| Model introduction                                           | Model name                   | Recommended scene | Detection model                                              | Direction classifier                                         | Recognition model                                            |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Chinese and English ultra-lightweight OCR model (9.4M)       | ch_ppocr_mobile_v2.0_xx      | Mobile & server   |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_pre.tar)      |
| Chinese and English general OCR model (143.4M)               | ch_ppocr_server_v2.0_xx      | Server            |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)    |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_traingit.tar)    |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_pre.tar)  |  


For more model downloads (including multiple languages), please refer to [PP-OCR v2.0 series model downloads](./doc/doc_en/models_list_en.md).

For a new language request, please refer to [Guideline for new language_requests](#language_requests).

## Tutorials
- [Installation](./doc/doc_en/installation_en.md)
- [Quick Start](./doc/doc_en/quickstart_en.md)
- [Code Structure](./doc/doc_en/tree_en.md)
- Algorithm Introduction
    - [Text Detection Algorithm](./doc/doc_en/algorithm_overview_en.md)
    - [Text Recognition Algorithm](./doc/doc_en/algorithm_overview_en.md)
    - [PP-OCR Pipeline](#PP-OCR-Pipeline)
- Model Training/Evaluation
    - [Text Detection](./doc/doc_en/detection_en.md)
    - [Text Recognition](./doc/doc_en/recognition_en.md)
    - [Direction Classification](./doc/doc_en/angle_class_en.md)
    - [Yml Configuration](./doc/doc_en/config_en.md)
- Inference and Deployment
    - [Quick Inference Based on PIP](./doc/doc_en/whl_en.md)
    - [Python Inference](./doc/doc_en/inference_en.md)
    - [C++ Inference](./deploy/cpp_infer/readme_en.md)
    - [Serving](./deploy/hubserving/readme_en.md)
    - [Mobile](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/lite/readme_en.md)
    - [Benchmark](./doc/doc_en/benchmark_en.md)  
- Data Annotation and Synthesis
    - [Semi-automatic Annotation Tool: PPOCRLabel](./PPOCRLabel/README.md)
    - [Data Synthesis Tool: Style-Text](./StyleText/README.md)
    - [Other Data Annotation Tools](./doc/doc_en/data_annotation_en.md)
    - [Other Data Synthesis Tools](./doc/doc_en/data_synthesis_en.md)
- Datasets
    - [General OCR Datasets(Chinese/English)](./doc/doc_en/datasets_en.md)
    - [HandWritten_OCR_Datasets(Chinese)](./doc/doc_en/handwritten_datasets_en.md)
    - [Various OCR Datasets(multilingual)](./doc/doc_en/vertical_and_multilingual_datasets_en.md)
- [Visualization](#Visualization)
- [New language requests](#language_requests)
- [FAQ](./doc/doc_en/FAQ_en.md)
- [Community](#Community)
- [References](./doc/doc_en/reference_en.md)
- [License](#LICENSE)
- [Contribution](#CONTRIBUTION)

## Visualization [more](./doc/doc_en/visualization_en.md)
- Chinese OCR model
<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/test_add_91.jpg" width="800">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/00015504.jpg" width="800">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/00056221.jpg" width="800">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/rotate_00052204.jpg" width="800">
</div>

- English OCR model
<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/img_12.jpg" width="800">
</div>

- Multilingual OCR model
<div align="center">
    <img src="./doc/imgs_results/french_0.jpg" width="800">
    <img src="./doc/imgs_results/korean.jpg" width="800">
</div>

<a name="language_requests"></a>
## Guideline for NEW Language Support

If you want to request a new language support, a PR with 2 following files are needed:

1. In folder [ppocr/utils/dict](./ppocr/utils/dict),
it is necessary to submit the dict text to this path and name it with `{language}_dict.txt` that contains a list of all characters. Please see the format example from others in that folder.

2. In folder [ppocr/utils/corpus](./ppocr/utils/corpus),
it is necessary to submit the corpus to this path and name it with `{language}_corpus.txt` that contains a list of words in your language. Based on previous experience, at least 50000 words per language is necessary. Of course, the more the better.

If your language contains unique elements, please INFO us in advance, such as useful links, wikipedia pages et. al. For more, please refer to [Multilingual OCR Development Plan](https://github.com/PaddlePaddle/PaddleOCR/issues/1048).

<a name="LICENSE"></a>
## License 版权
This project is under the <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>.

<a name="CONTRIBUTION"></a>
## Contribution 贡献
We welcome all the contributions to PaddleOCR and appreciate your feedback very much.

- Thanks [Khanh Tran](https://github.com/xxxpsyduck) and [Karl Horky](https://github.com/karlhorky) for contributing and revising the English documentation.
- Thanks [zhangxin](https://github.com/ZhangXinNan) for contributing the new visualize function, add .gitignore and discard set PYTHONPATH manually.
- Thanks [lyl120117](https://github.com/lyl120117) for contributing the code for printing the network structure.
- Thanks [xiangyubo](https://github.com/xiangyubo) for contributing the handwritten Chinese OCR datasets.
- Thanks [authorfu](https://github.com/authorfu) for contributing the Android demo and [xiadeye](https://github.com/xiadeye) for contributing the iOS demo, respectively.
- Thanks [BeyondYourself](https://github.com/BeyondYourself) for contributing many good suggestions and simplifying the code style.
- Thanks [tangmq](https://gitee.com/tangmq) for contributing the Dockerized deployment services to PaddleOCR and supporting the rapid release of the Callable Restful API Services.
- Thanks [lijinhan](https://github.com/lijinhan) for contributing a new way, i.e., java SpringBoot, to achieve the request for the using the Hubserving as the deployment method.
- Thanks [Mejans](https://github.com/Mejans) for contributing the Occitan corpus and the character set.
- Thanks [LKKlein](https://github.com/LKKlein) for contributing a new deploying package with the Golang program language.
- Thanks [Evezerest](https://github.com/Evezerest), [ninetailskim](https://github.com/ninetailskim), [edencfc](https://github.com/edencfc), [BeyondYourself](https://github.com/BeyondYourself) and [1084667371](https://github.com/1084667371) for contributing a new data annotation tool, i.e., PPOCRLabel.
