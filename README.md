English | [简体中文](README_ch.md)

## Introduction
PaddleOCR aims to create multilingual, awesome, leading, and practical OCR tools that help users train better models and apply them into practice.

## Notice
PaddleOCR supports both dynamic graph and static graph programming paradigm
- Dynamic graph: dygraph branch (default)
- Static graph: develop branch

**Recent updates**
- 2020.12.15 update Data synthesis tool, i.e., [Style-Text](./StyleText/README.md)，easy to synthesize a large number of images which are similar to the target scene image.
- 2020.11.25 Update a new data annotation tool, i.e., [PPOCRLabel](./PPOCRLabel/README.md), which is helpful to improve the labeling efficiency. Moreover, the labeling results can be used in training of the PP-OCR system directly.
- 2020.9.22 Update the PP-OCR technical article, https://arxiv.org/abs/2009.09941
- [more](./doc/doc_en/update_en.md)

## Features
- PPOCR series of high-quality pre-trained models, comparable to commercial effects
    - Ultra lightweight ppocr_mobile series models: detection (3.0M) + direction classifier (1.4M) + recognition (5.0M) = 9.4M
    - General ppocr_server series models: detection (47.1M) + direction classifier (1.4M) + recognition (94.9M) = 143.4M
    - Support Chinese, English, and digit recognition, vertical text recognition, and long text recognition
    - Support multi-language recognition: Korean, Japanese, German, French
- Rich toolkits related to the OCR areas
    - Semi-automatic data annotation tool, i.e., PPOCRLabel: support fast and efficient data annotation
    - Data synthesis tool, i.e., Style-Text: easy to synthesize a large number of images which are similar to the target scene image
- Support user-defined training, provides rich predictive inference deployment solutions
- Support PIP installation, easy to use
- Support Linux, Windows, MacOS and other systems

## Visualization

<div align="center">
    <img src="doc/imgs_results/ch_ppocr_mobile_v2.0/test_add_91.jpg" width="800">
    <img src="doc/imgs_results/ch_ppocr_mobile_v2.0/00018069.jpg" width="800">
</div>

The above pictures are the visualizations of the general ppocr_server model. For more effect pictures, please see [More visualizations](./doc/doc_en/visualization_en.md).

<a name="Community"></a>
## Community
- Scan the QR code below with your Wechat, you can access to official technical exchange group. Look forward to your participation.

<div align="center">
<img src="./doc/joinus.PNG"  width = "200" height = "200" />
</div>


## Quick Experience

You can also quickly experience the ultra-lightweight OCR : [Online Experience](https://www.paddlepaddle.org.cn/hub/scene/ocr)

Mobile DEMO experience (based on EasyEdge and Paddle-Lite, supports iOS and Android systems): [Sign in to the website to obtain the QR code for  installing the App](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)

 Also, you can scan the QR code below to install the App (**Android support only**)

<div align="center">
<img src="./doc/ocr-android-easyedge.png"  width = "200" height = "200" />
</div>

- [**OCR Quick Start**](./doc/doc_en/quickstart_en.md)

<a name="Supported-Chinese-model-list"></a>


## PP-OCR 2.0 series model list（Update on Dec 15）

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
    - [Data Synthesis Tool: Style_Edit](./StyleTextRec/README.md)
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



<a name="PP-OCR-Pipeline"></a>

## PP-OCR Pipeline

<div align="center">
    <img src="./doc/ppocr_framework.png" width="800">
</div>

PP-OCR is a practical ultra-lightweight OCR system. It is mainly composed of three parts: DB text detection, detection frame correction and CRNN text recognition. The system adopts 19 effective strategies from 8 aspects including backbone network selection and adjustment, prediction head design, data augmentation, learning rate transformation strategy, regularization parameter selection, pre-training model use, and automatic model tailoring and quantization to optimize and slim down the models of each module. The final results are an ultra-lightweight Chinese and English OCR model with an overall size of 3.5M and a 2.8M English digital OCR model. For more details, please refer to the PP-OCR technical article (https://arxiv.org/abs/2009.09941). Besides, The implementation of the FPGM Pruner and PACT quantization is based on [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim).



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
## Guideline for new language requests

If you want to request a new language support, a PR with 2 following files are needed：

1. In folder [ppocr/utils/dict](./ppocr/utils/dict),
it is necessary to submit the dict text to this path and name it with `{language}_dict.txt` that contains a list of all characters. Please see the format example from other files in that folder.

2. In folder [ppocr/utils/corpus](./ppocr/utils/corpus),
it is necessary to submit the corpus to this path and name it with `{language}_corpus.txt` that contains a list of words in your language.
Maybe, 50000 words per language is necessary at least.
Of course, the more, the better.

If your language has unique elements, please tell me in advance within any way, such as useful links, wikipedia and so on.

More details, please refer to [Multilingual OCR Development Plan](https://github.com/PaddlePaddle/PaddleOCR/issues/1048).


<a name="LICENSE"></a>
## License
This project is released under <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>

<a name="CONTRIBUTION"></a>
## Contribution
We welcome all the contributions to PaddleOCR and appreciate for your feedback very much.

- Many thanks to [Khanh Tran](https://github.com/xxxpsyduck) and [Karl Horky](https://github.com/karlhorky) for contributing and revising the English documentation.
- Many thanks to [zhangxin](https://github.com/ZhangXinNan) for contributing the new visualize function、add .gitgnore and discard set PYTHONPATH manually.
- Many thanks to [lyl120117](https://github.com/lyl120117) for contributing the code for printing the network structure.
- Thanks [xiangyubo](https://github.com/xiangyubo) for contributing the handwritten Chinese OCR datasets.
- Thanks [authorfu](https://github.com/authorfu) for contributing Android demo  and [xiadeye](https://github.com/xiadeye) contributing iOS demo, respectively.
- Thanks [BeyondYourself](https://github.com/BeyondYourself) for contributing many great suggestions and simplifying part of the code style.
- Thanks [tangmq](https://gitee.com/tangmq) for contributing Dockerized deployment services to PaddleOCR and supporting the rapid release of callable Restful API services.
- Thanks [lijinhan](https://github.com/lijinhan) for contributing a new way, i.e., java SpringBoot, to achieve the request for the Hubserving deployment.
- Thanks [Mejans](https://github.com/Mejans) for contributing the Occitan corpus and character set.
- Thanks [LKKlein](https://github.com/LKKlein) for contributing a new deploying package with the Golang program language.
- Thanks [Evezerest](https://github.com/Evezerest), [ninetailskim](https://github.com/ninetailskim), [edencfc](https://github.com/edencfc), [BeyondYourself](https://github.com/BeyondYourself) and [1084667371](https://github.com/1084667371) for contributing a new data annotation tool, i.e., PPOCRLabel。
