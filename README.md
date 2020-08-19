English | [简体中文](README_cn.md)

## Introduction
PaddleOCR aims to create rich, leading, and practical OCR tools that help users train better models and apply them into practice.

**Recent updates**
- 2020.8.16, Release text detection algorithm [SAST](https://arxiv.org/abs/1908.05498) and text recognition algorithm [SRN](https://arxiv.org/abs/2003.12294)
- 2020.7.23, Release the playback and PPT of live class on BiliBili station, PaddleOCR Introduction, [address](https://aistudio.baidu.com/aistudio/course/introduce/1519)
- 2020.7.15, Add mobile App demo , support both iOS and  Android  ( based on easyedge and Paddle Lite)
- 2020.7.15, Improve the  deployment ability, add the C + +  inference , serving deployment. In addition, the benchmarks of the ultra-lightweight OCR model are provided.
- 2020.7.15, Add several related datasets, data annotation and synthesis tools.
- [more](./doc/doc_en/update_en.md)

## Features
- Ultra-lightweight OCR model, total model size is only 8.6M
    - Single model supports Chinese/English numbers combination recognition, vertical text recognition, long text recognition
    - Detection model DB (4.1M) + recognition model CRNN (4.5M)
- Various text detection algorithms: EAST, DB
- Various text recognition algorithms: Rosetta, CRNN, STAR-Net, RARE
- Support Linux, Windows, macOS and other systems.

## Visualization

![](doc/imgs_results/11.jpg)

![](doc/imgs_results/img_10.jpg)

[More visualization](./doc/doc_en/visualization_en.md)

You can also quickly experience the ultra-lightweight OCR : [Online Experience](https://www.paddlepaddle.org.cn/hub/scene/ocr)

Mobile DEMO experience (based on EasyEdge and Paddle-Lite, supports iOS and Android systems): [Sign in to the website to obtain the QR code for  installing the App](https://ai.baidu.com/easyedge/app/openSource?from=paddlelite)

 Also, you can scan the QR code below to install the App (**Android support only**)

<div align="center">
<img src="./doc/ocr-android-easyedge.png"  width = "200" height = "200" />
</div>

- [**OCR Quick Start**](./doc/doc_en/quickstart_en.md)

<a name="Supported-Chinese-model-list"></a>

### Supported Models:

|Model Name|Description |Detection Model link|Recognition Model link| Support for space Recognition Model link|
|-|-|-|-|-|
|db_crnn_mobile|ultra-lightweight OCR model|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) / [pre-train model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)
|db_crnn_server|General OCR model|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar) / [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) / [pre-train model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)


## Tutorials
- [Installation](./doc/doc_en/installation_en.md)
- [Quick Start](./doc/doc_en/quickstart_en.md)
- Algorithm introduction
    - [Text Detection Algorithm](#TEXTDETECTIONALGORITHM)
    - [Text Recognition Algorithm](#TEXTRECOGNITIONALGORITHM)
    - [END-TO-END OCR Algorithm](#ENDENDOCRALGORITHM)
- Model training/evaluation
    - [Text Detection](./doc/doc_en/detection_en.md)
    - [Text Recognition](./doc/doc_en/recognition_en.md)
    - [Yml Configuration](./doc/doc_en/config_en.md)
    - [Tricks](./doc/doc_en/tricks_en.md)
- Deployment
    - [Python Inference](./doc/doc_en/inference_en.md)
    - [C++ Inference](./deploy/cpp_infer/readme_en.md)
    - [Serving](./doc/doc_en/serving_en.md)
    - [Mobile](./deploy/lite/readme_en.md)
    - Model Quantization and Compression (coming soon)
    - [Benchmark](./doc/doc_en/benchmark_en.md)
- Datasets
    - [General OCR Datasets(Chinese/English)](./doc/doc_en/datasets_en.md)
    - [HandWritten_OCR_Datasets(Chinese)](./doc/doc_en/handwritten_datasets_en.md)
    - [Various OCR Datasets(multilingual)](./doc/doc_en/vertical_and_multilingual_datasets_en.md)
    - [Data Annotation Tools](./doc/doc_en/data_annotation_en.md)
    - [Data Synthesis Tools](./doc/doc_en/data_synthesis_en.md)
- [FAQ](#FAQ)
- Visualization
    - [Ultra-lightweight Chinese/English OCR Visualization](#UCOCRVIS)
    - [General Chinese/English OCR Visualization](#GeOCRVIS)
    - [Chinese/English OCR Visualization (Support Space Recognition )](#SpaceOCRVIS)
- [Community](#Community)
- [References](./doc/doc_en/reference_en.md)
- [License](#LICENSE)
- [Contribution](#CONTRIBUTION)

<a name="TEXTDETECTIONALGORITHM"></a>
## Text Detection Algorithm

PaddleOCR open source text detection algorithms list:
- [x]  EAST([paper](https://arxiv.org/abs/1704.03155))
- [x]  DB([paper](https://arxiv.org/abs/1911.08947))
- [x]  SAST([paper](https://arxiv.org/abs/1908.05498))(Baidu Self-Research)

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|precision|recall|Hmean|Download link|
|-|-|-|-|-|-|
|EAST|ResNet50_vd|88.18%|85.51%|86.82%|[Download link](https://paddleocr.bj.bcebos.com/det_r50_vd_east.tar)|
|EAST|MobileNetV3|81.67%|79.83%|80.74%|[Download link](https://paddleocr.bj.bcebos.com/det_mv3_east.tar)|
|DB|ResNet50_vd|83.79%|80.65%|82.19%|[Download link](https://paddleocr.bj.bcebos.com/det_r50_vd_db.tar)|
|DB|MobileNetV3|75.92%|73.18%|74.53%|[Download link](https://paddleocr.bj.bcebos.com/det_mv3_db.tar)|
|SAST|ResNet50_vd|92.18%|82.96%|87.33%|[Download link](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_icdar2015.tar)|

On Total-Text dataset, the text detection result is as follows:

|Model|Backbone|precision|recall|Hmean|Download link|
|-|-|-|-|-|-|
|SAST|ResNet50_vd|88.74%|79.80%|84.03%|[Download link](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_total_text.tar)|

**Note：** Additional data, like icdar2013, icdar2017, COCO-Text, ArT, was added to the model training of SAST. Download English public dataset in organized format used by PaddleOCR from [Baidu Drive](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (download code: 2bpi).

For use of [LSVT](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/datasets_en.md#1-icdar2019-lsvt) street view dataset with a total of 3w training data，the related configuration and pre-trained models for text detection task are as follows:  
|Model|Backbone|Configuration file|Pre-trained model|
|-|-|-|-|
|ultra-lightweight OCR model|MobileNetV3|det_mv3_db.yml|[Download link](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|
|General OCR model|ResNet50_vd|det_r50_vd_db.yml|[Download link](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|

* Note: For the training and evaluation of the above DB model, post-processing parameters box_thresh=0.6 and unclip_ratio=1.5 need to be set. If using different datasets and different models for training, these two parameters can be adjusted for better result.

For the training guide and use of PaddleOCR text detection algorithms, please refer to the document [Text detection model training/evaluation/prediction](./doc/doc_en/detection_en.md)

<a name="TEXTRECOGNITIONALGORITHM"></a>
## Text Recognition Algorithm

PaddleOCR open-source text recognition algorithms list:
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))
- [x]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))
- [x]  RARE([paper](https://arxiv.org/abs/1603.03915v1))
- [x]  SRN([paper](https://arxiv.org/abs/2003.12294))(Baidu Self-Research)

Refer to [DTRB](https://arxiv.org/abs/1904.01906), the training and evaluation result of these above text recognition (using MJSynth and SynthText for training, evaluate on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) is as follow:

|Model|Backbone|Avg Accuracy|Module combination|Download link|
|-|-|-|-|-|
|Rosetta|Resnet34_vd|80.24%|rec_r34_vd_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_none_ctc.tar)|
|Rosetta|MobileNetV3|78.16%|rec_mv3_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_none_none_ctc.tar)|
|CRNN|Resnet34_vd|82.20%|rec_r34_vd_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_bilstm_ctc.tar)|
|CRNN|MobileNetV3|79.37%|rec_mv3_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_none_bilstm_ctc.tar)|
|STAR-Net|Resnet34_vd|83.93%|rec_r34_vd_tps_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_ctc.tar)|
|STAR-Net|MobileNetV3|81.56%|rec_mv3_tps_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_ctc.tar)|
|RARE|Resnet34_vd|84.90%|rec_r34_vd_tps_bilstm_attn|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_attn.tar)|
|RARE|MobileNetV3|83.32%|rec_mv3_tps_bilstm_attn|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_attn.tar)|
|SRN|Resnet50_vd_fpn|88.33%|rec_r50fpn_vd_none_srn|[Download link](https://paddleocr.bj.bcebos.com/SRN/rec_r50fpn_vd_none_srn.tar)|

**Note：** SRN model uses data expansion method to expand the two training sets mentioned above, and the expanded data can be downloaded from [Baidu Drive](https://pan.baidu.com/s/1-HSZ-ZVdqBF2HaBZ5pRAKA) (download code: y3ry).

The average accuracy of the two-stage training in the original paper is 89.74%, and that of one stage training in paddleocr is 88.33%. Both pre-trained weights can be downloaded [here](https://paddleocr.bj.bcebos.com/SRN/rec_r50fpn_vd_none_srn.tar).

We use [LSVT](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/datasets_en.md#1-icdar2019-lsvt) dataset and cropout 30w  training data from original photos by using position groundtruth and make some calibration needed. In addition, based on the LSVT corpus, 500w synthetic data is generated to train the model. The related configuration and pre-trained models are as follows:

|Model|Backbone|Configuration file|Pre-trained model|
|-|-|-|-|
|ultra-lightweight OCR model|MobileNetV3|rec_chinese_lite_train.yml|[Download link](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) & [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)|
|General OCR model|Resnet34_vd|rec_chinese_common_train.yml|[Download link](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) & [pre-trained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)|

Please refer to the document for training guide and use of PaddleOCR text recognition algorithms [Text recognition model training/evaluation/prediction](./doc/doc_en/recognition_en.md)

<a name="ENDENDOCRALGORITHM"></a>
## END-TO-END OCR Algorithm
- [ ]  [End2End-PSL](https://arxiv.org/abs/1909.07808)(Baidu Self-Research, coming soon)

## Visualization

<a name="UCOCRVIS"></a>
### 1.Ultra-lightweight Chinese/English OCR Visualization [more](./doc/doc_en/visualization_en.md)

<div align="center">
    <img src="doc/imgs_results/1.jpg" width="800">
</div>

<a name="GeOCRVIS"></a>
### 2. General Chinese/English OCR Visualization [more](./doc/doc_en/visualization_en.md)

<div align="center">
    <img src="doc/imgs_results/chinese_db_crnn_server/11.jpg" width="800">
</div>

<a name="SpaceOCRVIS"></a>
### 3.Chinese/English OCR Visualization (Space_support) [more](./doc/doc_en/visualization_en.md)

<div align="center">
    <img src="doc/imgs_results/chinese_db_crnn_server/en_paper.jpg" width="800">
</div>

<a name="FAQ"></a>

## FAQ
1. Error when using attention-based recognition model: KeyError: 'predict'

    The inference of recognition model based on attention loss is still being debugged. For Chinese text recognition, it is recommended to choose the recognition model based on CTC loss first. In practice, it is also found that the recognition model based on attention loss is not as effective as the one based on CTC loss.

2. About inference speed

    When there are a lot of texts in the picture, the prediction time will increase. You can use `--rec_batch_num` to set a smaller prediction batch size. The default value is 30, which can be changed to 10 or other values.

3. Service deployment and mobile deployment

    It is expected that the service deployment based on Serving and the mobile deployment based on Paddle Lite will be released successively in mid-to-late June. Stay tuned for more updates.

4. Release time of self-developed algorithm

    Baidu Self-developed algorithms such as SAST, SRN and end2end PSL will be released in June or July. Please be patient.

[more](./doc/doc_en/FAQ_en.md)

<a name="Community"></a>
## Community
Scan  the QR code below with your wechat and completing the questionnaire, you can access to offical technical exchange group.

<div align="center">
<img src="./doc/joinus.jpg"  width = "200" height = "200" />
</div>

<a name="LICENSE"></a>
## License
This project is released under <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>

<a name="CONTRIBUTION"></a>
## Contribution
We welcome all the contributions to PaddleOCR and appreciate for your feedback very much.

- Many thanks to [Khanh Tran](https://github.com/xxxpsyduck) for contributing the English documentation.
- Many thanks to [zhangxin](https://github.com/ZhangXinNan) for contributing the new visualize function、add .gitgnore and discard set PYTHONPATH manually.
- Many thanks to [lyl120117](https://github.com/lyl120117) for contributing the code for printing the network structure.
- Thanks [xiangyubo](https://github.com/xiangyubo) for contributing the handwritten Chinese OCR datasets.
- Thanks [authorfu](https://github.com/authorfu) for contributing Android demo  and [xiadeye](https://github.com/xiadeye) contributing iOS demo, respectively.
- Thanks [BeyondYourself](https://github.com/BeyondYourself) for contributing many great suggestions and simplifying part of the code style.
- Thanks [tangmq](https://gitee.com/tangmq) for contributing Dockerized deployment services to PaddleOCR and supporting the rapid release of callable Restful API services.
