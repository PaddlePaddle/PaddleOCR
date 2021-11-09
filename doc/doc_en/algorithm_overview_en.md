# Two-stage Algorithm

- [1. Algorithm Introduction](#1-algorithm-introduction)
  * [1.1 Text Detection Algorithm](#11-text-detection-algorithm)
  * [1.2 Text Recognition Algorithm](#12-text-recognition-algorithm)
- [2. Training](#2-training)
- [3. Inference](#3-inference)

<a name="Algorithm_introduction"></a>

## 1. Algorithm Introduction

This tutorial lists the text detection algorithms and text recognition algorithms supported by PaddleOCR, as well as the models and metrics of each algorithm on **English public datasets**. It is mainly used for algorithm introduction and algorithm performance comparison. For more models on other datasets including Chinese, please refer to [PP-OCR v2.0 models list](./models_list_en.md).


- [1. Text Detection Algorithm](#TEXTDETECTIONALGORITHM)
- [2. Text Recognition Algorithm](#TEXTRECOGNITIONALGORITHM)

<a name="TEXTDETECTIONALGORITHM"></a>

### 1.1 Text Detection Algorithm

PaddleOCR open source text detection algorithms list:
- [x]  EAST([paper](https://arxiv.org/abs/1704.03155))[2]
- [x]  DB([paper](https://arxiv.org/abs/1911.08947))[1]
- [x]  SAST([paper](https://arxiv.org/abs/1908.05498))[4]

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|EAST|ResNet50_vd|85.80%|86.71%|86.25%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)|
|EAST|MobileNetV3|79.42%|80.64%|80.03%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar)|
|DB|ResNet50_vd|86.41%|78.72%|82.38%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|77.29%|73.08%|75.12%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|SAST|ResNet50_vd|91.39%|83.77%|87.42%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|

On Total-Text dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|89.63%|78.44%|83.66%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|

**Note：** Additional data, like icdar2013, icdar2017, COCO-Text, ArT, was added to the model training of SAST. Download English public dataset in organized format used by PaddleOCR from:
* [Baidu Drive](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (download code: 2bpi).
* [Google Drive](https://drive.google.com/drive/folders/1ll2-XEVyCQLpJjawLDiRlvo_i4BqHCJe?usp=sharing)

For the training guide and use of PaddleOCR text detection algorithms, please refer to the document [Text detection model training/evaluation/prediction](./detection_en.md)

<a name="TEXTRECOGNITIONALGORITHM"></a>
### 1.2 Text Recognition Algorithm

PaddleOCR open-source text recognition algorithms list:
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))[7]
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))[10]
- [x]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))[11]
- [x]  RARE([paper](https://arxiv.org/abs/1603.03915v1))[12]
- [x]  SRN([paper](https://arxiv.org/abs/2003.12294))[5]
- [x]  NRTR([paper](https://arxiv.org/abs/1806.00926v2))

Refer to [DTRB](https://arxiv.org/abs/1904.01906), the training and evaluation result of these above text recognition (using MJSynth and SynthText for training, evaluate on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) is as follow:

|Model|Backbone|Avg Accuracy|Module combination|Download link|
|---|---|---|---|---|
|Rosetta|Resnet34_vd|80.9%|rec_r34_vd_none_none_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|78.05%|rec_mv3_none_none_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|
|CRNN|Resnet34_vd|82.76%|rec_r34_vd_none_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar)|
|CRNN|MobileNetV3|79.97%|rec_mv3_none_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar)|
|StarNet|Resnet34_vd|84.44%|rec_r34_vd_tps_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|81.42%|rec_mv3_tps_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|
|RARE|MobileNetV3|82.5%|rec_mv3_tps_bilstm_att |[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar)|
|RARE|Resnet34_vd|83.6%|rec_r34_vd_tps_bilstm_att |[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar)|
|SRN|Resnet50_vd_fpn| 88.52% | rec_r50fpn_vd_none_srn |[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar)|
|NRTR|NRTR_MTB| 84.3% | rec_mtb_nrtr | [Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) |

Please refer to the document for training guide and use of PaddleOCR 

## 2. Training

For the training guide and use of PaddleOCR text detection algorithms, please refer to the document [Text detection model training/evaluation/prediction](./detection_en.md). For text recognition algorithms, please refer to [Text recognition model training/evaluation/prediction](./recognition_en.md)

## 3. Inference

Except for the PP-OCR series models of the above models, the other models only support inference based on the Python engine. For details, please refer to [Inference based on Python prediction engine](./inference_en.md)
