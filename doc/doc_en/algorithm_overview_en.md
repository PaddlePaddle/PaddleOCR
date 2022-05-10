# OCR Algorithms

- [1. Two-stage Algorithms](#1)
  * [1.1 Text Detection Algorithms](#11)
  * [1.2 Text Recognition Algorithms](#12)
- [2. End-to-end Algorithms](#2)


This tutorial lists the OCR algorithms supported by PaddleOCR, as well as the models and metrics of each algorithm on **English public datasets**. It is mainly used for algorithm introduction and algorithm performance comparison. For more models on other datasets including Chinese, please refer to [PP-OCR v2.0 models list](./models_list_en.md).

<a name="1"></a>

## 1. Two-stage Algorithms

<a name="11"></a>

### 1.1 Text Detection Algorithms

Supported text detection algorithms (Click the link to get the tutorial):
- [x]  [DB](./algorithm_det_db_en.md)
- [x]  [EAST](./algorithm_det_east_en.md)
- [x]  [SAST](./algorithm_det_sast_en.md)
- [x]  [PSENet](./algorithm_det_psenet_en.md)
- [x]  [FCENet](./algorithm_det_fcenet_en.md)

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|EAST|ResNet50_vd|88.71%|81.36%|84.88%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)|
|EAST|MobileNetV3|78.2%|79.1%|78.65%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar)|
|DB|ResNet50_vd|86.41%|78.72%|82.38%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|77.29%|73.08%|75.12%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|SAST|ResNet50_vd|91.39%|83.77%|87.42%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|
|PSE|ResNet50_vd|85.81%|79.53%|82.55%|[trianed model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar)|
|PSE|MobileNetV3|82.20%|70.48%|75.89%|[trianed model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar)|

On Total-Text dataset, the text detection result is as follows:

|Model|Backbone|Precision|Recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|89.63%|78.44%|83.66%|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|

**Note：** Additional data, like icdar2013, icdar2017, COCO-Text, ArT, was added to the model training of SAST. Download English public dataset in organized format used by PaddleOCR from:
* [Baidu Drive](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (download code: 2bpi).
* [Google Drive](https://drive.google.com/drive/folders/1ll2-XEVyCQLpJjawLDiRlvo_i4BqHCJe?usp=sharing)


<a name="12"></a>
### 1.2 Text Recognition Algorithms

Supported text recognition algorithms (Click the link to get the tutorial):
- [x]  [CRNN](./algorithm_rec_crnn_en.md)
- [x]  [Rosetta](./algorithm_rec_rosetta_en.md)
- [x]  [STAR-Net](./algorithm_rec_starnet_en.md)
- [x]  [RARE](./algorithm_rec_rare_en.md)
- [x]  [SRN](./algorithm_rec_srn_en.md)
- [x]  [NRTR](./algorithm_rec_nrtr_en.md)
- [x]  [SAR](./algorithm_rec_sar_en.md)
- [x]  [SEED](./algorithm_rec_seed_en.md)
- [x]  [SVTR](./algorithm_rec_svtr_en.md)

Refer to [DTRB](https://arxiv.org/abs/1904.01906), the training and evaluation result of these above text recognition (using MJSynth and SynthText for training, evaluate on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) is as follow:

|Model|Backbone|Avg Accuracy|Module combination|Download link|
|---|---|---|---|---|
|Rosetta|Resnet34_vd|79.11%|rec_r34_vd_none_none_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|75.80%|rec_mv3_none_none_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|
|CRNN|Resnet34_vd|81.04%|rec_r34_vd_none_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar)|
|CRNN|MobileNetV3|77.95%|rec_mv3_none_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar)|
|StarNet|Resnet34_vd|82.85%|rec_r34_vd_tps_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|79.28%|rec_mv3_tps_bilstm_ctc|[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|
|RARE|Resnet34_vd|83.98%|rec_r34_vd_tps_bilstm_att |[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar)|
|RARE|MobileNetV3|81.76%|rec_mv3_tps_bilstm_att |[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar)|
|SRN|Resnet50_vd_fpn| 86.31% | rec_r50fpn_vd_none_srn |[trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar)|
|NRTR|NRTR_MTB| 84.21% | rec_mtb_nrtr | [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) |
|SAR|Resnet31| 87.20% | rec_r31_sar | [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_sar_train.tar) |
|SEED|Aster_Resnet| 85.35% | rec_resnet_stn_bilstm_att | [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_resnet_stn_bilstm_att.tar) |
|SVTR|SVTR-Tiny| 89.25% | rec_svtr_tiny_none_ctc_en | [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar) |


<a name="2"></a>

## 2. End-to-end Algorithms

Supported end-to-end algorithms (Click the link to get the tutorial):
- [x]  [PGNet](./algorithm_e2e_pgnet_en.md)
