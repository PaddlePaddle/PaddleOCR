# OCR算法

- [1. 两阶段算法](#1-两阶段算法)
    - [1.1 文本检测算法](#11-文本检测算法)
    - [1.2 文本识别算法](#12-文本识别算法)
- [2. 端到端算法](#2-端到端算法)


本文给出了PaddleOCR已支持的OCR算法列表，以及每个算法在**英文公开数据集**上的模型和指标，主要用于算法简介和算法性能对比，更多包括中文在内的其他数据集上的模型请参考[PP-OCR v2.0 系列模型下载](./models_list.md)。

<a name="1"></a>

## 1. 两阶段算法

<a name="11"></a>

### 1.1 文本检测算法

已支持的文本检测算法列表（戳链接获取使用教程）：
- [x]  [DB](./algorithm_det_db.md)
- [x]  [EAST](./algorithm_det_east.md)
- [x]  [SAST](./algorithm_det_sast.md)
- [x]  [PSENet](./algorithm_det_psenet.md)
- [x]  [FCENet](./algorithm_det_fcenet.md)

在ICDAR2015文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- |
|EAST|ResNet50_vd|88.71%|81.36%|84.88%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)|
|EAST|MobileNetV3|78.2%|79.1%|78.65%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar)|
|DB|ResNet50_vd|86.41%|78.72%|82.38%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|77.29%|73.08%|75.12%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|SAST|ResNet50_vd|91.39%|83.77%|87.42%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|
|PSE|ResNet50_vd|85.81%|79.53%|82.55%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar)|
|PSE|MobileNetV3|82.20%|70.48%|75.89%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar)|

在Total-text文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|89.63%|78.44%|83.66%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|

在CTW1500文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- |         
|FCE|ResNet50_dcn|88.39%|82.18%|85.27%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/det_r50_dcn_fce_ctw_v2.0_train.tar)|

**说明：** SAST模型训练额外加入了icdar2013、icdar2017、COCO-Text、ArT等公开数据集进行调优。PaddleOCR用到的经过整理格式的英文公开数据集下载：
* [百度云地址](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (提取码: 2bpi)
* [Google Drive下载地址](https://drive.google.com/drive/folders/1ll2-XEVyCQLpJjawLDiRlvo_i4BqHCJe?usp=sharing)


<a name="12"></a>

### 1.2 文本识别算法

已支持的文本识别算法列表（戳链接获取使用教程）：
- [x]  [CRNN](./algorithm_rec_crnn.md)
- [x]  [Rosetta](./algorithm_rec_rosetta.md)
- [x]  [STAR-Net](./algorithm_rec_starnet.md)
- [x]  [RARE](./algorithm_rec_rare.md)
- [x]  [SRN](./algorithm_rec_srn.md)
- [x]  [NRTR](./algorithm_rec_nrtr.md)
- [x]  [SAR](./algorithm_rec_sar.md)
- [x]  [SEED](./algorithm_rec_seed.md)

参考[DTRB](https://arxiv.org/abs/1904.01906)[3]文字识别训练和评估流程，使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法效果如下：

|模型|骨干网络|Avg Accuracy|模型存储命名|下载链接|
|---|---|---|---|---|
|Rosetta|Resnet34_vd|79.11%|rec_r34_vd_none_none_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|75.80%|rec_mv3_none_none_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|
|CRNN|Resnet34_vd|81.04%|rec_r34_vd_none_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar)|
|CRNN|MobileNetV3|77.95%|rec_mv3_none_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar)|
|StarNet|Resnet34_vd|82.85%|rec_r34_vd_tps_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|79.28%|rec_mv3_tps_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|
|RARE|Resnet34_vd|83.98%|rec_r34_vd_tps_bilstm_att |[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar)|
|RARE|MobileNetV3|81.76%|rec_mv3_tps_bilstm_att |[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar)|
|SRN|Resnet50_vd_fpn| 86.31% | rec_r50fpn_vd_none_srn | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar) |
|NRTR|NRTR_MTB| 84.21% | rec_mtb_nrtr | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) |
|SAR|Resnet31| 87.20% | rec_r31_sar | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_sar_train.tar) |
|SEED|Aster_Resnet| 85.35% | rec_resnet_stn_bilstm_att | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_resnet_stn_bilstm_att.tar) |


<a name="2"></a>

## 2. 端到端算法

已支持的端到端OCR算法列表（戳链接获取使用教程）：
- [x]  [PGNet](./algorithm_e2e_pgnet.md)


