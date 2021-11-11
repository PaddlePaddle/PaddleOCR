# 两阶段算法

- [两阶段算法](#-----)
  * [1. 算法介绍](#1)
    + [1.1 文本检测算法](#11)
    + [1.2 文本识别算法](#12)
  * [2. 模型训练](#2)
  * [3. 模型推理](#3)

<a name="1"></a>

## 1. 算法介绍
本文给出了PaddleOCR已支持的文本检测算法和文本识别算法列表，以及每个算法在**英文公开数据集**上的模型和指标，主要用于算法简介和算法性能对比，更多包括中文在内的其他数据集上的模型请参考[PP-OCR v2.0 系列模型下载](./models_list.md)。

<a name="11"></a>

### 1.1 文本检测算法

PaddleOCR开源的文本检测算法列表：
- [x]  DB([paper]( https://arxiv.org/abs/1911.08947)) [2]（ppocr推荐）
- [x]  EAST([paper](https://arxiv.org/abs/1704.03155))[1]
- [x]  SAST([paper](https://arxiv.org/abs/1908.05498))[4]
- [x]  PSENet([paper](https://arxiv.org/abs/1903.12473v2)）

在ICDAR2015文本检测公开数据集上，算法效果如下：
|模型|骨干网络|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- |
|EAST|ResNet50_vd|85.80%|86.71%|86.25%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)|
|EAST|MobileNetV3|79.42%|80.64%|80.03%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar)|
|DB|ResNet50_vd|86.41%|78.72%|82.38%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|77.29%|73.08%|75.12%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|SAST|ResNet50_vd|91.39%|83.77%|87.42%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|
|PSE|ResNet50_vd|85.81%|79.53%|82.55%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_r50_vd_pse_v2.0_train.tar)|
|PSE|MobileNetV3|82.20%|70.48%|75.89%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/en_det/det_mv3_pse_v2.0_train.tar)|

在Total-text文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|89.63%|78.44%|83.66%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|

**说明：** SAST模型训练额外加入了icdar2013、icdar2017、COCO-Text、ArT等公开数据集进行调优。PaddleOCR用到的经过整理格式的英文公开数据集下载：
* [百度云地址](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (提取码: 2bpi)
* [Google Drive下载地址](https://drive.google.com/drive/folders/1ll2-XEVyCQLpJjawLDiRlvo_i4BqHCJe?usp=sharing)

<a name="12"></a>

### 1.2 文本识别算法

PaddleOCR基于动态图开源的文本识别算法列表：
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))[7]（ppocr推荐）
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))[10]
- [x]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))[11]
- [x]  RARE([paper](https://arxiv.org/abs/1603.03915v1))[12]
- [x]  SRN([paper](https://arxiv.org/abs/2003.12294))[5]
- [x]  NRTR([paper](https://arxiv.org/abs/1806.00926v2))[13]
- [x]  SAR([paper](https://arxiv.org/abs/1811.00751v2))
- [x] SEED([paper](https://arxiv.org/pdf/2005.10977.pdf))

参考[DTRB][3](https://arxiv.org/abs/1904.01906)文字识别训练和评估流程，使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法效果如下：

|模型|骨干网络|Avg Accuracy|模型存储命名|下载链接|
|---|---|---|---|---|
|Rosetta|Resnet34_vd|80.9%|rec_r34_vd_none_none_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|78.05%|rec_mv3_none_none_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|
|CRNN|Resnet34_vd|82.76%|rec_r34_vd_none_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar)|
|CRNN|MobileNetV3|79.97%|rec_mv3_none_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar)|
|StarNet|Resnet34_vd|84.44%|rec_r34_vd_tps_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|81.42%|rec_mv3_tps_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|
|RARE|MobileNetV3|82.5%|rec_mv3_tps_bilstm_att |[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_att_v2.0_train.tar)|
|RARE|Resnet34_vd|83.6%|rec_r34_vd_tps_bilstm_att |[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_att_v2.0_train.tar)|
|SRN|Resnet50_vd_fpn| 88.52% | rec_r50fpn_vd_none_srn | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar) |
|NRTR|NRTR_MTB| 84.3% | rec_mtb_nrtr | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) |
|SAR|Resnet31| 87.2% | rec_r31_sar | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_r31_sar_train.tar) |
|SEED|Aster_Resnet| 85.2% | rec_resnet_stn_bilstm_att | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_resnet_stn_bilstm_att.tar) |

<a name="2"></a>

## 2. 模型训练

PaddleOCR文本检测算法的训练和使用请参考文档教程中[模型训练/评估中的文本检测部分](./detection.md)。文本识别算法的训练和使用请参考文档教程中[模型训练/评估中的文本识别部分](./recognition.md)。

<a name="3"></a>

## 3. 模型推理

上述模型中除PP-OCR系列模型以外，其余模型仅支持基于Python引擎的推理，具体内容可参考[基于Python预测引擎推理](./inference.md)
