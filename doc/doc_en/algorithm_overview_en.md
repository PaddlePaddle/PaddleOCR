<a name="Algorithm_introduction"></a>
## Algorithm introduction

This tutorial lists the text detection algorithms and text recognition algorithms supported by PaddleOCR, as well as the models and metrics of each algorithm on **English public datasets**. It is mainly used for algorithm introduction and algorithm performance comparison. For more models on other datasets including Chinese, please refer to [PP-OCR v2.0 models list](./models_list_en.md).


- [1. Text Detection Algorithm](#TEXTDETECTIONALGORITHM)
- [2. Text Recognition Algorithm](#TEXTRECOGNITIONALGORITHM)

<a name="TEXTDETECTIONALGORITHM"></a>
### 1. Text Detection Algorithm

PaddleOCR open source text detection algorithms list:
- [x]  EAST([paper](https://arxiv.org/abs/1704.03155))[2]
- [x]  DB([paper](https://arxiv.org/abs/1911.08947))[1]
- [x]  SAST([paper](https://arxiv.org/abs/1908.05498))[4]

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|precision|recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|EAST|ResNet50_vd|88.76%|81.36%|84.90%|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_east_v2.0_train.tar)|
|EAST|MobileNetV3|78.24%|79.15%|78.69%|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_east_v2.0_train.tar)|
|DB|ResNet50_vd|86.41%|78.72%|82.38%|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|77.29%|73.08%|75.12%|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|
|SAST|ResNet50_vd|91.39%|83.77%|87.42%|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_icdar15_v2.0_train.tar)|

On Total-Text dataset, the text detection result is as follows:

|Model|Backbone|precision|recall|Hmean|Download link|
| --- | --- | --- | --- | --- | --- |
|SAST|ResNet50_vd|89.63%|78.44%|83.66%|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_sast_totaltext_v2.0_train.tar)|

**Note：** Additional data, like icdar2013, icdar2017, COCO-Text, ArT, was added to the model training of SAST. Download English public dataset in organized format used by PaddleOCR from [Baidu Drive](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (download code: 2bpi).

For the training guide and use of PaddleOCR text detection algorithms, please refer to the document [Text detection model training/evaluation/prediction](./detection_en.md)

<a name="TEXTRECOGNITIONALGORITHM"></a>
### 2. Text Recognition Algorithm

PaddleOCR open-source text recognition algorithms list:
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))[7]
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))[10]
- [ ]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))[11] coming soon
- [ ]  RARE([paper](https://arxiv.org/abs/1603.03915v1))[12] coming soon
- [ ]  SRN([paper](https://arxiv.org/abs/2003.12294))[5] coming soon

Refer to [DTRB](https://arxiv.org/abs/1904.01906), the training and evaluation result of these above text recognition (using MJSynth and SynthText for training, evaluate on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) is as follow:

|Model|Backbone|Avg Accuracy|Module combination|Download link|
|-|-|-|-|-|
|Rosetta|Resnet34_vd|80.9%|rec_r34_vd_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|78.05%|rec_mv3_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|
|CRNN|Resnet34_vd|82.76%|rec_r34_vd_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar)|
|CRNN|MobileNetV3|79.97%|rec_mv3_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar)|

Please refer to the document for training guide and use of PaddleOCR text recognition algorithms [Text recognition model training/evaluation/prediction](./recognition_en.md)
