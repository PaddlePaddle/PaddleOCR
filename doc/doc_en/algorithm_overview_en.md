<a name="Algorithm_introduction"></a>
## Algorithm introduction

This tutorial lists the text detection algorithms and text recognition algorithms supported by PaddleOCR, as well as the models and metrics of each algorithm on **English public datasets**. It is mainly used for algorithm introduction and algorithm performance comparison. For more models on other datasets including Chinese, please refer to [PP-OCR v1.1 models list](./models_list_en.md).


- [1. Text Detection Algorithm](#TEXTDETECTIONALGORITHM)
- [2. Text Recognition Algorithm](#TEXTRECOGNITIONALGORITHM)

<a name="TEXTDETECTIONALGORITHM"></a>
### 1. Text Detection Algorithm

PaddleOCR open source text detection algorithms list:
- [x]  EAST([paper](https://arxiv.org/abs/1704.03155))
- [x]  DB([paper](https://arxiv.org/abs/1911.08947))
- [x]  SAST([paper](https://arxiv.org/abs/1908.05498))(Baidu Self-Research)

On the ICDAR2015 dataset, the text detection result is as follows:

|Model|Backbone|precision|recall|Hmean|Download link|
|-|-|-|-|-|-|
|EAST|ResNet50_vd||||[Coming soon]()|
|EAST|MobileNetV3||||[Coming soon]()|
|DB|ResNet50_vd||||[Coming soon]()|
|DB|MobileNetV3||||[Coming soon]()|
|SAST|ResNet50_vd||||[Coming soon]()|

On Total-Text dataset, the text detection result is as follows:

|Model|Backbone|precision|recall|Hmean|Download link|
|-|-|-|-|-|-|
|SAST|ResNet50_vd||||[Coming soon]()|

**Note：** Additional data, like icdar2013, icdar2017, COCO-Text, ArT, was added to the model training of SAST. Download English public dataset in organized format used by PaddleOCR from [Baidu Drive](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (download code: 2bpi).

For the training guide and use of PaddleOCR text detection algorithms, please refer to the document [Text detection model training/evaluation/prediction](./doc/doc_en/detection_en.md)

<a name="TEXTRECOGNITIONALGORITHM"></a>
### 2. Text Recognition Algorithm

PaddleOCR open-source text recognition algorithms list:
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))
- [x]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))
- [x]  RARE([paper](https://arxiv.org/abs/1603.03915v1))
- [x]  SRN([paper](https://arxiv.org/abs/2003.12294))(Baidu Self-Research)

Refer to [DTRB](https://arxiv.org/abs/1904.01906), the training and evaluation result of these above text recognition (using MJSynth and SynthText for training, evaluate on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) is as follow:

|Model|Backbone|Avg Accuracy|Module combination|Download link|
|-|-|-|-|-|
|Rosetta|Resnet34_vd||rec_r34_vd_none_none_ctc|[Coming soon]()|
|Rosetta|MobileNetV3||rec_mv3_none_none_ctc|[Coming soon]()|
|CRNN|Resnet34_vd||rec_r34_vd_none_bilstm_ctc|[Coming soon]()|
|CRNN|MobileNetV3||rec_mv3_none_bilstm_ctc|[Coming soon]()|
|STAR-Net|Resnet34_vd||rec_r34_vd_tps_bilstm_ctc|[Coming soon]()|
|STAR-Net|MobileNetV3||rec_mv3_tps_bilstm_ctc|[Coming soon]()|
|RARE|Resnet34_vd||rec_r34_vd_tps_bilstm_attn|[Coming soon]()|
|RARE|MobileNetV3||rec_mv3_tps_bilstm_attn|[Coming soon]()|
|SRN|Resnet50_vd_fpn||rec_r50fpn_vd_none_srn|[Coming soon]()|

**Note：** SRN model uses data expansion method to expand the two training sets mentioned above, and the expanded data can be downloaded from [Baidu Drive](https://pan.baidu.com/s/1-HSZ-ZVdqBF2HaBZ5pRAKA) (download code: y3ry).

The average accuracy of the two-stage training in the original paper is 89.74%, and that of one stage training in paddleocr is 88.33%. Both pre-trained weights can be downloaded [here](https://paddleocr.bj.bcebos.com/SRN/rec_r50fpn_vd_none_srn.tar).

Please refer to the document for training guide and use of PaddleOCR text recognition algorithms [Text recognition model training/evaluation/prediction](./doc/doc_en/recognition_en.md)
