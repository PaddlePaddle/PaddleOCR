<a name="Algorithm_introduction"></a>
## Algorithm introduction

This tutorial lists the text detection algorithms and text recognition algorithms supported by PaddleOCR, as well as the models and metrics of each algorithm on **English public datasets**. It is mainly used for algorithm introduction and algorithm performance comparison. For more models on other datasets including Chinese, please refer to [PP-OCR v2.0 models list](./models_list_en.md).


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
|EAST|ResNet50_vd|88.18%|85.51%|86.82%|[Download link](link)|
|EAST|MobileNetV3|81.67%|79.83%|80.74%|[Download link](link)|
|DB|ResNet50_vd|83.79%|80.65%|82.19%|[Download link](link)|
|DB|MobileNetV3|75.92%|73.18%|74.53%|[Download link](link)|
|SAST|ResNet50_vd|92.18%|82.96%|87.33%|[Download link](link)|

On Total-Text dataset, the text detection result is as follows:

|Model|Backbone|precision|recall|Hmean|Download link|
|-|-|-|-|-|-|
|SAST|ResNet50_vd|88.74%|79.80%|84.03%|[Download link](link)|

**Note：** Additional data, like icdar2013, icdar2017, COCO-Text, ArT, was added to the model training of SAST. Download English public dataset in organized format used by PaddleOCR from [Baidu Drive](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (download code: 2bpi).

For the training guide and use of PaddleOCR text detection algorithms, please refer to the document [Text detection model training/evaluation/prediction](./doc/doc_en/detection_en.md)

<a name="TEXTRECOGNITIONALGORITHM"></a>
### 2. Text Recognition Algorithm

PaddleOCR open-source text recognition algorithms list:
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))
- [x]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))
- [ ]  RARE([paper](https://arxiv.org/abs/1603.03915v1)) coming soon
- [ ]  SRN([paper](https://arxiv.org/abs/2003.12294))(Baidu Self-Research) coming soon

Refer to [DTRB](https://arxiv.org/abs/1904.01906), the training and evaluation result of these above text recognition (using MJSynth and SynthText for training, evaluate on IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE) is as follow:

|Model|Backbone|Avg Accuracy|Module combination|Download link|
|-|-|-|-|-|
|Rosetta|Resnet34_vd|80.24%|rec_r34_vd_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_none_ctc.tar)|
|Rosetta|MobileNetV3|78.16%|rec_mv3_none_none_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_none_none_ctc.tar)|
|CRNN|Resnet34_vd|82.20%|rec_r34_vd_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_bilstm_ctc.tar)|
|CRNN|MobileNetV3|79.37%|rec_mv3_none_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_none_bilstm_ctc.tar)|
|STAR-Net|Resnet34_vd|83.93%|rec_r34_vd_tps_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_ctc.tar)|
|STAR-Net|MobileNetV3|81.56%|rec_mv3_tps_bilstm_ctc|[Download link](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_ctc.tar)|


Please refer to the document for training guide and use of PaddleOCR text recognition algorithms [Text recognition model training/evaluation/prediction](./doc/doc_en/recognition_en.md)
