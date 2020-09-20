<a name="算法介绍"></a>
## 算法介绍
- [1.文本检测算法](#文本检测算法)
- [2.文本识别算法](#文本识别算法)

<a name="文本检测算法"></a>
### 1.文本检测算法

PaddleOCR开源的文本检测算法列表：
- [x]  DB([paper](https://arxiv.org/abs/1911.08947))（ppocr推荐）
- [x]  EAST([paper](https://arxiv.org/abs/1704.03155))
- [x]  SAST([paper](https://arxiv.org/abs/1908.05498))

在ICDAR2015文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
|-|-|-|-|-|-|
|EAST|ResNet50_vd|88.18%|85.51%|86.82%|[下载链接](https://paddleocr.bj.bcebos.com/det_r50_vd_east.tar)|
|EAST|MobileNetV3|81.67%|79.83%|80.74%|[下载链接](https://paddleocr.bj.bcebos.com/det_mv3_east.tar)|
|DB|ResNet50_vd|83.79%|80.65%|82.19%|[下载链接](https://paddleocr.bj.bcebos.com/det_r50_vd_db.tar)|
|DB|MobileNetV3|75.92%|73.18%|74.53%|[下载链接](https://paddleocr.bj.bcebos.com/det_mv3_db.tar)|
|SAST|ResNet50_vd|92.18%|82.96%|87.33%|[下载链接](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_icdar2015.tar)|

在Total-text文本检测公开数据集上，算法效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
|-|-|-|-|-|-|
|SAST|ResNet50_vd|88.74%|79.80%|84.03%|[下载链接](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_total_text.tar)|

**说明：** SAST模型训练额外加入了icdar2013、icdar2017、COCO-Text、ArT等公开数据集进行调优。PaddleOCR用到的经过整理格式的英文公开数据集下载：[百度云地址](https://pan.baidu.com/s/12cPnZcVuV1zn5DOd4mqjVw) (提取码: 2bpi)


使用[LSVT](./datasets.md#1icdar2019-lsvt)街景数据集共3w张数据，训练中文检测模型的相关配置和预训练文件如下：

|模型|骨干网络|配置文件|预训练模型|
|-|-|-|-|
|超轻量中文模型|MobileNetV3|det_mv3_db.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|
|通用中文OCR模型|ResNet50_vd|det_r50_vd_db.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|

* 注： 上述DB模型的训练和评估，需设置后处理参数box_thresh=0.6，unclip_ratio=1.5，使用不同数据集、不同模型训练，可调整这两个参数进行优化

PaddleOCR文本检测算法的训练和使用请参考文档教程中[模型训练/评估中的文本检测部分](./detection.md)。

<a name="文本识别算法"></a>
### 2.文本识别算法

PaddleOCR开源的文本识别算法列表：
- [x]  CRNN([paper](https://arxiv.org/abs/1507.05717))（ppocr推荐）
- [x]  Rosetta([paper](https://arxiv.org/abs/1910.05085))
- [x]  STAR-Net([paper](http://www.bmva.org/bmvc/2016/papers/paper043/index.html))
- [x]  RARE([paper](https://arxiv.org/abs/1603.03915v1))
- [x]  SRN([paper](https://arxiv.org/abs/2003.12294))

参考[DTRB](https://arxiv.org/abs/1904.01906)文字识别训练和评估流程，使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法效果如下：

|模型|骨干网络|Avg Accuracy|模型存储命名|下载链接|
|-|-|-|-|-|
|Rosetta|Resnet34_vd|80.24%|rec_r34_vd_none_none_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_none_ctc.tar)|
|Rosetta|MobileNetV3|78.16%|rec_mv3_none_none_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_none_none_ctc.tar)|
|CRNN|Resnet34_vd|82.20%|rec_r34_vd_none_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_none_bilstm_ctc.tar)|
|CRNN|MobileNetV3|79.37%|rec_mv3_none_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_none_bilstm_ctc.tar)|
|STAR-Net|Resnet34_vd|83.93%|rec_r34_vd_tps_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_ctc.tar)|
|STAR-Net|MobileNetV3|81.56%|rec_mv3_tps_bilstm_ctc|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_ctc.tar)|
|RARE|Resnet34_vd|84.90%|rec_r34_vd_tps_bilstm_attn|[下载链接](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_attn.tar)|
|RARE|MobileNetV3|83.32%|rec_mv3_tps_bilstm_attn|[下载链接](https://paddleocr.bj.bcebos.com/rec_mv3_tps_bilstm_attn.tar)|
|SRN|Resnet50_vd_fpn|88.33%|rec_r50fpn_vd_none_srn|[下载链接](https://paddleocr.bj.bcebos.com/SRN/rec_r50fpn_vd_none_srn.tar)|

**说明：** SRN模型使用了数据扰动方法对上述提到对两个训练集进行增广，增广后的数据可以在[百度网盘](https://pan.baidu.com/s/1-HSZ-ZVdqBF2HaBZ5pRAKA)上下载，提取码: y3ry。
原始论文使用两阶段训练平均精度为89.74%，PaddleOCR中使用one-stage训练，平均精度为88.33%。两种预训练权重均在[下载链接](https://paddleocr.bj.bcebos.com/SRN/rec_r50fpn_vd_none_srn.tar)中。

使用[LSVT](./datasets.md#1icdar2019-lsvt)街景数据集根据真值将图crop出来30w数据，进行位置校准。此外基于LSVT语料生成500w合成数据训练中文模型，相关配置和预训练文件如下：  

|模型|骨干网络|配置文件|预训练模型|
|-|-|-|-|
|超轻量中文模型|MobileNetV3|rec_chinese_lite_train.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|
|通用中文OCR模型|Resnet34_vd|rec_chinese_common_train.yml|[下载链接](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|

PaddleOCR文本识别算法的训练和使用请参考文档教程中[模型训练/评估中的文本识别部分](./recognition.md)。
