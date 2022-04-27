## OCR数据集

- [1. 文本检测](#1)
  - [1.1 ICDAR 2015](#11)
- [2. 文本识别](#2)

这里整理了OCR中常用的公开数据集，持续更新中，欢迎各位小伙伴贡献数据集～

<a name="1"></a>
### 1. 文本检测

| 数据集名称 |图片下载地址| PPOCR标注下载地址 |
|---|---|---|
| ICDAR 2015 |https://rrc.cvc.uab.es/?ch=4&com=downloads| [train](https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt) / [test](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt) |
| ctw1500 |https://paddleocr.bj.bcebos.com/dataset/ctw1500.zip| 图片下载地址中已包含 |
| total text |https://paddleocr.bj.bcebos.com/dataset/total_text.tar| 图片下载地址中已包含 |

<a name="11"></a>
#### 1.1 ICDAR 2015
icdar2015 数据集包含1000张训练图像和500张测试图像。icdar2015数据集可以从上表中链接下载，首次下载需注册。
注册完成登陆后，下载下图中红色框标出的部分，其中， `Training Set Images`下载的内容保存在`icdar_c4_train_imgs`文件夹下，`Test Set Images` 下载的内容保存早`ch4_test_images`文件夹下

<p align="center">
 <img src="../../datasets/ic15_location_download.png" align="middle" width = "700"/>
<p align="center">

将下载到的数据集解压到工作目录下，假设解压在 PaddleOCR/train_data/下。然后从上表中下载转换好的标注文件。

PaddleOCR 也提供了数据格式转换脚本，可以将官网 label 转换支持的数据格式。 数据转换工具在 `ppocr/utils/gen_label.py`, 这里以训练集为例：

```
# 将官网下载的标签文件转换为 train_icdar2015_label.txt
python gen_label.py --mode="det" --root_path="/path/to/icdar_c4_train_imgs/"  \
                    --input_path="/path/to/ch4_training_localization_transcription_gt" \
                    --output_label="/path/to/train_icdar2015_label.txt"
```

解压数据集和下载标注文件后，PaddleOCR/train_data/ 有两个文件夹和两个文件，按照如下方式组织icdar2015数据集：
```
/PaddleOCR/train_data/icdar2015/text_localization/
  └─ icdar_c4_train_imgs/         icdar 2015 数据集的训练数据
  └─ ch4_test_images/             icdar 2015 数据集的测试数据
  └─ train_icdar2015_label.txt    icdar 2015 数据集的训练标注
  └─ test_icdar2015_label.txt     icdar 2015 数据集的测试标注
```

<a name="2"></a>
### 2. 文本识别

| 数据集名称 | 图片下载地址 | PPOCR标注下载地址                                                         |
|---|---|---------------------------------------------------------------------|
| en benchmark(MJ, SJ, IIIT, SVT, IC03, IC13, IC15, SVTP, and CUTE.) | [DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | LMDB格式，可直接用[lmdb_dataset.py](../../../ppocr/data/lmdb_dataset.py)加载 |
