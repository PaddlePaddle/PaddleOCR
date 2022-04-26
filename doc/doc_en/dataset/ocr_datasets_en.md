## OCR datasets

- [1. text detection](#1)
- [2. text recognition](#2)

Here is a list of public datasets commonly used in OCR, which are being continuously updated. Welcome to contribute datasets~

<a name="1"></a>
#### 1. text detection

| dataset | Image download link | PPOCR format annotation download link |
|---|---|---|
| ICDAR 2015 | https://rrc.cvc.uab.es/?ch=4&com=downloads            | [train](https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt) / [test](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt) |
| ctw1500 | https://paddleocr.bj.bcebos.com/dataset/ctw1500.zip   | Included in the downloaded image zip                                                                                                           |
| total text | https://paddleocr.bj.bcebos.com/dataset/total_text.tar |  Included in the downloaded image zip                                                                                                                                     |

<a name="2"></a>
#### 2. text recognition

| dataset | Image download link | PPOCR format annotation download link |
|---|---|---|
| en benchmark(MJ, SJ, IIIT, SVT, IC03, IC13, IC15, SVTP, and CUTE.) | [DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | LMDB format, which can be loaded directly with [lmdb_dataset.py](../../../ppocr/data/lmdb_dataset.py) |
