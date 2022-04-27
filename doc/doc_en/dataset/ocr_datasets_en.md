## OCR datasets

- [1. text detection](#1)
  - [1.1 ICDAR 2015](#11)
- [2. text recognition](#2)

Here is a list of public datasets commonly used in OCR, which are being continuously updated. Welcome to contribute datasets~

<a name="1"></a>
#### 1. text detection

| dataset | Image download link | PPOCR format annotation download link |
|---|---|---|
| ICDAR 2015 | https://rrc.cvc.uab.es/?ch=4&com=downloads            | [train](https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt) / [test](https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt) |
| ctw1500 | https://paddleocr.bj.bcebos.com/dataset/ctw1500.zip   | Included in the downloaded image zip                                                                                                           |
| total text | https://paddleocr.bj.bcebos.com/dataset/total_text.tar |  Included in the downloaded image zip                                                                                                                                     |

<a name="11"></a>
#### 1.1 ICDAR 2015

The icdar2015 dataset contains train set which has 1000 images obtained with wearable cameras and test set which has 500 images obtained with wearable cameras. The icdar2015 dataset can be downloaded from the link in the table above. Registration is required for downloading.


After registering and logging in, download the part marked in the red box in the figure below. And, the content downloaded by `Training Set Images` should be saved as the folder `icdar_c4_train_imgs`, and the content downloaded by `Test Set Images` is saved as the folder `ch4_test_images`

<p align="center">
 <img src="../../datasets/ic15_location_download.png" align="middle" width = "700"/>
<p align="center">

Decompress the downloaded dataset to the working directory, assuming it is decompressed under PaddleOCR/train_data/. Then download the PPOCR format annotation file from the table above.

PaddleOCR also provides a data format conversion script, which can convert the official website label to the PPOCR format. The data conversion tool is in `ppocr/utils/gen_label.py`, here is the training set as an example:
```
# Convert the label file downloaded from the official website to train_icdar2015_label.txt
python gen_label.py --mode="det" --root_path="/path/to/icdar_c4_train_imgs/"  \
                    --input_path="/path/to/ch4_training_localization_transcription_gt" \
                    --output_label="/path/to/train_icdar2015_label.txt"
```

After decompressing the data set and downloading the annotation file, PaddleOCR/train_data/ has two folders and two files, which are:
```
/PaddleOCR/train_data/icdar2015/text_localization/
  └─ icdar_c4_train_imgs/         Training data of icdar dataset
  └─ ch4_test_images/             Testing data of icdar dataset
  └─ train_icdar2015_label.txt    Training annotation of icdar dataset
  └─ test_icdar2015_label.txt     Test annotation of icdar dataset
```


<a name="2"></a>
#### 2. text recognition

| dataset | Image download link | PPOCR format annotation download link |
|---|---|---|
| en benchmark(MJ, SJ, IIIT, SVT, IC03, IC13, IC15, SVTP, and CUTE.) | [DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here) | LMDB format, which can be loaded directly with [lmdb_dataset.py](../../../ppocr/data/lmdb_dataset.py) |
