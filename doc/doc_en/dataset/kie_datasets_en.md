## Key Information Extraction dataset

Here are the common datasets key information extraction, which are being updated continuously. Welcome to contribute datasets.

- [FUNSD dataset](#funsd)
- [XFUND dataset](#xfund)
- [wildreceipt dataset](#wildreceipt-dataset)

<a name="funsd"></a>
#### 1. FUNSD dataset
- **Data source**: https://guillaumejaume.github.io/FUNSD/
- **Data Introduction**: The FUNSD dataset is a dataset for form comprehension. It contains 199 real, fully annotated scanned images, including market reports, advertisements, and academic reports, etc., and is divided into 149 training set and 50 test set. The FUNSD dataset is suitable for many types of DocVQA tasks, such as field-level entity classification, field-level entity connection, etc. Part of the image and the annotation box visualization are shown below:
<div align="center">
    <img src="../../datasets/funsd_demo/gt_train_00040534.jpg" width="500">
    <img src="../../datasets/funsd_demo/gt_train_00070353.jpg" width="500">
</div>
    In the figure, the orange area represents `header`, the light blue area represents `question`, the green area represents `answer`, and the pink area represents `other`.

- **Download address**: https://guillaumejaume.github.io/FUNSD/download/

<a name="xfund"></a>
#### 2. XFUND dataset
- **Data source**: https://github.com/doc-analysis/XFUND
- **Data introduction**: XFUND is a multilingual form comprehension dataset, which contains form data in 7 different languages, and all are manually annotated in the form of key-value pairs. The data for each language contains 199 form data, which are divided into 149 training sets and 50 test sets. Part of the image and the annotation box visualization are shown below.

<div align="center">
    <img src="../../datasets/xfund_demo/gt_zh_train_0.jpg" width="500">
    <img src="../../datasets/xfund_demo/gt_zh_train_1.jpg" width="500">
</div>

- **Download address**: https://github.com/doc-analysis/XFUND/releases/tag/v1.0

<a name="wildreceipt"></a>

## 3. wildreceipt dataset

- **Data source**: https://arxiv.org/abs/2103.14470
- **Data introduction**: wildreceipt is an English receipt dataset, which contains 26 different categories. There are 1267 training images and 472 evaluation images, in which 50,000 textlines and boxes are annotated. Part of the image and the annotation box visualization are shown below.

<div align="center">
    <img src="../../datasets/wildreceipt_demo/2769.jpeg" width="500">
    <img src="../../datasets/wildreceipt_demo/1bbe854b8817dedb8585e0732089fd1f752d2cec.jpeg" width="500">
</div>

**Note：** Boxes with category `Ignore` or `Others` are not visualized here.

- **Download address**：
    - Offical dataset: [link](https://download.openmmlab.com/mmocr/data/wildreceipt.tar)
    - Dataset converted for PaddleOCR training process: [link](https://paddleocr.bj.bcebos.com/ppstructure/dataset/wildreceipt.tar)
