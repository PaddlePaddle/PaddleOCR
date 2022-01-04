

# Key Information Extraction

This section will introduce a quickly use and train keynotes extraction(KIE)  using SDMGR in PaddleOCR.

SDMGR is a Key Information Extraction algorithm that is classifies each detected text area into predefined categories, such as order ID, invoice number, amount etc.



* [1. Quick Start](#1-----)
* [2. Execution Training](#2-----)
* [3. Execution Evaluation](#3-----)

<a name="1-----"></a>
## 1. Quick Start

The Training and Evaluation data using wildreceipt dataset, It using as below commnad download:

```
wget https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/wildreceipt.tar && tar xf wildreceipt.tar
```

Evaluation：

```
cd PaddleOCR/
wget https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar && tar xf kie_vgg16.tar
python3.7 tools/infer_kie.py -c configs/kie/kie_unet_sdmgr.yml -o Global.checkpoints=kie_vgg16/best_accuracy  Global.infer_img=../wildreceipt/1.txt
```

The prediction result is saved as the folder`./output/sdmgr_kie/predicts_kie.txt`，the visualization result is saved as the folder`/output/sdmgr_kie/kie_results/`。

Visualization Result Figures：

<div align="center">
    <img src="./imgs/0.png" width="800">
</div>

<a name="2-----"></a>
## 2. Execution Training

Building dataset link to PaddleOCR/train_data:

```
cd PaddleOCR/ && mkdir train_data && cd train_data
 ln -s ../../wildreceipt ./
```
The training using configure file is configs/kie/kie_unet_sdmgr.yml，configure file default train path is `train_data/wildreceipt`， when prepare train data using as below command execute
training:
```
python3.7 tools/train.py -c configs/kie/kie_unet_sdmgr.yml -o Global.save_model_dir=./output/kie/
```
<a name="3-----"></a>
## 3. Execution Evaluation

```
python3.7 tools/eval.py -c configs/kie/kie_unet_sdmgr.yml -o Global.checkpoints=./output/kie/best_accuracy
```


**Reference Documents：**

<!-- [ALGORITHM] -->

```bibtex
@misc{sun2021spatial,
      title={Spatial Dual-Modality Graph Reasoning for Key Information Extraction},
      author={Hongbin Sun and Zhanghui Kuang and Xiaoyu Yue and Chenhao Lin and Wayne Zhang},
      year={2021},
      eprint={2103.14470},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
