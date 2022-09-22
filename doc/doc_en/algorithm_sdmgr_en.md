- [Key Information Extraction(KIE)](#key-information-extractionkie)
  - [1. Quick Use](#1-quick-use)
  - [2. Model Training](#2-model-training)
  - [3. Model Evaluation](#3-model-evaluation)
  - [4. Reference](#4-reference)

# Key Information Extraction(KIE)

This section provides a tutorial example on how to quickly use, train, and evaluate a key information extraction(KIE) model, [SDMGR](https://arxiv.org/abs/2103.14470), in PaddleOCR.

[SDMGR(Spatial Dual-Modality Graph Reasoning)](https://arxiv.org/abs/2103.14470) is a KIE algorithm that classifies each detected text region into predefined categories, such as order ID, invoice number, amount, and etc.

## 1. Quick Use

[Wildreceipt dataset](https://paperswithcode.com/dataset/wildreceipt) is used for this tutorial. It contains 1765 photos, with 25 classes, and 50000 text boxes, which can be downloaded by wget:

```shell
wget https://paddleocr.bj.bcebos.com/ppstructure/dataset/wildreceipt.tar && tar xf wildreceipt.tar
```

Download the pretrained model and predict the result:

```shell
cd PaddleOCR/
wget https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar && tar xf kie_vgg16.tar
python3.7 tools/infer_kie.py -c configs/kie/kie_unet_sdmgr.yml -o Global.checkpoints=kie_vgg16/best_accuracy  Global.infer_img=../wildreceipt/1.txt
```

The prediction result is saved as `./output/sdmgr_kie/predicts_kie.txt`, and the visualization results are saved in the folder`/output/sdmgr_kie/kie_results/`.

The visualization results are shown in the figure below:

<div align="center">
    <img src="../../ppstructure/docs/imgs/sdmgr_result.png" width="800">
</div>

## 2. Model Training

Create a softlink to the folder, `PaddleOCR/train_data`:
```shell
cd PaddleOCR/ && mkdir train_data && cd train_data

ln -s ../../wildreceipt ./
```

The configuration file used for training is `configs/kie/kie_unet_sdmgr.yml`. The default training data path in the configuration file is `train_data/wildreceipt`. After preparing the data, you can execute the model training with the following command:
```shell
python3.7 tools/train.py -c configs/kie/kie_unet_sdmgr.yml -o Global.save_model_dir=./output/kie/
```

## 3. Model Evaluation

After training, you can execute the model evaluation with the following command:

```shell
python3.7 tools/eval.py -c configs/kie/kie_unet_sdmgr.yml -o Global.checkpoints=./output/kie/best_accuracy
```

## 4. Reference

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
