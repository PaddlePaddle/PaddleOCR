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
wget https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/wildreceipt.tar && tar xf wildreceipt.tar
```

The dataset format are as follows:
```
./wildreceipt
├── class_list.txt          # The text category inside the box, such as amount, time, date, etc.
├── dict.txt                # A recognized dictionary file, a list of characters contained in the dataset
├── wildreceipt_train.txt   # training data label file
└── wildreceipt_test.txt    # testing data label file
└── image_files/            # image dataset file
```

The format in the label file is:
```
" The image file path                    Image annotation information encoded by json.dumps"
image_files/Image_16/11/d5de7f2a20751e50b84c747c17a24cd98bed3554.jpeg    [{"label": 1, "transcription": "SAFEWAY", "points": [[550.0, 190.0], [937.0, 190.0], [937.0, 104.0], [550.0, 104.0]]}, {"label": 25, "transcription": "TM", "points": [[1048.0, 211.0], [1074.0, 211.0], [1074.0, 196.0], [1048.0, 196.0]]}, {"label": 25, "transcription": "ATOREMGRTOMMILAZZO", "points": [[535.0, 239.0], [833.0, 239.0], [833.0, 200.0], [535.0, 200.0]]}, {"label": 5, "transcription": "703-777-5833", "points": [[907.0, 256.0], [1081.0, 256.0], [1081.0, 223.0], [907.0, 223.0]]}......
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
    <img src="./imgs/0.png" width="800">
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
