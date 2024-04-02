# Table Recognition Algorithm-TableMASTER

- [1. Introduction](#1-introduction)
- [2. Environment](#2-environment)
- [3. Model Training / Evaluation / Prediction](#3-model-training--evaluation--prediction)
- [4. Inference and Deployment](#4-inference-and-deployment)
  - [4.1 Python Inference](#41-python-inference)
  - [4.2 C++ Inference](#42-c-inference)
  - [4.3 Serving](#43-serving)
  - [4.4 More](#44-more)
- [5. FAQ](#5-faq)
- [Citation](#citation)

<a name="1"></a>
## 1. Introduction

Paper:
> [TableMaster: PINGAN-VCGROUP’S SOLUTION FOR ICDAR 2021 COMPETITION ON SCIENTIFIC LITERATURE PARSING TASK B: TABLE RECOGNITION TO HTML](https://arxiv.org/pdf/2105.01848.pdf)
> Ye, Jiaquan and Qi, Xianbiao and He, Yelin and Chen, Yihao and Gu, Dengyi and Gao, Peng and Xiao, Rong
> 2021


On the PubTabNet table recognition public data set, the algorithm reproduction acc is as follows:

|Model|Backbone|Cnnfig|Acc|Download link|
| --- | --- | --- | --- | --- |
|TableMaster|TableResNetExtra|[configs/table/table_master.yml](../../configs/table/table_master.yml)|77.47%|[trained model](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_train.tar)/[inference model](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_infer.tar)|


<a name="2"></a>
## 2. Environment
Please refer to ["Environment Preparation"](./environment_en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](./clone_en.md) to clone the project code.


<a name="3"></a>
## 3. Model Training / Evaluation / Prediction

The above TableMaster model is trained using the PubTabNet table recognition public dataset. For the download of the dataset, please refer to [table_datasets](./dataset/table_datasets_en.md).

After the data download is complete, please refer to [Text Recognition Training Tutorial](./recognition_en.md) for training. PaddleOCR has modularized the code structure, so that you only need to **replace the configuration file** to train different models.


<a name="4"></a>
## 4. Inference and Deployment

<a name="4-1"></a>
### 4.1 Python Inference

First, convert the model saved in the TableMaster table recognition training process into an inference model. Taking the model based on the TableResNetExtra backbone network and trained on the PubTabNet dataset as example ([model download link](https://paddleocr.bj.bcebos.com/contribution/table_master.tar)), you can use the following command to convert:


```shell
python3 tools/export_model.py -c configs/table/table_master.yml -o Global.pretrained_model=output/table_master/best_accuracy Global.save_inference_dir=./inference/table_master
```

**Note: **
- If you trained the model on your own dataset and adjusted the dictionary file, please pay attention to whether the `character_dict_path` in the modified configuration file is the correct dictionary file


Execute the following command for model inference:

```shell
cd ppstructure/
# When predicting all images in a folder, you can modify image_dir to a folder, such as --image_dir='docs/table'.
python3.7 table/predict_structure.py --table_model_dir=../output/table_master/table_structure_tablemaster_infer/ --table_algorithm=TableMaster --table_char_dict_path=../ppocr/utils/dict/table_master_structure_dict.txt --table_max_len=480 --image_dir=docs/table/table.jpg

```

After executing the command, the prediction results of the above image (structural information and the coordinates of each cell in the table) are printed to the screen, and the visualization of the cell coordinates is also saved. An example is as follows:

result：
```shell
[2022/06/16 13:06:54] ppocr INFO: result: ['<html>', '<body>', '<table>', '<thead>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</thead>', '<tbody>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</tbody>', '</table>', '</body>', '</html>'], [[72.17591094970703, 10.759100914001465, 60.29658508300781, 16.6805362701416], [161.85562133789062, 10.884308815002441, 14.9495210647583, 16.727018356323242], [277.79876708984375, 29.54340362548828, 31.490320205688477, 18.143272399902344],
...
[336.11724853515625, 280.3601989746094, 39.456939697265625, 18.121286392211914]]
[2022/06/16 13:06:54] ppocr INFO: save vis result to ./output/table.jpg
[2022/06/16 13:06:54] ppocr INFO: Predict time of docs/table/table.jpg: 17.36806297302246
```

**Note**:

- TableMaster is relatively slow during inference, and it is recommended to use GPU for use.

<a name="4-2"></a>
### 4.2 C++ Inference

Since the post-processing is not written in CPP, the TableMaster does not support CPP inference.


<a name="4-3"></a>
### 4.3 Serving

Not supported

<a name="4-4"></a>
### 4.4 More

Not supported

<a name="5"></a>
## 5. FAQ

## Citation

```bibtex
@article{ye2021pingan,
  title={PingAn-VCGroup's Solution for ICDAR 2021 Competition on Scientific Literature Parsing Task B: Table Recognition to HTML},
  author={Ye, Jiaquan and Qi, Xianbiao and He, Yelin and Chen, Yihao and Gu, Dengyi and Gao, Peng and Xiao, Rong},
  journal={arXiv preprint arXiv:2105.01848},
  year={2021}
}
```
