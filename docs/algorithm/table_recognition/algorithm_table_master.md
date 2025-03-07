---
typora-copy-images-to: images
comments: true
---


# 表格识别算法-TableMASTER

## 1. 算法简介

论文信息：
> [TableMaster: PINGAN-VCGROUP’S SOLUTION FOR ICDAR 2021 COMPETITION ON SCIENTIFIC LITERATURE PARSING TASK B: TABLE RECOGNITION TO HTML](https://arxiv.org/pdf/2105.01848.pdf)
> Ye, Jiaquan and Qi, Xianbiao and He, Yelin and Chen, Yihao and Gu, Dengyi and Gao, Peng and Xiao, Rong
> 2021

在PubTabNet表格识别公开数据集上，算法复现效果如下：

|模型|骨干网络|配置文件|acc|下载链接|
| --- | --- | --- | --- | --- |
|TableMaster|TableResNetExtra|[configs/table/table_master.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/table/table_master.yml)|77.47%|[训练模型](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_train.tar)/[推理模型](https://paddleocr.bj.bcebos.com/ppstructure/models/tablemaster/table_structure_tablemaster_infer.tar)|

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

上述TableMaster模型使用PubTabNet表格识别公开数据集训练得到，数据集下载可参考 [table_datasets](../../datasets/table_datasets.md)。

数据下载完成后，请参考[文本识别教程](../../ppocr/model_train/recognition.md)进行训练。PaddleOCR对代码进行了模块化，训练不同的模型只需要**更换配置文件**即可。

## 4. 推理部署

### 4.1 Python推理

首先将训练得到best模型，转换成inference model。以基于TableResNetExtra骨干网络，在PubTabNet数据集训练的模型为例([模型下载地址](https://paddleocr.bj.bcebos.com/contribution/table_master.tar))，可以使用如下命令进行转换：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c configs/table/table_master.yml -o Global.pretrained_model=output/table_master/best_accuracy Global.save_inference_dir=./inference/table_master
```

**注意：** 如果您是在自己的数据集上训练的模型，并且调整了字典文件，请注意修改配置文件中的`character_dict_path`是否为所正确的字典文件。

转换成功后，在目录下有三个文件：

```text linenums="1"
./inference/table_master/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```

执行如下命令进行模型推理：

```bash linenums="1"
cd ppstructure/
python3 table/predict_structure.py --table_model_dir=../output/table_master/table_structure_tablemaster_infer/ --table_algorithm=TableMaster --table_char_dict_path=../ppocr/utils/dict/table_master_structure_dict.txt --table_max_len=480 --image_dir=docs/table/table.jpg
# 预测文件夹下所有图像时，可修改image_dir为文件夹，如 --image_dir='docs/table'。
```

执行命令后，上面图像的预测结果（结构信息和表格中每个单元格的坐标）会打印到屏幕上，同时会保存单元格坐标的可视化结果。示例如下：
结果如下：

```bash linenums="1"
[2022/06/16 13:06:54] ppocr INFO: result: ['<html>', '<body>', '<table>', '<thead>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</thead>', '<tbody>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</tbody>', '</table>', '</body>', '</html>'], [[72.17591094970703, 10.759100914001465, 60.29658508300781, 16.6805362701416], [161.85562133789062, 10.884308815002441, 14.9495210647583, 16.727018356323242], [277.79876708984375, 29.54340362548828, 31.490320205688477, 18.143272399902344],
...
[336.11724853515625, 280.3601989746094, 39.456939697265625, 18.121286392211914]]
[2022/06/16 13:06:54] ppocr INFO: save vis result to ./output/table.jpg
[2022/06/16 13:06:54] ppocr INFO: Predict time of docs/table/table.jpg: 17.36806297302246
```

**注意**：

- TableMaster在推理时比较慢，建议使用GPU进行使用。

### 4.2 C++推理部署

由于C++预处理后处理还未支持TableMaster，所以暂未支持

### 4.3 Serving服务化部署

暂不支持

### 4.4 更多推理部署

暂不支持

## 5. FAQ

## 引用

```bibtex
@article{ye2021pingan,
  title={PingAn-VCGroup's Solution for ICDAR 2021 Competition on Scientific Literature Parsing Task B: Table Recognition to HTML},
  author={Ye, Jiaquan and Qi, Xianbiao and He, Yelin and Chen, Yihao and Gu, Dengyi and Gao, Peng and Xiao, Rong},
  journal={arXiv preprint arXiv:2105.01848},
  year={2021}
}
```
