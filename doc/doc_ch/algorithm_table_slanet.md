# 表格识别算法-SLANet-LCNetV2

- [1. 算法简介](#1-算法简介)
- [2. 环境配置](#2-环境配置)
- [3. 模型训练、评估、预测](#3-模型训练评估预测)
- [4. 推理部署](#4-推理部署)
  - [4.1 Python推理](#41-python推理)
  - [4.2 C++推理部署](#42-c推理部署)
  - [4.3 Serving服务化部署](#43-serving服务化部署)
  - [4.4 更多推理部署](#44-更多推理部署)
- [5. FAQ](#5-faq)

<a name="1"></a>
## 1. 算法简介

该算法由来自北京交通大学机器学习与认识计算研究团队的ocr识别队研发，其在PaddleOCR算法模型挑战赛 - 赛题二：通用表格识别任务中排行榜荣获一等奖，排行榜精度相比PP-Structure表格识别模型提升0.8%，推理速度提升3倍。优化思路如下：

- 1. 改善推理过程，至EOS停止，速度提升3倍
- 2. 升级Backbone为LCNetV2（SSLD版本）
- 3. 行列特征增强模块
- 4. 提升分辨率488至512
- 5. 三阶段训练策略

在PubTabNet表格识别公开数据集上，算法复现效果如下：

|模型|骨干网络|配置文件|acc|下载链接|
| --- | --- | --- | --- | --- |
|SLANet|LCNetV2|[configs/table/SLANet_lcnetv2.yml](../../configs/table/SLANet_lcnetv2.yml)|76.67%| [训练模型](https://paddleocr.bj.bcebos.com/openatom/ch_ppstructure_openatom_SLANetv2_train.tar) /[推理模型](https://paddleocr.bj.bcebos.com/openatom/ch_ppstructure_openatom_SLANetv2_infer.tar) |


<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

上述SLANet_LCNetv2模型使用PubTabNet表格识别公开数据集训练得到，数据集下载可参考 [table_datasets](./dataset/table_datasets.md)。

### 启动训练

数据下载完成后，请参考[文本识别教程](./recognition.md)进行训练。PaddleOCR对代码进行了模块化，训练不同的模型只需要**更换配置文件**即可。

训练命令如下：
```shell
# stage1
python3 -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/table/SLANet_lcnetv2.yml
# stage2 加载stage1的best model作为预训练模型，学习率调整为0.0001;
# stage3 加载stage2的best model作为预训练模型，不调整学习率，将配置文件中所有的488修改为512.
```

<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理
将训练得到best模型，转换成inference model，可以使用如下命令进行转换：

```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c configs/table/SLANet_lcnetv2.yml -o Global.pretrained_model=path/best_accuracy Global.save_inference_dir=./inference/slanet_lcnetv2_infer
```

**注意：**
- 如果您是在自己的数据集上训练的模型，并且调整了字典文件，请注意修改配置文件中的`character_dict_path`是否为所正确的字典文件。

转换成功后，在目录下有三个文件：
```
./inference/slanet_lcnetv2_infer/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```


执行如下命令进行模型推理：

```shell
cd ppstructure/
python table/predict_structure.py --table_model_dir=../inference/slanet_lcnetv2_infer/ --table_char_dict_path=../ppocr/utils/dict/table_structure_dict_ch.txt --image_dir=docs/table/table.jpg --output=../output/table_slanet_lcnetv2 --use_gpu=False --benchmark=True --enable_mkldnn=True --table_max_len=512
# 预测文件夹下所有图像时，可修改image_dir为文件夹，如 --image_dir='docs/table'。
```

执行命令后，上面图像的预测结果（结构信息和表格中每个单元格的坐标）会打印到屏幕上，同时会保存单元格坐标的可视化结果。示例如下：
结果如下：
```shell
[2022/06/16 13:06:54] ppocr INFO: result: ['<html>', '<body>', '<table>', '<thead>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</thead>', '<tbody>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</tbody>', '</table>', '</body>', '</html>'], [[72.17591094970703, 10.759100914001465, 60.29658508300781, 16.6805362701416], [161.85562133789062, 10.884308815002441, 14.9495210647583, 16.727018356323242], [277.79876708984375, 29.54340362548828, 31.490320205688477, 18.143272399902344],
...
[336.11724853515625, 280.3601989746094, 39.456939697265625, 18.121286392211914]]
[2022/06/16 13:06:54] ppocr INFO: save vis result to ./output/table.jpg
[2022/06/16 13:06:54] ppocr INFO: Predict time of docs/table/table.jpg: 17.36806297302246
```

<a name="4-2"></a>
### 4.2 C++推理部署

由于C++预处理后处理还未支持SLANet

<a name="4-3"></a>
### 4.3 Serving服务化部署

暂不支持

<a name="4-4"></a>
### 4.4 更多推理部署

暂不支持

<a name="5"></a>
## 5. FAQ
