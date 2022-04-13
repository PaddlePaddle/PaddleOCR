# DB

- [1. 算法简介](#1)
- [2. 环境配置](#2)
- [3. 快速使用](#3)
- [4. 模型训练、评估、预测](#4)
- [5. 推理部署](#5)
- [6. FAQ](#6)

<a name="1"></a>
## 1. 算法简介

论文信息：
> [Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)
> Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang
> AAAI, 2020

在ICDAR2015文本检测公开数据集上，算法复现效果如下：

|模型|骨干网络|precision|recall|Hmean|下载链接|
| --- | --- | --- | --- | --- | --- |
|DB|ResNet50_vd|86.41%|78.72%|82.38%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar)|
|DB|MobileNetV3|77.29%|73.08%|75.12%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar)|


<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目


<a name="3"></a>
## 3. 快速使用
参考本节，可以直接下载训好的模型，进行基于训练引擎的模型预测。

### 训练模型下载
根据第1节给出的模型列表，选择下载训练模型：
```bash
mkdir trained_models && cd trained_models
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_mv3_db_v2.0_train.tar && tar xf det_mv3_db_v2.0_train.tar
cd ..
```
* windows 环境下如果没有安装wget,下载模型时可将链接复制到浏览器中下载，并解压放置在相应目录下

解压完毕后应有如下文件结构：
```
├── det_mv3_db_v2.0_train
│   ├── best_accuracy.states
│   ├── best_accuracy.pdparams
│   ├── best_accuracy.pdopt
│   └── train.log
```
### 单张图像或者图像集合预测
```bash
# 预测image_dir指定的单张图像
python3 tools/infer/predict_e2e.py --e2e_algorithm="PGNet" --image_dir="./doc/imgs_en/img623.jpg" --e2e_model_dir="./inference/e2e_server_pgnetA_infer/" --e2e_pgnet_valid_set="totaltext"

# 预测image_dir指定的图像集合
python3 tools/infer/predict_e2e.py --e2e_algorithm="PGNet" --image_dir="./doc/imgs_en/" --e2e_model_dir="./inference/e2e_server_pgnetA_infer/" --e2e_pgnet_valid_set="totaltext"

# 如果想使用CPU进行预测，需设置use_gpu参数为False
python3 tools/infer/predict_e2e.py --e2e_algorithm="PGNet" --image_dir="./doc/imgs_en/img623.jpg" --e2e_model_dir="./inference/e2e_server_pgnetA_infer/" --e2e_pgnet_valid_set="totaltext" --use_gpu=False
```
### 可视化结果
可视化文本检测结果默认保存到./inference_results文件夹里面，结果文件的名称前缀为'e2e_res'。结果示例如下：
![](../imgs_results/e2e_res_img623_pgnet.jpg)

<a name="4"></a>
## 4. 模型训练、评估、预测
### 4.1 训练
### 4.2 评估
### 4.3 预测

<a name="5"></a>
## 5. 推理部署
### 5.1 Python推理
首先将DB文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在ICDAR2015英文数据集训练的模型为例（ [模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/det_r50_vd_db_v2.0_train.tar) )，可以使用如下命令进行转换：

```
python3 tools/export_model.py -c configs/det/det_r50_vd_db.yml -o Global.pretrained_model=./det_r50_vd_db_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/det_db
```

DB文本检测模型推理，可以执行如下命令：

```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_db/"
```

可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![](../imgs_results/det_res_img_10_db.jpg)

**注意**：由于ICDAR2015数据集只有1000张训练图像，且主要针对英文场景，所以上述模型对中文文本图像检测效果会比较差。

### 5.2 C++推理
敬请期待

### 5.3 Serving服务化部署
敬请期待

### 5.4 Paddle2ONNX推理
敬请期待

<a name="6"></a>
## 6. FAQ


## 引用

```bibtex
@inproceedings{liao2020real,
  title={Real-time scene text detection with differentiable binarization},
  author={Liao, Minghui and Wan, Zhaoyi and Yao, Cong and Chen, Kai and Bai, Xiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={11474--11481},
  year={2020}
}
```