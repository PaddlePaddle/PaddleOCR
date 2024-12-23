# 印刷数学公式识别算法-PP-FormulaNet

## 1. 算法简介

`PP-FormulaNet` 是百度飞桨自研的公式识别模型，采用 PaddleX 内部自建的 5百万数据集进行训练，在对应测试集上的精度如下：



| 模型        | 骨干网络       | 配置文件                                                  | SPE-<br/>BLEU↑ | CPE-<br/>BLEU↑  | Easy-<br/>BLEU↑ | Middle-<br/>BLEU↑ | Hard-<br/>BLEU↑| Avg-<br/>BLEU↑ | 下载链接 |
|-----------|------------|------------------|:--------------:|:---------:|:----------:|:----------------:|:---------:|:-----------------:|:-----------------:|
| UniMERNet | Donut Swin | [rec_unimernet.yml](../../../configs/rec/rec_unimernet.yml) |     0.9187  |    0.9252       | 0.8658  |    0.8228   | 0.7740 |     0.8613        |[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_unimernet_train.tar)|
| PP-FormulaNet-S | PPHGNetV2_B4 | [rec_pp_formulanet_s.yml](../../../configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml) |    0.8694   |    0.8071       | 0.9294  |    0.9112    | 0.8391 |    0.8712       |[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_s_train.tar)|
| PP-FormulaNet-L | Vary_VIT_B | [rec_pp_formulanet_l.yml](../../../configs/rec/PP-FormuaNet/rec_pp_formulanet_l.yml) |     0.9055   |     0.9206       | 0.9392  |     0.9273    | 0.9141 |     0.9213         |[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_l_train.tar )|

其中，SPE、CPE为UniMERNet的简单公式数据集和复杂公式数据集；Easy、Middle、Hard为PaddleX内部自建的简单公式数据集（LaTeX 代码长度 0-64）、中等公式数据集（LaTeX 代码长度  64-256）和复杂公式数据集（LaTeX 代码长度  256+）。

## 2. 环境配置
请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

此外，需要安装额外的依赖：
```shell
sudo apt-get update
sudo apt-get install libmagickwand-dev
pip install -r docs/algorithm/formula_recognition/requirements.txt
```

## 3. 模型训练、评估、预测

### 3.1 准备数据集

```shell
# 下载 PaddleX 官方示例数据集
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_latexocr_dataset_example.tar
tar -xf ocr_rec_latexocr_dataset_example.tar
```


### 3.2 下载预训练模型

```shell
# 下载 PP-FormulaNet-S 预训练模型
wget https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_s_train.tar 
tar -xf rec_ppformulanet_s_train.tar
```


### 3.3 模型训练

请参考[文本识别训练教程](../../ppocr/model_train/recognition.md)。PaddleOCR对代码进行了模块化，训练 `PP-FormulaNet-S` 识别模型时需要**更换配置文件**为 `PP-FormulaNet-S` 的[配置文件](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml)。

#### 启动训练

具体地，在完成数据准备后，便可以启动训练，训练命令如下：
```shell
#单卡训练 (默认训练方式)
python3 tools/train.py -c configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml \
   -o Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3' --ips=127.0.0.1   tools/train.py -c configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml \
        -o Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
```

**注意：**

- 默认每训练 1个epoch（179 次iteration）进行1次评估，若您更改训练的batch_size，或更换数据集，请在训练时作出如下修改
```
python3  -m paddle.distributed.launch --gpus '0,1,2,3' --ips=127.0.0.1   tools/train.py -c configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml \
  -o Global.eval_batch_step=[0,{length_of_dataset//batch_size//4}] \
   Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
```

### 3.4 评估

可下载已训练完成的[模型文件](https://paddleocr.bj.bcebos.com/contribution/rec_ppformulanet_s_train.tar )，使用如下命令进行评估：

```shell
# 注意将pretrained_model的路径设置为本地路径。若使用自行训练保存的模型，请注意修改路径和文件名为{path/to/weights}/{model_name}。
 # demo 测试集评估
 python3 tools/eval.py -c configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml -o \
 Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
```

### 3.5 预测

使用如下命令进行单张图片预测：
```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c configs/rec/PP-FormuaNet/rec_pp_formulanet_s.yml \
  -o  Global.infer_img='./docs/datasets/images/pme_demo/0000099.png'\
   Global.pretrained_model=./rec_ppformulanet_s_train/best_accuracy.pdparams
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/datasets/pme_demo/'。
```

## 4. FAQ
