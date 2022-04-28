# 场景文本识别算法-NRTR

- [1. 算法简介](#1)
- [2. 环境配置](#2)
- [3. 模型训练、评估、预测](#3)
    - [3.1 训练](#3-1)
    - [3.2 评估](#3-2)
    - [3.3 预测](#3-3)
- [4. 推理部署](#4)
    - [4.1 Python推理](#4-1)
- [5. FAQ](#5)

<a name="1"></a>
## 1. 算法简介

论文信息：
> [NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition](https://arxiv.org/abs/1806.00926)
> Fenfen Sheng and Zhineng Chen and Bo Xu
> ICDAR, 2019


<a name="model"></a>
`NRTR`在场景文本识别公开数据集上的精度和模型文件如下：

|     | Avg accruacy |                                      下载链接                                      | 配置文件 |
|-----| --- | --- | --- |
| NRTR |       84.21%    | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) |   [config](../../configs/rec/rec_mtb_nrtr.yml)     |


<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

<a name="3-1"></a>
### 3.1 模型训练

#### 数据集准备

数据集采用[DTRB](https://arxiv.org/abs/1904.01906) 文字识别训练和评估流程，使用`MJSynth`和`SynthText`两个识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估。

#### 启动训练

请参考[文本识别训练教程](./recognition.md)。PaddleOCR对代码进行了模块化，训练`NRTR`识别模型时需要**更换配置文件**为`NRTR`的[配置文件](../../configs/rec/rec_mtb_nrtr.yml)。

<a name="3-2"></a>
### 3.2 评估

可下载已训练完成的[模型文件](#model)，使用如下命令进行评估：

```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/eval.py -c configs/rec/rec_mtb_nrtr.yml -o Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy
```

<a name="3-3"></a>
### 3.3 预测

使用如下命令进行单张图片预测：
```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c configs/rec/rec_mtb_nrtr.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy Global.load_static_weights=false
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/imgs_words_en/'。
```


<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理
首先将训练得到best模型，转换成inference model。这里以训练完成的模型为例（[模型下载地址](#model))，可以使用如下命令进行转换：

```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c configs/rec/rec_mtb_nrtr.yml -o Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy Global.save_inference_dir=./inference/rec_mtb_nrtr/ Global.load_static_weights=False
```

执行如下命令进行模型推理：

```shell
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_mtb_nrtr/' --rec_algorithm='NRTR' --rec_image_shape='1,32,100' --rec_char_dict_path='./ppocr/utils/EN_symbol_dict.txt'
# 预测文件夹下所有图像时，可修改image_dir为文件夹，如 --image_dir='./doc/imgs_words_en/'。
```
<a name="5"></a>
## 5. FAQ

1. `NRTR`论文中使用Beam搜索进行解码字符，但是速度较慢，这里默认未使用Beam搜索，以贪婪搜索进行解码字符。
