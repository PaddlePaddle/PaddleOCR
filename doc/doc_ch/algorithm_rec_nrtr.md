# 场景文本识别算法-NRTR

- [1. 算法简介](#1)
- [2. 环境配置](#2)
- [3. 模型训练、评估、预测](#3)
    - [3.1 训练](#3-1)
    - [3.2 评估](#3-2)
    - [3.3 预测](#3-3)
- [4. 推理部署](#4)
    - [4.1 Python推理](#4-1)
    - [4.2 C++推理](#4-2)
    - [4.3 Serving服务化部署](#4-3)
    - [4.4 更多推理部署](#4-4)
- [5. FAQ](#5)

<a name="1"></a>
## 1. 算法简介

论文信息：
> [NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition](https://arxiv.org/abs/1806.00926)
> Fenfen Sheng and Zhineng Chen and Bo Xu
> ICDAR, 2019


<a name="model"></a>
`NRTR`使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法复现效果如下：

|模型|骨干网络|配置文件|Acc|下载链接|
| --- | --- | --- | --- | --- |
|NRTR|MTB|[rec_mtb_nrtr.yml](../../configs/rec/rec_mtb_nrtr.yml)|84.21%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar)|

<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

<a name="3-1"></a>
### 3.1 模型训练

请参考[文本识别训练教程](./recognition.md)。PaddleOCR对代码进行了模块化，训练`NRTR`识别模型时需要**更换配置文件**为`NRTR`的[配置文件](../../configs/rec/rec_mtb_nrtr.yml)。

#### 启动训练


具体地，在完成数据准备后，便可以启动训练，训练命令如下：
```shell
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_mtb_nrtr.yml

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_mtb_nrtr.yml
```

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
python3 tools/infer_rec.py -c configs/rec/rec_mtb_nrtr.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/imgs_words_en/'。
```


<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理
首先将训练得到best模型，转换成inference model。这里以训练完成的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mtb_nrtr_train.tar) )，可以使用如下命令进行转换：

```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c configs/rec/rec_mtb_nrtr.yml -o Global.pretrained_model=./rec_mtb_nrtr_train/best_accuracy Global.save_inference_dir=./inference/rec_mtb_nrtr/
```
**注意：**
- 如果您是在自己的数据集上训练的模型，并且调整了字典文件，请注意修改配置文件中的`character_dict_path`是否是所需要的字典文件。
- 如果您修改了训练时的输入大小，请修改`tools/export_model.py`文件中的对应NRTR的`infer_shape`。

转换成功后，在目录下有三个文件：
```
/inference/rec_mtb_nrtr/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```

执行如下命令进行模型推理：

```shell
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_mtb_nrtr/' --rec_algorithm='NRTR' --rec_image_shape='1,32,100' --rec_char_dict_path='./ppocr/utils/EN_symbol_dict.txt'
# 预测文件夹下所有图像时，可修改image_dir为文件夹，如 --image_dir='./doc/imgs_words_en/'。
```

![](../imgs_words_en/word_10.png)

执行命令后，上面图像的预测结果（识别的文本和得分）会打印到屏幕上，示例如下：
结果如下：
```shell
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9265879392623901)
```

**注意**：

- 训练上述模型采用的图像分辨率是[1，32，100]，需要通过参数`rec_image_shape`设置为您训练时的识别图像形状。
- 在推理时需要设置参数`rec_char_dict_path`指定字典，如果您修改了字典，请修改该参数为您的字典文件。
- 如果您修改了预处理方法，需修改`tools/infer/predict_rec.py`中NRTR的预处理为您的预处理方法。


<a name="4-2"></a>
### 4.2 C++推理部署

由于C++预处理后处理还未支持NRTR，所以暂未支持

<a name="4-3"></a>
### 4.3 Serving服务化部署

暂不支持

<a name="4-4"></a>
### 4.4 更多推理部署

暂不支持

<a name="5"></a>
## 5. FAQ

1. `NRTR`论文中使用Beam搜索进行解码字符，但是速度较慢，这里默认未使用Beam搜索，以贪婪搜索进行解码字符。

## 引用

```bibtex
@article{Sheng2019NRTR,
  author    = {Fenfen Sheng and Zhineng Chen andBo Xu},
  title     = {NRTR: A No-Recurrence Sequence-to-Sequence Model For Scene Text Recognition},
  journal   = {ICDAR},
  year      = {2019},
  url       = {http://arxiv.org/abs/1806.00926},
  pages     = {781-786}
}
```
