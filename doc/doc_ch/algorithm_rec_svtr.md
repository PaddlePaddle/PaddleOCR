# 场景文本识别算法-SVTR

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
> [SVTR: Scene Text Recognition with a Single Visual Model]()
> Yongkun Du and Zhineng Chen and Caiyan Jia Xiaoting Yin and Tianlun Zheng and Chenxia Li and Yuning Du and Yu-Gang Jiang
> IJCAI, 2022

场景文本识别旨在将自然图像中的文本转录为数字字符序列，从而传达对场景理解至关重要的高级语义。这项任务由于文本变形、字体、遮挡、杂乱背景等方面的变化具有一定的挑战性。先前的方法为提高识别精度做出了许多工作。然而文本识别器除了准确度外，还因为实际需求需要考虑推理速度等因素。

### SVTR算法简介

主流的场景文本识别模型通常包含两个模块：用于特征提取的视觉模型和用于文本转录的序列模型。这种架构虽然准确，但复杂且效率较低，限制了在实际场景中的应用。SVTR提出了一种用于场景文本识别的单视觉模型，该模型在patch-wise image tokenization框架内，完全摒弃了序列建模，在精度具有竞争力的前提下，模型参数量更少，速度更快，主要有以下几点贡献：
1. 首次发现单视觉模型可以达到与视觉语言模型相媲美甚至更高的准确率，并且其具有效率高和适应多语言的优点，在实际应用中很有前景。
2. SVTR从字符组件的角度出发，逐渐的合并字符组件，自下而上地完成字符的识别。
3. SVTR引入了局部和全局Mixing，分别用于提取字符组件特征和字符间依赖关系，与多尺度的特征一起，形成多粒度特征描述。


<a name="model"></a>
SVTR在场景文本识别公开数据集上的精度(%)和模型文件如下：

* 中文数据集来自于[Chinese Benckmark](https://arxiv.org/abs/2112.15093) ，SVTR的中文训练评估策略遵循该论文。

|    模型      |IC13<br/>857 |  SVT  |IIIT5k<br/>3000 |IC15<br/>1811| SVTP  |CUTE80 | Avg_6 |IC15<br/>2077 |IC13<br/>1015 |IC03<br/>867|IC03<br/>860|Avg_10 | Chinese<br/>scene_test|      下载链接       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|:-----:|:-----:|:---------------------------------------------:|:-----:|:-----:|
| SVTR Tiny  | 96.85  | 91.34 |   94.53   | 83.99  | 85.43 | 89.24 | 90.87 |  80.55  |  95.37  | 95.27 | 95.70 | 90.13 | 67.90 | [英文](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar)  / [中文](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_ch_train.tar)  |
| SVTR Small | 95.92  | 93.04 |   95.03   | 84.70  | 87.91 | 92.01 | 91.63 |  82.72  |  94.88  | 96.08 | 96.28 | 91.02 | 69.00 | [英文](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_small_none_ctc_en_train.tar) / [中文](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_small_none_ctc_ch_train.tar) |
| SVTR Base  | 97.08  | 91.50 |   96.03   | 85.20  | 89.92 | 91.67 | 92.33 |  83.73  |  95.66  | 95.62 | 95.81 | 91.61 | 71.40 | [英文](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_base_none_ctc_en_train.tar)  /                                              -                                             |
| SVTR Large | 97.20  | 91.65 |   96.30   | 86.58  | 88.37 | 95.14 | 92.82 |  84.54  |  96.35  | 96.54 | 96.74 | 92.24 | 72.10 | [英文](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_large_none_ctc_en_train.tar) / [中文](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_large_none_ctc_ch_train.tar) |



<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

<a name="3-1"></a>
### 3.1 模型训练

#### 数据集准备

[英文数据集下载](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)
[中文数据集下载](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

#### 启动训练

请参考[文本识别训练教程](./recognition.md)。PaddleOCR对代码进行了模块化，训练`SVTR`识别模型时需要**更换配置文件**为`SVTR`的[配置文件](../../configs/rec/rec_svtrnet.yml)。

具体地，在完成数据准备后，便可以启动训练，训练命令如下：
```shell
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_svtrnet.yml

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_svtrnet.yml
```

<a name="3-2"></a>
### 3.2 评估

可下载`SVTR`提供的模型文件和配置文件：[下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar) ，以`SVTR-T`为例，使用如下命令进行评估：

```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/eval.py -c ./rec_svtr_tiny_en_train/rec_svtr_tiny_6local_6global_stn_en.yml -o Global.pretrained_model=./rec_svtr_tiny_none_ctc_en_train/best_accuracy
```

<a name="3-3"></a>
### 3.3 预测

使用如下命令进行单张图片预测：
```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c ./rec_svtr_tiny_en_train/rec_svtr_tiny_6local_6global_stn_en.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_svtr_tiny_none_ctc_en_train/best_accuracy
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/imgs_words_en/'。
```


<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理
首先将训练得到best模型，转换成inference model。下面以基于`SVTR-T`，在英文数据集训练的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar) )，可以使用如下命令进行转换：

```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c ./rec_svtr_tiny_en_train/rec_svtr_tiny_6local_6global_stn_en.yml -o Global.pretrained_model=./rec_svtr_tiny_none_ctc_en_train/best_accuracy Global.save_inference_dir=./inference/rec_svtr_tiny_stn_en
```

**注意：**
- 如果您是在自己的数据集上训练的模型，并且调整了字典文件，请注意修改配置文件中的`character_dict_path`是否为所正确的字典文件。
- 如果您修改了训练时的输入大小，请修改`tools/export_model.py`文件中的对应SVTR的`infer_shape`。

转换成功后，在目录下有三个文件：
```
/inference/rec_svtr_tiny_stn_en/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```


执行如下命令进行模型推理：

```shell
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_svtr_tiny_stn_en/' --rec_algorithm='SVTR' --rec_image_shape='3,64,256' --rec_char_dict_path='./ppocr/utils/ic15_dict.txt'
# 预测文件夹下所有图像时，可修改image_dir为文件夹，如 --image_dir='./doc/imgs_words_en/'。
```
![](../imgs_words_en/word_10.png)

执行命令后，上面图像的预测结果（识别的文本和得分）会打印到屏幕上，示例如下：
结果如下：
```shell
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9999998807907104)
```

**注意**：

- 如果您调整了训练时的输入分辨率，需要通过参数`rec_image_shape`设置为您需要的识别图像形状。
- 在推理时需要设置参数`rec_char_dict_path`指定字典，如果您修改了字典，请修改该参数为您的字典文件。
- 如果您修改了预处理方法，需修改`tools/infer/predict_rec.py`中SVTR的预处理为您的预处理方法。

<a name="4-2"></a>
### 4.2 C++推理部署

由于C++预处理后处理还未支持SVTR，所以暂未支持

<a name="4-3"></a>
### 4.3 Serving服务化部署

暂不支持

<a name="4-4"></a>
### 4.4 更多推理部署

暂不支持

<a name="5"></a>
## 5. FAQ

1. 由于`SVTR`使用的op算子大多为矩阵相乘，在GPU环境下，速度具有优势，但在CPU开启mkldnn加速环境下，`SVTR`相比于被优化的卷积网络没有优势。
