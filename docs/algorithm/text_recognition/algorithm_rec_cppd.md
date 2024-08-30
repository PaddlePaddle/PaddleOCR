---
comments: true
---

# 场景文本识别算法-CPPD

## 1. 算法简介

论文信息：
> [Context Perception Parallel Decoder for Scene Text Recognition](https://arxiv.org/abs/2307.12270)
> Yongkun Du and Zhineng Chen and Caiyan Jia and Xiaoting Yin and Chenxia Li and Yuning Du and Yu-Gang Jiang

### CPPD算法简介

基于深度学习的场景文本识别模型通常是Encoder-Decoder结构，其中decoder可以分为两种：(1)CTC，(2)Attention-based。目前SOTA模型大多使用Attention-based的decoder，而attention-based可以分为AR和PD两种，一般来说，AR解码器识别精度优于PD，而PD解码速度快于AR，CPPD通过精心设计的CO和CC模块，达到了“AR的精度，PD的速度”的效果。

CPPD在场景文本识别公开数据集上的精度(%)和模型文件如下：

* 英文训练集和测试集来自于[PARSeq](https://github.com/baudm/parseq)。

|    模型      |IC13<br/>857 |  SVT  |IIIT5k<br/>3000 |IC15<br/>1811| SVTP  |CUTE80 | Avg |      下载链接       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|
| CPPD Tiny  | 97.1  | 94.4 |   96.6   | 86.6  | 88.5 | 90.3 | 92.25 | [英文](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_tiny_en_train.tar) |
| CPPD Base | 98.2  | 95.5 |   97.6   | 87.9  | 90.0 | 92.7 | 93.80 | [英文](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar)|
| CPPD Base 48*160  | 97.5  | 95.5 |   97.7   | 87.7  | 92.4 | 93.7 | 94.10 | [英文](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_48_160_en_train.tar) |

* 英文合成数据集(MJ+ST)训练，英文Union14M-L benchmark测试结果[U14m](https://github.com/Mountchicken/Union14M/)。

|    模型      |Curve |  Multi-<br/>Oriented  |Artistic |Contextless| Salient  | Multi-<br/>word | General | Avg |     下载链接       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|
| CPPD Tiny  | 52.4  | 12.3 |   48.2   | 54.4  | 61.5 | 53.4 | 61.4 | 49.10 | 同上表 |
| CPPD Base | 65.5  | 18.6 |   56.0   | 61.9  | 71.0 | 57.5 | 65.8 | 56.63 | 同上表 |
| CPPD Base 48*160  | 71.9  | 22.1 |   60.5   | 67.9  | 78.3 | 63.9 | 67.1 | 61.69 | 同上表 |

* Union14M-L 训练集From scratch训练，英文测试结果。

|    模型      |IC13<br/>857 |  SVT  |IIIT5k<br/>3000 |IC15<br/>1811| SVTP  |CUTE80 | Avg |      下载链接       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|
| CPPD Base 32*128  | 98.5  | 97.7 |   99.2   | 90.3  | 94.6 | 98.3 | 96.42 | Coming soon |

|    模型      |Curve |  Multi-<br/>Oriented  |Artistic |Contextless| Salient  | Multi-<br/>word | General | Avg |     下载链接       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|
| CPPD Base 32*128  | 83.0  | 71.2 |   75.1   | 80.9  | 79.4 | 82.6 | 83.7 | 79.41 | Coming soon |

* 加载合成数据集预训练模型，Union14M-L 训练集微调训练，英文测试结果。

|    模型      |IC13<br/>857 |  SVT  |IIIT5k<br/>3000 |IC15<br/>1811| SVTP  |CUTE80 | Avg |      下载链接       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|
| CPPD Base 32*128  | 98.7  | 98.5 |   99.4   | 91.7  | 96.7 | 99.7 | 97.44 | [英文](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_u14m_train.tar) |

|    模型      |Curve |  Multi-<br/>Oriented  |Artistic |Contextless| Salient  | Multi-<br/>word | General | Avg |     下载链接       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|:-----:|:-------:|:-------:|
| CPPD Base 32*128  | 87.5  | 70.7 |   78.2   | 82.9  | 85.5 | 85.4 | 84.3 | 82.08 | 同上表 |

* 中文训练集和测试集来自于[Chinese Benckmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition)。

|    模型      | Scene | Web | Document | Handwriting | Avg |      下载链接       |
|:----------:|:------:|:-----:|:---------:|:------:|:-----:|:-----:|
| CPPD Base  | 74.4  | 76.1 |   98.6   | 55.3  | 76.10 | [中文](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_ch_train.tar)  |
| CPPD Base + STN | 78.4  | 79.3 |   98.9   | 57.6  | 78.55 | [中文](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_stn_ch_train.tar) |

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

### 3.1 模型训练

#### 数据集准备

[英文数据集下载](https://github.com/baudm/parseq)

[Union14M-L 下载](https://github.com/Mountchicken/Union14M)

[中文数据集下载](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

#### 启动训练

请参考[文本识别训练教程](../../ppocr/model_train/recognition.md)。PaddleOCR对代码进行了模块化，训练`CPPD`识别模型时需要**更换配置文件**为`CPPD`的[配置文件](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_svtrnet_cppd_base_en.yml)。

具体地，在完成数据准备后，便可以启动训练，训练命令如下：

```bash linenums="1"
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_svtrnet_cppd_base_en.yml

# 多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_svtrnet_cppd_base_en.yml
```

### 3.2 评估

可下载`CPPD`提供的模型文件和配置文件：[下载地址](https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar) ，以`CPPD-B`为例，使用如下命令进行评估：

```bash linenums="1"
# 下载包含CPPD-B的模型文件和配置文件的tar压缩包并解压
wget https://paddleocr.bj.bcebos.com/CCPD/rec_svtr_cppd_base_en_train.tar && tar xf rec_svtr_cppd_base_en_train.tar
# 注意将pretrained_model的路径设置为本地路径。
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c ./rec_svtr_cppd_base_en_train/rec_svtrnet_cppd_base_en.yml -o Global.pretrained_model=./rec_svtr_cppd_base_en_train/best_model
```

### 3.3 预测

使用如下命令进行单张图片预测：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c ./rec_svtr_cppd_base_en_train/rec_svtrnet_cppd_base_en.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_svtr_cppd_base_en_train/best_model
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/imgs_words_en/'。
```

## 4. 推理部署

### 4.1 Python推理

首先将训练得到best模型，转换成inference model。下面以基于`CPPD-B`，在英文数据集训练的模型为例（[模型和配置文件下载地址](https://paddleocr.bj.bcebos.com/CPPD/rec_svtr_cppd_base_en_train.tar)，可以使用如下命令进行转换：

**注意：**

* 如果您是在自己的数据集上训练的模型，并且调整了字典文件，请注意修改配置文件中的`character_dict_path`是否为所正确的字典文件。

执行如下命令进行模型导出和推理：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
# export model
# en
python3 tools/export_model.py -c configs/rec/rec_svtrnet_cppd_base_en.yml -o Global.pretrained_model=./rec_svtr_cppd_base_en_train/best_model.pdparams Global.save_inference_dir=./rec_svtr_cppd_base_en_infer
# ch
python3 tools/export_model.py -c configs/rec/rec_svtrnet_cppd_base_ch.yml -o Global.pretrained_model=./rec_svtr_cppd_base_ch_train/best_model.pdparams Global.save_inference_dir=./rec_svtr_cppd_base_ch_infer

# speed test
# docker image https://hub.docker.com/r/paddlepaddle/paddle/tags/: sudo docker pull paddlepaddle/paddle:2.4.2-gpu-cuda11.2-cudnn8.2-trt8.0
# install auto_log: pip install https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
# en
python3 tools/infer/predict_rec.py --image_dir='../iiik' --rec_model_dir='./rec_svtr_cppd_base_en_infer/' --rec_algorithm='CPPD' --rec_image_shape='3,32,100' --rec_char_dict_path='./ppocr/utils/ic15_dict.txt' --warmup=True --benchmark=True --rec_batch_num=1 --use_tensorrt=True
# ch
python3 tools/infer/predict_rec.py --image_dir='../iiik' --rec_model_dir='./rec_svtr_cppd_base_ch_infer/' --rec_algorithm='CPPDPadding' --rec_image_shape='3,32,256' --warmup=True --benchmark=True --rec_batch_num=1 --use_tensorrt=True
# stn_ch
python3 tools/infer/predict_rec.py --image_dir='../iiik' --rec_model_dir='./rec_svtr_cppd_base_stn_ch_infer/' --rec_algorithm='CPPD' --rec_image_shape='3,64,256' --warmup=True --benchmark=True --rec_batch_num=1 --use_tensorrt=True
```

导出成功后，在目录下有三个文件：

```
/inference/rec_svtr_cppd_base_en_infer/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```

### 4.2 C++推理部署

由于C++预处理后处理还未支持CPPD，所以暂未支持

### 4.3 Serving服务化部署

暂不支持

### 4.4 更多推理部署

暂不支持

## 引用

```bibtex
@article{Du2023CPPD,
  title     = {Context Perception Parallel Decoder for Scene Text Recognition},
  author    = {Du, Yongkun and Chen, Zhineng and Jia, Caiyan and Yin, Xiaoting and Li, Chenxia and Du, Yuning and Jiang, Yu-Gang},
  booktitle = {Arxiv},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.12270}
}
```
