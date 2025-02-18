---
comments: true
---

# 场景文本识别算法-IGTR

## 1. 算法简介

论文信息：
> [Instruction-Guided Scene Text Recognition](https://arxiv.org/abs/2401.17851),
> Yongkun Du, Zhineng Chen, Yuchen Su, Caiyan Jia, Yu-Gang Jiang,
> TPAMI 2025, 
> 源仓库: [OpenOCR](https://github.com/Topdu/OpenOCR)

### IGTR算法简介

IGTR是由复旦大学[FVL实验室](https://fvl.fudan.edu.cn/) [OCR团队](https://github.com/Topdu/OpenOCR)提出的基于指令学习的场景文本识别（STR）方法。IGTR将STR视为一个跨模态指令学习问题，通过预测字符属性（如频率、位置等）来理解文本图像。具体而言，IGTR提出了以⟨条件，问题，答案⟩三元组格式的指令，提供丰富的字符属性描述；开发了轻量级指令编码器、跨模态特征融合模块和多任务答案头，增强文本图像理解能力。IGTR在英文和中文基准测试中均显著优于现有模型，同时保持较小的模型尺寸和快速推理速度。此外，通过调整指令采样规则，IGTR能够优雅地解决罕见字符和形态相似字符的识别问题。IGTR开创了基于指令学习的STR新范式，为多模态模型在特定任务中的应用提供了重要参考。

IGTR在场景文本识别公开数据集上的精度(%)和模型文件如下：
- 合成数据集（MJ+ST）训练，在Common Benchmarks测试, 训练集和测试集来自于 [PARSeq](https://github.com/baudm/parseq).

|  Model  | IC13<br/>857 | SVT  | IIIT5k<br/>3000 | IC15<br/>1811 | SVTP | CUTE80 |  Avg  |                                        Config&Model&Log                                         |
| :-----: | :----------: | :--: | :-------------: | :-----------: | :--: | :----: | :---: | :---------------------------------------------------------------------------------------------: |
| IGTR-PD |     97.6     | 95.2 |      97.6       |     88.4      | 91.6 |  95.5  | 94.30 | TODO |
| IGTR-AR |     98.6     | 95.7 |      98.2       |     88.4      | 92.4 |  95.5  | 94.78 |                                            as above                                             |

- 合成数据集（MJ+ST）训练，在Union14M-Benchmark测试, 测试集来自于 [Union14M](https://github.com/Mountchicken/Union14M/).

|  Model  | Curve | Multi-<br/>Oriented | Artistic | Contextless | Salient | Multi-<br/>word | General |  Avg  |    Config&Model&Log     |
| :-----: | :---: | :-----------------: | :------: | :---------: | :-----: | :-------------: | :-----: | :---: | :---------------------: |
| IGTR-PD | 76.9  |        30.6         |   59.1   |    63.3     |  77.8   |      62.5       |  66.7   | 62.40 | Same as the above table |
| IGTR-AR | 78.4  |        31.9         |   61.3   |    66.5     |  80.2   |      69.3       |  67.9   | 65.07 |        as above         |

- 在大规模真实数据集Union14M-L-LMDB-Filtered上训练的结果.

|    Model     | IC13<br/>857 | SVT  | IIIT5k<br/>3000 | IC15<br/>1811 | SVTP | CUTE80 |  Avg  |                                        Config&Model&Log                                         |
| :----------: | :----------: | :--: | :-------------: | :-----------: | :--: | :----: | :---: | :---------------------------------------------------------------------------------------------: |
|   IGTR-PD    |     97.7     | 97.7 |      98.3       |     89.8      | 93.7 |  97.9  | 95.86 | [PaddleOCR Model](https://paddleocr.bj.bcebos.com/igtr/rec_svtr_igtr_train.tar) |
|   IGTR-AR    |     98.1     | 98.4 |      98.7       |     90.5      | 94.9 |  98.3  | 96.48 |                                            as above                                             |
| IGTR-PD-60ep |     97.9     | 98.3 |      99.2       |     90.8      | 93.7 |  97.6  | 96.24 | TODO|
| IGTR-AR-60ep |     98.4     | 98.1 |      99.3       |     91.5      | 94.3 |  97.6  | 96.54 |                                            as above                                             |
|  IGTR-PD-PT  |     98.6     | 98.0 |      99.1       |     91.7      | 96.8 |  99.0  | 97.20 | TODO |
|  IGTR-AR-PT  |     98.8     | 98.3 |      99.2       |     92.0      | 96.8 |  99.0  | 97.34 |                                            as above                                             |

|    Model     | Curve | Multi-<br/>Oriented | Artistic | Contextless | Salient | Multi-<br/>word | General |  Avg  |    Config&Model&Log     |
| :----------: | :---: | :-----------------: | :------: | :---------: | :-----: | :-------------: | :-----: | :---: | :---------------------: |
|   IGTR-PD    | 88.1  |        89.9         |   74.2   |    80.3     |  82.8   |      79.2       |  83.0   | 82.51 | Same as the above table |
|   IGTR-AR    | 90.4  |        91.2         |   77.0   |    82.4     |  84.7   |      84.0       |  84.4   | 84.86 |        as above         |
| IGTR-PD-60ep | 90.0  |        92.1         |   77.5   |    82.8     |  86.0   |      83.0       |  84.8   | 85.18 | Same as the above table |
| IGTR-AR-60ep | 91.0  |        93.0         |   78.7   |    84.6     |  87.3   |      84.8       |  85.6   | 86.43 |        as above         |
|  IGTR-PD-PT  | 92.4  |        92.1         |   80.7   |    83.6     |  87.7   |      86.9       |  85.0   | 86.92 | Same as the above table |
|  IGTR-AR-PT  | 93.0  |        92.9         |   81.3   |    83.4     |  88.6   |      88.7       |  85.6   | 87.65 |        as above         |

- 中文文本识别的结果, 训练集和测试集来自于 [Chinese Benckmark](https://github.com/FudanVI/benchmarking-chinese-text-recognition).

|    Model    | Scene | Web  | Document | Handwriting |  Avg  |                                        Config&Model&Log                                         |
| :---------: | :---: | :--: | :------: | :---------: | :---: | :---------------------------------------------------------------------------------------------: |
|   IGTR-PD   | 73.1  | 74.8 |   98.6   |    52.5     | 74.75 |                                                                                                 |
|   IGTR-AR   | 75.1  | 76.4 |   98.7   |    55.3     | 76.37 |                                                                                                 |
| IGTR-PD-TS  | 73.5  | 75.9 |   98.7   |    54.5     | 75.65 | TODO |
| IGTR-AR-TS  | 75.6  | 77.0 |   98.8   |    57.3     | 77.17 |                                            as above                                             |
| IGTR-PD-Aug | 79.5  | 80.0 |   99.4   |    58.9     | 79.45 | TODO |
| IGTR-AR-Aug | 82.0  | 81.7 |   99.5   |    63.8     | 81.74 |                                            as above                                             |

从[OpenOCR](https://github.com/Topdu/OpenOCR/blob/main/configs/rec/igtr/readme.md)可以下载所有的模型文本和训练日志, 将模型文件转换为符合paddleocr 模型参数的要求后，即可在PaddleOCR中使用.

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

### 3.1 模型训练

#### 数据集准备

[英文数据集下载](https://github.com/baudm/parseq)

[Union14M-L-LMDB-Filtered](https://github.com/Mountchicken/Union14M)

[中文数据集下载](https://github.com/fudanvi/benchmarking-chinese-text-recognition#download)

#### 启动训练

请参考[文本识别训练教程](../../ppocr/model_train/recognition.md)。在完成数据准备后，便可以启动训练，训练命令如下：

```bash linenums="1"
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_svtrnet_igtr.yml

# 多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_svtrnet_igtr.yml
```

### 3.2 评估

可下载`IGTR`提供的模型文件和配置文件：[下载地址](https://paddleocr.bj.bcebos.com/igtr/rec_svtr_igtr_train.tar) ，使用如下命令进行评估：

```bash linenums="1"
# 下载包含IGTR的模型文件和配置文件的tar压缩包并解压
wget https://paddleocr.bj.bcebos.com/igtr/rec_svtr_igtr_train.tar && tar xf rec_svtr_igtr_train.tar
# 注意将pretrained_model的路径设置为本地路径。
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_svtrnet_igtr.yml -o Global.pretrained_model=./rec_svtr_igtr_train/best_model
```

### 3.3 预测

使用如下命令进行单张图片预测：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c configs/rec/rec_svtrnet_igtr.yml -o Global.infer_img='./doc/imgs_words/word_10.png' Global.pretrained_model=./rec_svtr_igtr_train/best_model
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/imgs_words_en/'。
```

## 4. 推理部署

### 4.1 Python推理

即将实现

### 4.2 C++推理部署

暂不支持

### 4.3 Serving服务化部署

暂不支持

### 4.4 更多推理部署

暂不支持

## 引用

```bibtex
@article{Du2025IGTR,
  title     = {Instruction-Guided Scene Text Recognition},
  author    = {Du, Yongkun and Chen, Zhineng and Su, Yuchen and Jia, Caiyan and Jiang, Yu-Gang},
  journal   = {IEEE Trans. Pattern Anal. Mach. Intell.},
  year      = {2025},
  url       = {https://arxiv.org/abs/2401.17851}
}
```
