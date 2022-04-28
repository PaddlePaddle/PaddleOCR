# SEED

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
> [STAR-Net: a spatial attention residue network for scene text recognition.](https://arxiv.org/pdf/2005.10977.pdf)

> Qiao, Zhi and Zhou, Yu and Yang, Dongbao and Zhou, Yucan and Wang, Weiping

> CVPR, 2020

参考[DTRB](https://arxiv.org/abs/1904.01906) 文字识别训练和评估流程，使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法效果如下：

|模型|骨干网络|Avg Accuracy|模型存储命名|下载链接|
|---|---|---|---|---|
|SEED|Aster_Resnet| 85.2% | rec_resnet_stn_bilstm_att | [训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/rec/rec_resnet_stn_bilstm_att.tar) |

<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

请参考[文本识别训练教程](./recognition.md)。PaddleOCR对代码进行了模块化，训练不同的识别模型只需要**更换配置文件**即可。


<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理

comming soon

<a name="4-2"></a>
### 4.2 C++推理

comming soon

<a name="4-3"></a>
### 4.3 Serving服务化部署

comming soon

<a name="4-4"></a>
### 4.4 更多推理部署

comming soon

<a name="5"></a>
## 5. FAQ


## 引用

```bibtex
@inproceedings{qiao2020seed,
  title={Seed: Semantics enhanced encoder-decoder framework for scene text recognition},
  author={Qiao, Zhi and Zhou, Yu and Yang, Dongbao and Zhou, Yucan and Wang, Weiping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13528--13537},
  year={2020}
}
```
