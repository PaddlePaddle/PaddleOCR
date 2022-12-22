# SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition

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
> [SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition](https://arxiv.org/abs/2005.13117)
> Chengwei Zhang, Yunlu Xu, Zhanzhan Cheng, Shiliang Pu, Yi Niu, Fei Wu, Futai Zou
> AAAI, 2020

SPIN收录于AAAI2020。主要用于OCR识别任务。在任意形状文本识别中，矫正网络是一种较为常见的前置处理模块，但诸如RARE\ASTER\ESIR等只考虑了空间变换，并没有考虑色度变换。本文提出了一种结构Structure-Preserving Inner Offset Network (SPIN)，可以在色彩空间上进行变换。该模块是可微分的，可以加入到任意识别器中。
使用MJSynth和SynthText两个合成文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法复现效果如下：

|模型|骨干网络|配置文件|Acc|下载链接|
| --- | --- | --- | --- | --- |
|SPIN|ResNet32|[rec_r32_gaspin_bilstm_att.yml](../../configs/rec/rec_r32_gaspin_bilstm_att.yml)|90.00%|[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_r32_gaspin_bilstm_att.tar)|


<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

请参考[文本识别教程](./recognition.md)。PaddleOCR对代码进行了模块化，训练不同的识别模型只需要**更换配置文件**即可。

训练

具体地，在完成数据准备后，便可以启动训练，训练命令如下：

```
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml
```

评估

```
# GPU 评估， Global.pretrained_model 为待测权重
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

预测：

```
# 预测使用的配置文件必须与训练一致
python3 tools/infer_rec.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理
首先将SPIN文本识别训练过程中保存的模型，转换成inference model。可以使用如下命令进行转换：

```
python3 tools/export_model.py -c configs/rec/rec_r32_gaspin_bilstm_att.yml -o Global.pretrained_model={path/to/weights}/best_accuracy  Global.save_inference_dir=./inference/rec_r32_gaspin_bilstm_att
```
SPIN文本识别模型推理，可以执行如下命令：

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_r32_gaspin_bilstm_att/" --rec_image_shape="3, 32, 100" --rec_algorithm="SPIN" --rec_char_dict_path="/ppocr/utils/dict/spin_dict.txt" --use_space_char=Falsee
```

<a name="4-2"></a>
### 4.2 C++推理

由于C++预处理后处理还未支持SPIN，所以暂未支持

<a name="4-3"></a>
### 4.3 Serving服务化部署

暂不支持

<a name="4-4"></a>
### 4.4 更多推理部署

暂不支持

<a name="5"></a>
## 5. FAQ


## 引用

```bibtex
@article{2020SPIN,
  title={SPIN: Structure-Preserving Inner Offset Network for Scene Text Recognition},
  author={Chengwei Zhang and Yunlu Xu and Zhanzhan Cheng and Shiliang Pu and Yi Niu and Fei Wu and Futai Zou},
  journal={AAAI2020},
  year={2020},
}
```
