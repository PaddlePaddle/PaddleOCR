# SRN

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
> [Towards Accurate Scene Text Recognition with Semantic Reasoning Networks](https://arxiv.org/abs/2003.12294#)
> Deli Yu, Xuan Li, Chengquan Zhang, Junyu Han, Jingtuo Liu, Errui Ding
> CVPR,2020

使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法复现效果如下：

|模型|骨干网络|配置文件|Acc|下载链接|
| --- | --- | --- | --- | --- |
|SRN|Resnet50_vd_fpn|[rec_r50_fpn_srn.yml](../../configs/rec/rec_r50_fpn_srn.yml)|86.31%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar)|


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
python3 tools/train.py -c configs/rec/rec_r50_fpn_srn.yml

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r50_fpn_srn.yml
```

评估

```
# GPU 评估， Global.pretrained_model 为待测权重
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r50_fpn_srn.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

预测：

```
# 预测使用的配置文件必须与训练一致
python3 tools/infer_rec.py -c configs/rec/rec_r50_fpn_srn.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理
首先将SRN文本识别训练过程中保存的模型，转换成inference model。（ [模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r50_vd_srn_train.tar) )，可以使用如下命令进行转换：

```
python3 tools/export_model.py -c configs/rec/rec_r50_fpn_srn.yml -o Global.pretrained_model=./rec_r50_vd_srn_train/best_accuracy  Global.save_inference_dir=./inference/rec_srn
```

SRN文本识别模型推理，可以执行如下命令：

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_srn/" --rec_image_shape="1,64,256"  --rec_algorithm="SRN" --rec_char_dict_path=./ppocr/utils/ic15_dict.txt  --use_space_char=False
```

<a name="4-2"></a>
### 4.2 C++推理

由于C++预处理后处理还未支持SRN，所以暂未支持

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
@article{Yu2020TowardsAS,
  title={Towards Accurate Scene Text Recognition With Semantic Reasoning Networks},
  author={Deli Yu and Xuan Li and Chengquan Zhang and Junyu Han and Jingtuo Liu and Errui Ding},
  journal={2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
  pages={12110-12119}
}
```
