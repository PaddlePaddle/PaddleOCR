---
comments: true
---

# ParseQ

## 1. 算法简介

论文信息：
> [Scene Text Recognition with Permuted Autoregressive Sequence Models](https://arxiv.org/abs/2207.06966)
> Darwin Bautista, Rowel Atienza
> ECCV, 2021

原论文分别使用真实文本识别数据集(Real)和合成文本识别数据集(Synth)进行训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估。其中：

- 真实文本识别数据集(Real)包含COCO-Text, RCTW17, Uber-Text, ArT, LSVT, MLT19, ReCTS, TextOCR, OpenVINO数据集
- 合成文本识别数据集(Synth)包含MJSynth和SynthText数据集

在不同数据集上训练的算法的复现效果如下：

|数据集|模型|骨干网络|配置文件|Acc|下载链接|
| --- | --- | --- | --- | --- | --- |
|Synth|ParseQ|VIT|[rec_vit_parseq.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_vit_parseq.yml)|91.24%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/parseq/rec_vit_parseq_synth.tgz)|
|Real|ParseQ|VIT|[rec_vit_parseq.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_vit_parseq.yml)|94.74%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.1/parseq/rec_vit_parseq_real.tgz)|

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

请参考[文本识别教程](../../ppocr/model_train/recognition.md)。PaddleOCR对代码进行了模块化，训练不同的识别模型只需要**更换配置文件**即可。

### 训练

具体地，在完成数据准备后，便可以启动训练，训练命令如下：

```bash linenums="1"
# 单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_vit_parseq.yml

# 多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_vit_parseq.yml
```

### 评估

```bash linenums="1"
# GPU 评估， Global.pretrained_model 为待测权重
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_vit_parseq.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### 预测

```bash linenums="1"
# 预测使用的配置文件必须与训练一致
python3 tools/infer_rec.py -c configs/rec/rec_vit_parseq.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

## 4. 推理部署

### 4.1 Python推理

首先将ParseQ文本识别训练过程中保存的模型，转换成inference model。（ [模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.1/parseq/rec_vit_parseq_real.tgz) )，可以使用如下命令进行转换：

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_vit_parseq.yml -o Global.pretrained_model=./rec_vit_parseq_real/best_accuracy Global.save_inference_dir=./inference/rec_parseq
```

ParseQ文本识别模型推理，可以执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_parseq/" --rec_image_shape="3, 32, 128" --rec_algorithm="ParseQ" --rec_char_dict_path="ppocr/utils/dict/parseq_dict.txt" --max_text_length=25 --use_space_char=False
```

### 4.2 C++推理

由于C++预处理后处理还未支持ParseQ，所以暂未支持

### 4.3 Serving服务化部署

暂不支持

### 4.4 更多推理部署

暂不支持

## 5. FAQ

## 引用

```bibtex
@InProceedings{bautista2022parseq,
  title={Scene Text Recognition with Permuted Autoregressive Sequence Models},
  author={Bautista, Darwin and Atienza, Rowel},
  booktitle={European Conference on Computer Vision},
  pages={178--196},
  month={10},
  year={2022},
  publisher={Springer Nature Switzerland},
  address={Cham},
  doi={10.1007/978-3-031-19815-1_11},
  url={https://doi.org/10.1007/978-3-031-19815-1_11}
}
```
