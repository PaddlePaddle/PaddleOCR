---
typora-copy-images-to: images
comments: true
---

# Rosetta

## 1. 算法简介

论文信息：
> [Rosetta: Large Scale System for Text Detection and Recognition in Images](https://arxiv.org/abs/1910.05085)
> Borisyuk F ,  Gordo A ,  V  Sivakumar
> KDD, 2018

使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估， 算法复现效果如下：

|模型|骨干网络|配置文件|Avg Accuracy|下载链接|
| --- | --- | --- | --- | --- |
|Rosetta|Resnet34_vd|[configs/rec/rec_r34_vd_none_none_ctc.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_r34_vd_none_none_ctc.yml)|79.11%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar)|
|Rosetta|MobileNetV3|[configs/rec/rec_mv3_none_none_ctc.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_mv3_none_none_ctc.yml)|75.80%|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_none_ctc_v2.0_train.tar)|

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

请参考[文本识别训练教程](../../ppocr/model_train/recognition.md)。PaddleOCR对代码进行了模块化，训练不同的识别模型只需要**更换配置文件**即可。 以基于Resnet34_vd骨干网络为例:

### 3.1 训练

```bash linenums="1"
# 单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_r34_vd_none_none_ctc.yml
# 多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r34_vd_none_none_ctc.yml
```

### 3.2 评估

```bash linenums="1"
# GPU评估, Global.pretrained_model为待评估模型
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r34_vd_none_none_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

### 3.3 预测

```bash linenums="1"
python3 tools/infer_rec.py -c configs/rec/rec_r34_vd_none_none_ctc.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

## 4. 推理部署

### 4.1 Python推理

首先将Rosetta文本识别训练过程中保存的模型，转换成inference model。以基于Resnet34_vd骨干网络，在MJSynth和SynthText两个文字识别数据集训练得到的模型为例（ [模型下载地址](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_none_ctc_v2.0_train.tar) )，可以使用如下命令进行转换：

```bash linenums="1"
python3 tools/export_model.py -c configs/rec/rec_r34_vd_none_none_ctc.yml -o Global.pretrained_model=./rec_r34_vd_none_none_ctc_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/rec_rosetta
```

Rosetta文本识别模型推理，可以执行如下命令：

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir="doc/imgs_words/en/word_1.png" --rec_model_dir="./inference/rec_rosetta/" --rec_image_shape="3, 32, 100" --rec_char_dict_path="./ppocr/utils/ic15_dict.txt"
```

推理结果如下所示：

![img](./images/word_1-20240704184113913.png)

```bash linenums="1"
Predicts of doc/imgs_words/en/word_1.png:('joint', 0.9999982714653015)
```

### 4.2 C++推理

暂不支持

### 4.3 Serving服务化部署

暂不支持

### 4.4 更多推理部署

Rosetta模型还支持以下推理部署方式：

- Paddle2ONNX推理：准备好推理模型后，参考[paddle2onnx](../../ppocr/infer_deploy/paddle2onnx.md)教程操作。

## 5. FAQ

## 引用

```bibtex
@inproceedings{2018Rosetta,
  title={Rosetta: Large Scale System for Text Detection and Recognition in Images},
  author={ Borisyuk, Fedor  and  Gordo, Albert  and  Sivakumar, Viswanath },
  booktitle={the 24th ACM SIGKDD International Conference},
  year={2018},
}
```
