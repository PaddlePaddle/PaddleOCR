# STAR-Net

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
> [STAR-Net: a spatial attention residue network for scene text recognition.](http://www.bmva.org/bmvc/2016/papers/paper043/paper043.pdf)

> Wei Liu, Chaofeng Chen, Kwan-Yee K. Wong, Zhizhong Su and Junyu Han.

> BMVC, pages 43.1-43.13, 2016

参考[DTRB](https://arxiv.org/abs/1904.01906) 文字识别训练和评估流程，使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法效果如下：

|模型|骨干网络|Avg Accuracy|模型存储命名|下载链接|
|---|---|---|---|---|
|StarNet|Resnet34_vd|84.44%|rec_r34_vd_tps_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_tps_bilstm_ctc_v2.0_train.tar)|
|StarNet|MobileNetV3|81.42%|rec_mv3_tps_bilstm_ctc|[训练模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_tps_bilstm_ctc_v2.0_train.tar)|


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

首先将 STAR-Net 文本识别训练过程中保存的模型，转换成inference model。以基于Resnet34_vd骨干网络，使用MJSynth和SynthText两个英文文本识别合成数据集训练的[模型](https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_r34_vd_none_bilstm_ctc_v2.0_train.tar) 为例，可以使用如下命令进行转换：
```shell
python3 tools/export_model.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml -o Global.pretrained_model=./rec_r34_vd_tps_bilstm_ctc_v2.0_train/best_accuracy  Global.save_inference_dir=./inference/rec_starnet
```
STAR-Net 文本识别模型推理，可以执行如下命令：

```shell
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./inference/rec_starnet/" --rec_image_shape="3, 32, 100" --rec_char_dict_path="./ppocr/utils/ic15_dict.txt"
```

![](../imgs_words_en/word_336.png)

执行命令后，上面图像的识别结果如下：

```bash
Predicts of ./doc/imgs_words_en/word_336.png:('super', 0.9999073)
```

**注意**：由于上述模型是参考[DTRB](https://arxiv.org/abs/1904.01906)文本识别训练和评估流程，与超轻量级中文识别模型训练有两方面不同：

- 训练时采用的图像分辨率不同，训练上述模型采用的图像分辨率是[3，32，100]，而中文模型训练时，为了保证长文本的识别效果，训练时采用的图像分辨率是[3, 32, 320]。预测推理程序默认的的形状参数是训练中文采用的图像分辨率，即[3, 32, 320]。因此，这里推理上述英文模型时，需要通过参数rec_image_shape设置识别图像的形状。

- 字符列表，DTRB论文中实验只是针对26个小写英文本母和10个数字进行实验，总共36个字符。所有大小字符都转成了小写字符，不在上面列表的字符都忽略，认为是空格。因此这里没有输入字符字典，而是通过如下命令生成字典.因此在推理时需要设置参数rec_char_dict_path，指定为英文字典"./ppocr/utils/ic15_dict.txt"。

```
self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
dict_character = list(self.character_str)
```


<a name="4-2"></a>
### 4.2 C++推理

准备好推理模型后，参考[cpp infer](../../deploy/cpp_infer/)教程进行操作即可。

<a name="4-3"></a>
### 4.3 Serving服务化部署

准备好推理模型后，参考[pdserving](../../deploy/pdserving/)教程进行Serving服务化部署，包括Python Serving和C++ Serving两种模式。

<a name="4-4"></a>
### 4.4 更多推理部署

STAR-Net模型还支持以下推理部署方式：

- Paddle2ONNX推理：准备好推理模型后，参考[paddle2onnx](../../deploy/paddle2onnx/)教程操作。

<a name="5"></a>
## 5. FAQ


## 引用

```bibtex
@inproceedings{liu2016star,
  title={STAR-Net: a spatial attention residue network for scene text recognition.},
  author={Liu, Wei and Chen, Chaofeng and Wong, Kwan-Yee K and Su, Zhizhong and Han, Junyu},
  booktitle={BMVC},
  volume={2},
  pages={7},
  year={2016}
}
```
