# Text Gestalt

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
> [Text Gestalt: Stroke-Aware Scene Text Image Super-Resolution](https://arxiv.org/pdf/2112.08171.pdf)

> Chen, Jingye and Yu, Haiyang and Ma, Jianqi and Li, Bin and Xue, Xiangyang

> AAAI, 2022

参考[FudanOCR](https://github.com/FudanVI/FudanOCR/tree/main/text-gestalt) 数据下载说明，在TextZoom测试集合上超分算法效果如下：

|模型|骨干网络|PSNR_Avg|SSIM_Avg|配置文件|下载链接|
|---|---|---|---|---|---|
|Text Gestalt|tsrn|19.28|0.6560| [configs/sr/sr_tsrn_transformer_strock.yml](../../configs/sr/sr_tsrn_transformer_strock.yml)|[训练模型](https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar)|


<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

请参考[文本识别训练教程](./recognition.md)。PaddleOCR对代码进行了模块化，训练不同的识别模型只需要**更换配置文件**即可。

- 训练

在完成数据准备后，便可以启动训练，训练命令如下：

```
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/sr/sr_tsrn_transformer_strock.yml

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/sr/sr_tsrn_transformer_strock.yml

```

- 评估

```
# GPU 评估， Global.pretrained_model 为待测权重
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/sr/sr_tsrn_transformer_strock.yml -o Global.pretrained_model={path/to/weights}/best_accuracy
```

- 预测：

```
# 预测使用的配置文件必须与训练一致
python3 tools/infer_sr.py -c configs/sr/sr_tsrn_transformer_strock.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words_en/word_52.png
```

![](../imgs_words_en/word_52.png)

执行命令后，上面图像的超分结果如下：

![](../imgs_results/sr_word_52.png)

<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理

首先将文本超分训练过程中保存的模型，转换成inference model。以 Text-Gestalt 训练的[模型](https://paddleocr.bj.bcebos.com/sr_tsrn_transformer_strock_train.tar) 为例，可以使用如下命令进行转换：
```shell
python3 tools/export_model.py -c configs/sr/sr_tsrn_transformer_strock.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.save_inference_dir=./inference/sr_out
```
Text-Gestalt 文本超分模型推理，可以执行如下命令：
```
python3 tools/infer/predict_sr.py --sr_model_dir=./inference/sr_out --image_dir=doc/imgs_words_en/word_52.png --sr_image_shape=3,32,128

```

执行命令后，图像的超分结果如下：

![](../imgs_results/sr_word_52.png)

<a name="4-2"></a>
### 4.2 C++推理

暂未支持

<a name="4-3"></a>
### 4.3 Serving服务化部署

暂未支持

<a name="4-4"></a>
### 4.4 更多推理部署

暂未支持

<a name="5"></a>
## 5. FAQ


## 引用

```bibtex
@inproceedings{chen2022text,
  title={Text gestalt: Stroke-aware scene text image super-resolution},
  author={Chen, Jingye and Yu, Haiyang and Ma, Jianqi and Li, Bin and Xue, Xiangyang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={1},
  pages={285--293},
  year={2022}
}
```
