---
comments: true
---

# 场景文本识别算法-ABINet

## 1. 算法简介

论文信息：
> [ABINet: Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition](https://openaccess.thecvf.com/content/CVPR2021/papers/Fang_Read_Like_Humans_Autonomous_Bidirectional_and_Iterative_Language_Modeling_for_CVPR_2021_paper.pdf)
> Shancheng Fang and Hongtao Xie and Yuxin Wang and Zhendong Mao and Yongdong Zhang
> CVPR, 2021

`ABINet`使用MJSynth和SynthText两个文字识别数据集训练，在IIIT, SVT, IC03, IC13, IC15, SVTP, CUTE数据集上进行评估，算法复现效果如下：

|模型|骨干网络|配置文件|Acc|下载链接|
| --- | --- | --- | --- | --- |
|ABINet|ResNet45|[rec_r45_abinet.yml](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_r45_abinet.yml)|90.75%|[预训练、训练模型](https://paddleocr.bj.bcebos.com/rec_r45_abinet_train.tar)|

## 2. 环境配置

请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

## 3. 模型训练、评估、预测

### 3.1 模型训练

请参考[文本识别训练教程](../../ppocr/model_train/recognition.md)。PaddleOCR对代码进行了模块化，训练`ABINet`识别模型时需要**更换配置文件**为`ABINet`的[配置文件](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/rec_r45_abinet.yml)。

#### 启动训练

具体地，在完成数据准备后，便可以启动训练，训练命令如下：

```bash linenums="1"
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/rec_r45_abinet.yml

# 多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_r45_abinet.yml
```

### 3.2 评估

可下载已训练完成的模型文件，使用如下命令进行评估：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_r45_abinet.yml -o Global.pretrained_model=./rec_r45_abinet_train/best_accuracy
```

### 3.3 预测

使用如下命令进行单张图片预测：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c configs/rec/rec_r45_abinet.yml -o Global.infer_img='./doc/imgs_words_en/word_10.png' Global.pretrained_model=./rec_r45_abinet_train/best_accuracy
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/imgs_words_en/'。
```

## 4. 推理部署

### 4.1 Python推理

首先将训练得到best模型，转换成inference model。这里以训练完成的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/rec_r45_abinet_train.tar) )，可以使用如下命令进行转换：

```bash linenums="1"
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c configs/rec/rec_r45_abinet.yml -o Global.pretrained_model=./rec_r45_abinet_train/best_accuracy Global.save_inference_dir=./inference/rec_r45_abinet/
```

**注意：**

- 如果您是在自己的数据集上训练的模型，并且调整了字典文件，请注意修改配置文件中的`character_dict_path`是否是所需要的字典文件。
- 如果您修改了训练时的输入大小，请修改`tools/export_model.py`文件中的对应ABINet的`infer_shape`。

转换成功后，在目录下有三个文件：

```text linenums="1"
/inference/rec_r45_abinet/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```

执行如下命令进行模型推理：

```bash linenums="1"
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_r45_abinet/' --rec_algorithm='ABINet' --rec_image_shape='3,32,128' --rec_char_dict_path='./ppocr/utils/ic15_dict.txt'
# 预测文件夹下所有图像时，可修改image_dir为文件夹，如 --image_dir='./doc/imgs_words_en/'。
```

![img](./images/word_10.png)

执行命令后，上面图像的预测结果（识别的文本和得分）会打印到屏幕上，示例如下：
结果如下：

```bash linenums="1"
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9999995231628418)
```

**注意**：

- 训练上述模型采用的图像分辨率是[3，32，128]，需要通过参数`rec_image_shape`设置为您训练时的识别图像形状。
- 在推理时需要设置参数`rec_char_dict_path`指定字典，如果您修改了字典，请修改该参数为您的字典文件。
- 如果您修改了预处理方法，需修改`tools/infer/predict_rec.py`中ABINet的预处理为您的预处理方法。

### 4.2 C++推理部署

由于C++预处理后处理还未支持ABINet，所以暂未支持

### 4.3 Serving服务化部署

暂不支持

### 4.4 更多推理部署

暂不支持

## 5. FAQ

1. MJSynth和SynthText两种数据集来自于[ABINet源repo](https://github.com/FangShancheng/ABINet) 。
2. 我们使用ABINet作者提供的预训练模型进行finetune训练。

## 引用

```bibtex
@article{Fang2021ABINet,
  title     = {ABINet: Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling for Scene Text Recognition},
  author    = {Shancheng Fang and Hongtao Xie and Yuxin Wang and Zhendong Mao and Yongdong Zhang},
  booktitle = {CVPR},
  year      = {2021},
  url       = {https://arxiv.org/abs/2103.06495},
  pages     = {7098-7107}
}
```
