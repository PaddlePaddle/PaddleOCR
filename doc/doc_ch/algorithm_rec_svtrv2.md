# 场景文本识别算法-SVTRv2

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

### SVTRv2算法简介

<a name="1"></a>
[PaddleOCR 算法模型挑战赛 - 赛题一：OCR 端到端识别任务](https://aistudio.baidu.com/competition/detail/1131/0/introduction)排行榜第一算法。主要思路：1、检测和识别模型的Backbone升级为RepSVTR；2、识别教师模型升级为SVTRv2，可识别长文本。


<a name="2"></a>
## 2. 环境配置
请先参考[《运行环境准备》](./environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](./clone.md)克隆项目代码。


<a name="3"></a>
## 3. 模型训练、评估、预测

<a name="3-1"></a>
### 3.1 模型训练


训练命令：
```shell
#单卡训练（训练周期长，不建议）
python3 tools/train.py -c configs/rec/SVTRv2/rec_repsvtr_gtc.yml

#多卡训练，通过--gpus参数指定卡号
# Rec 学生模型
python -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/SVTRv2/rec_repsvtr_gtc.yml
# Rec 教师模型
python -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/SVTRv2/rec_svtrv2_gtc.yml
# Rec 蒸馏训练
python -m paddle.distributed.launch --gpus '0,1,2,3,4,5,6,7'  tools/train.py -c configs/rec/SVTRv2/rec_svtrv2_gtc_distill.yml
```

<a name="3-2"></a>
### 3.2 评估


```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/SVTRv2/rec_repsvtr_gtc.yml -o Global.pretrained_model=output/rec_repsvtr_gtc/best_accuracy
```

<a name="3-3"></a>
### 3.3 预测

使用如下命令进行单张图片预测：
```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c tools/eval.py -c configs/rec/SVTRv2/rec_repsvtr_gtc.yml -o Global.pretrained_model=output/rec_repsvtr_gtc/best_accuracy Global.infer_img='./doc/imgs_words_en/word_10.png'
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/imgs_words_en/'。
```


<a name="4"></a>
## 4. 推理部署

<a name="4-1"></a>
### 4.1 Python推理
首先将训练得到best模型，转换成inference model，以RepSVTR为例，可以使用如下命令进行转换：

```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/export_model.py -c configs/rec/SVTRv2/rec_repsvtr_gtc.yml -o Global.pretrained_model=output/rec_repsvtr_gtc/best_accuracy Global.save_inference_dir=./inference/rec_repsvtr_infer
```

**注意：**
- 如果您是在自己的数据集上训练的模型，并且调整了字典文件，请注意修改配置文件中的`character_dict_path`是否为所正确的字典文件。

转换成功后，在目录下有三个文件：
```
./inference/rec_repsvtr_infer/
    ├── inference.pdiparams         # 识别inference模型的参数文件
    ├── inference.pdiparams.info    # 识别inference模型的参数信息，可忽略
    └── inference.pdmodel           # 识别inference模型的program文件
```


执行如下命令进行模型推理：

```shell
python3 tools/infer/predict_rec.py --image_dir='./doc/imgs_words_en/word_10.png' --rec_model_dir='./inference/rec_repsvtr_infer/'
# 预测文件夹下所有图像时，可修改image_dir为文件夹，如 --image_dir='./doc/imgs_words_en/'。
```
![](../imgs_words_en/word_10.png)

执行命令后，上面图像的预测结果（识别的文本和得分）会打印到屏幕上，示例如下：
结果如下：
```shell
Predicts of ./doc/imgs_words_en/word_10.png:('pain', 0.9999998807907104)
```

**注意**：

- 如果您调整了训练时的输入分辨率，需要通过参数`rec_image_shape`设置为您需要的识别图像形状。
- 在推理时需要设置参数`rec_char_dict_path`指定字典，如果您修改了字典，请修改该参数为您的字典文件。
- 如果您修改了预处理方法，需修改`tools/infer/predict_rec.py`中SVTR的预处理为您的预处理方法。

<a name="4-2"></a>
### 4.2 C++推理部署

由于C++预处理后处理还未支持SVTRv2

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
@article{Du2022SVTR,
  title     = {SVTR: Scene Text Recognition with a Single Visual Model},
  author    = {Du, Yongkun and Chen, Zhineng and Jia, Caiyan and Yin, Xiaoting and Zheng, Tianlun and Li, Chenxia and Du, Yuning and Jiang, Yu-Gang},
  booktitle = {IJCAI},
  year      = {2022},
  url       = {https://arxiv.org/abs/2205.00159}
}
```
