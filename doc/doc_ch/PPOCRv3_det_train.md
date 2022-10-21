
# PP-OCRv3 文本检测模型训练

- [1. 简介](#1)
- [2. PP-OCRv3检测训练](#2)
- [3. 基于PP-OCRv3检测的finetune训练](#3)

<a name="1"></a>
## 1. 简介

PP-OCRv3在PP-OCRv2的基础上进一步升级。本节介绍PP-OCRv3检测模型的训练步骤。有关PP-OCRv3策略介绍参考[文档](./PP-OCRv3_introduction.md)。


<a name="2"></a>
## 2. 检测训练

PP-OCRv3检测模型是对PP-OCRv2中的[CML](https://arxiv.org/pdf/2109.03144.pdf)（Collaborative Mutual Learning) 协同互学习文本检测蒸馏策略进行了升级。PP-OCRv3分别针对检测教师模型和学生模型进行进一步效果优化。其中，在对教师模型优化时，提出了大感受野的PAN结构LK-PAN和引入了DML（Deep Mutual Learning）蒸馏策略；在对学生模型优化时，提出了残差注意力机制的FPN结构RSE-FPN。

PP-OCRv3检测训练包括两个步骤：
- 步骤1：采用DML蒸馏方法训练检测教师模型
- 步骤2：使用步骤1得到的教师模型采用CML方法训练出轻量学生模型


### 2.1 准备数据和运行环境

训练数据采用icdar2015数据，准备训练集步骤参考[ocr_dataset](./dataset/ocr_datasets.md).

运行环境准备参考[文档](./installation.md)。


### 2.2 训练教师模型

教师模型训练的配置文件是[ch_PP-OCRv3_det_dml.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml)。教师模型模型结构的Backbone、Neck、Head分别为Resnet50, LKPAN, DBHead，采用DML的蒸馏方法训练。有关配置文件的详细介绍参考[文档](./knowledge_distillation)。


下载ImageNet预训练模型：
```
# 下载ResNet50_vd的预训练模型
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/ResNet50_vd_ssld_pretrained.pdparams
```

**启动训练**
```
# 单卡训练
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
    -o Architecture.Models.Student.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
       Architecture.Models.Student2.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
       Global.save_model_dir=./output/
# 如果要使用多GPU分布式训练，请使用如下命令：
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
    -o Architecture.Models.Student.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
       Architecture.Models.Student2.pretrained=./pretrain_models/ResNet50_vd_ssld_pretrained \
       Global.save_model_dir=./output/
```

训练过程中保存的模型在output目录下，包含以下文件：
```
best_accuracy.states  
best_accuracy.pdparams  # 默认保存最优精度的模型参数
best_accuracy.pdopt     # 默认保存最优精度的优化器相关参数
latest.states  
latest.pdparams  # 默认保存的最新模型参数
latest.pdopt     # 默认保存的最新模型的优化器相关参数
```
其中，best_accuracy是保存的精度最高的模型参数，可以直接使用该模型评估。

模型评估命令如下：
```
python3 tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml -o Global.checkpoints=./output/best_accuracy
```

训练的教师模型结构更大，精度更高，用于提升学生模型的精度。

**提取教师模型参数**
best_accuracy包含两个模型的参数，分别对应配置文件中的Student，Student2。提取Student的参数方法如下：

```
import paddle
# 加载预训练模型
all_params = paddle.load("output/best_accuracy.pdparams")
# 查看权重参数的keys
print(all_params.keys())
# 模型的权重提取
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# 查看模型权重参数的keys
print(s_params.keys())
# 保存
paddle.save(s_params, "./pretrain_models/dml_teacher.pdparams")
```

提取出来的模型参数可以用于模型进一步的finetune训练或者蒸馏训练。

### 2.3 训练学生模型

训练学生模型的配置文件是[ch_PP-OCRv3_det_cml.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)
上一节训练得到的教师模型作为监督，采用CML方式训练得到轻量的学生模型。

下载学生模型的ImageNet预训练模型：
```
# 下载MobileNetV3的预训练模型
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/MobileNetV3_large_x0_5_pretrained.pdparams
```

**启动训练**

```
# 单卡训练
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
    -o Architecture.Models.Student.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
       Architecture.Models.Student2.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
       Architecture.Models.Teacher.pretrained=./pretrain_models/dml_teacher \
       Global.save_model_dir=./output/
# 如果要使用多GPU分布式训练，请使用如下命令：
python3  -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
    -o Architecture.Models.Student.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
       Architecture.Models.Student2.pretrained=./pretrain_models/MobileNetV3_large_x0_5_pretrained \
       Architecture.Models.Teacher.pretrained=./pretrain_models/dml_teacher \
       Global.save_model_dir=./output/
```

训练过程中保存的模型在output目录下，
模型评估命令如下：
```
python3 tools/eval.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml -o Global.checkpoints=./output/best_accuracy
```

best_accuracy包含三个模型的参数，分别对应配置文件中的Student，Student2，Teacher。提取Student参数的方法如下：

```
import paddle
# 加载预训练模型
all_params = paddle.load("output/best_accuracy.pdparams")
# 查看权重参数的keys
print(all_params.keys())
# 模型的权重提取
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# 查看模型权重参数的keys
print(s_params.keys())
# 保存
paddle.save(s_params, "./pretrain_models/cml_student.pdparams")
```

提取出来的Student的参数可用于模型部署或者做进一步的finetune训练。



<a name="3"></a>
## 3. 基于PP-OCRv3检测finetune训练

本节介绍如何使用PP-OCRv3检测模型在其他场景上的finetune训练。

finetune训练适用于三种场景：
- 基于CML蒸馏方法的finetune训练，适用于教师模型在使用场景上精度高于PP-OCRv3检测模型，且希望得到一个轻量检测模型。
- 基于PP-OCRv3轻量检测模型的finetune训练，无需训练教师模型，希望在PP-OCRv3检测模型基础上提升使用场景上的精度。
- 基于DML蒸馏方法的finetune训练，适用于采用DML方法进一步提升精度的场景。


**基于CML蒸馏方法的finetune训练**

下载PP-OCRv3训练模型：
```
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar xf ch_PP-OCRv3_det_distill_train.tar
```
ch_PP-OCRv3_det_distill_train/best_accuracy.pdparams包含CML配置文件中Student、Student2、Teacher模型的参数。

启动训练：

```
# 单卡训练
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
    -o Global.pretrained_model=./ch_PP-OCRv3_det_distill_train/best_accuracy \
       Global.save_model_dir=./output/
# 如果要使用多GPU分布式训练，请使用如下命令：
python3  -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml \
    -o Global.pretrained_model=./ch_PP-OCRv3_det_distill_train/best_accuracy  \
       Global.save_model_dir=./output/
```

**基于PP-OCRv3轻量检测模型的finetune训练**


下载PP-OCRv3训练模型，并提取Student结构的模型参数：
```
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar xf ch_PP-OCRv3_det_distill_train.tar
```

提取Student参数的方法如下：

```
import paddle
# 加载预训练模型
all_params = paddle.load("output/best_accuracy.pdparams")
# 查看权重参数的keys
print(all_params.keys())
# 模型的权重提取
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# 查看模型权重参数的keys
print(s_params.keys())
# 保存
paddle.save(s_params, "./student.pdparams")
```

使用配置文件[ch_PP-OCRv3_det_student.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.5/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml)训练。

**启动训练**

```
# 单卡训练
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml \
    -o Global.pretrained_model=./student \
       Global.save_model_dir=./output/
# 如果要使用多GPU分布式训练，请使用如下命令：
python3  -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_student.yml \
    -o Global.pretrained_model=./student \
       Global.save_model_dir=./output/
```


**基于DML蒸馏方法的finetune训练**

以ch_PP-OCRv3_det_distill_train中的Teacher模型为例，首先提取Teacher结构的参数，方法如下：
```
import paddle
# 加载预训练模型
all_params = paddle.load("ch_PP-OCRv3_det_distill_train/best_accuracy.pdparams")
# 查看权重参数的keys
print(all_params.keys())
# 模型的权重提取
s_params = {key[len("Teacher."):]: all_params[key] for key in all_params if "Teacher." in key}
# 查看模型权重参数的keys
print(s_params.keys())
# 保存
paddle.save(s_params, "./teacher.pdparams")
```

**启动训练**
```
# 单卡训练
python3 tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
    -o Architecture.Models.Student.pretrained=./teacher \
       Architecture.Models.Student2.pretrained=./teacher \
       Global.save_model_dir=./output/
# 如果要使用多GPU分布式训练，请使用如下命令：
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml \
    -o Architecture.Models.Student.pretrained=./teacher \
       Architecture.Models.Student2.pretrained=./teacher \
       Global.save_model_dir=./output/
```
