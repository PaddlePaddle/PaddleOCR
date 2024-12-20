# 通用数学公式识别算法-UniMERNet

## 1. 算法简介

原始项目：
> [https://github.com/opendatalab/UniMERNet](https://github.com/opendatalab/UniMERNet)


`UniMERNet`使用[`UniMERNet通用公式识别数据集`](https://huggingface.co/datasets/wanderkid/UniMER_Dataset/tree/main)进行训练，在对应测试集上的精度如下：

| 模型        | 骨干网络       | 配置文件                                                  | SPE-<br/>BLEU↑ | SPE-<br/>EditDis↓ | CPE-<br/>BLEU↑  |CPE-<br/>EditDis↓ | SCE-<br/>BLEU↑ | SCE-<br/>EditDis↓ | HWE-<br/>BLEU↑ | HWE-<br/>EditDis↓ | 下载链接 |
|-----------|------------|-------------------------------------------------------|:--------------:|:-----------------:|:----------:|:----------------:|:---------:|:-----------------:|:--------------:|:-----------------:|-------|
| UniMERNet | Donut Swin | [rec_unimernet.yml](../../../configs/rec/rec_unimernet.yml) |     0.9187     |      0.0584       |  0.9252    |      0.0596      | 0.6068 |     0.2297        |   0.9157|     0.0546           |[训练模型](https://paddleocr.bj.bcebos.com/contribution/rec_unimernet_train.tar)|

其中，SPE表示简单公式，CPE表示复杂公式，SCE表示扫描捕捉公式，HWE表示手写公式。每种类型的公式示例图如下：
![unimernet_dataset](https://github.com/user-attachments/assets/fb801a36-5614-4031-8585-700bd9f8fb2e)

## 2. 环境配置
请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

此外，需要安装额外的依赖：
```shell
sudo apt-get update
sudo apt-get install libmagickwand-dev
pip install -r docs/algorithm/formula_recognition/requirements.txt
```

## 3. 模型训练、评估、预测

### 3.1 准备数据集

从 [Hugging Face](https://huggingface.co/datasets/wanderkid/UniMER_Dataset/tree/main) 上下载 UniMER-1M.zip 和 UniMER-Test.zip。
从 [好未来平台](https://ai.100tal.com/dataset) 下载 HME100K 数据集。之后， 使用如下命令创建数据集目录，并对数据集进行转换。

```shell
# 创建 UniMERNet 数据集目录
mkdir -p train_data/UniMERNet
# 解压 UniMERNet 、 UniMER-Test.zip 和 HME100K.zip
unzip -d train_data/UniMERNet path/UniMER-1M.zip
unzip -d train_data/UniMERNet path/UniMER-Test.zip
unzip -d train_data/UniMERNet/HME100K train_data/UniMERNet/HME100K/train.zip
unzip -d train_data/UniMERNet/HME100K train_data/UniMERNet/HME100K/test.zip
# 训练集转换   
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet \
     --datatype=unimernet_train \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-1M/train.txt \
     --hme100k_txt_path=train_data/UniMERNet/HME100K/train_labels.txt \
     --output_path=train_data/UniMERNet/train_unimernet_1M.txt
# 测试集转换
# SPE
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet/UniMER-Test/spe \
     --datatype=unimernet_test \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-Test/spe.txt \
     --output_path=train_data/UniMERNet/test_unimernet_spe.txt
# CPE
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet/UniMER-Test/cpe \
     --datatype=unimernet_test \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-Test/cpe.txt \
     --output_path=train_data/UniMERNet/test_unimernet_cpe.txt
# SCE
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet/UniMER-Test/sce \
     --datatype=unimernet_test \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-Test/sce.txt \
     --output_path=train_data/UniMERNet/test_unimernet_sce.txt
# HWE
python ppocr/utils/formula_utils/unimernet_data_convert.py \
     --image_dir=train_data/UniMERNet/UniMER-Test/hwe \
     --datatype=unimernet_test \
     --unimernet_txt_path=train_data/UniMERNet/UniMER-Test/hwe.txt \
     --output_path=train_data/UniMERNet/test_unimernet_hwe.txt
```


### 3.2 下载预训练模型

```shell
# 下载 Texify 预训练模型
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/pretrained/texify.pdparams
```


### 3.3 模型训练

请参考[文本识别训练教程](../../ppocr/model_train/recognition.md)。PaddleOCR对代码进行了模块化，训练`UniMERNet`识别模型时需要**更换配置文件**为`UniMERNet`的[配置文件](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/rec_unimernet.yml)。

#### 启动训练


具体地，在完成数据准备后，便可以启动训练，训练命令如下：
```shell
#单卡训练 (默认训练方式)
python3 tools/train.py -c configs/rec/rec_unimernet.yml \
   -o Global.pretrained_model=./pretrain_models/texify.pdparams
#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3' --ips=127.0.0.1   tools/train.py -c configs/rec/rec_unimernet.yml \
        -o Global.pretrained_model=./pretrain_models/texify.pdparams
```

**注意：**

- 默认每训练 1个epoch（37880 次iteration）进行1次评估，若您更改训练的batch_size，或更换数据集，请在训练时作出如下修改
```
python3  -m paddle.distributed.launch --gpus '0,1,2,3' --ips=127.0.0.1   tools/train.py -c configs/rec/rec_unimernet.yml \
  -o Global.eval_batch_step=[0,{length_of_dataset//batch_size//4}] \
   Global.pretrained_model=./pretrain_models/texify.pdparams
```

### 3.4 评估

可下载已训练完成的[模型文件](https://paddleocr.bj.bcebos.com/contribution/rec_unimernet_train.tar)，使用如下命令进行评估：

```shell
# 注意将pretrained_model的路径设置为本地路径。若使用自行训练保存的模型，请注意修改路径和文件名为{path/to/weights}/{model_name}。
 # SPE 测试集评估
 python3 tools/eval.py -c configs/rec/rec_unimernet.yml -o \
  Eval.dataset.data_dir=./train_data/UniMERNet/UniMER-Test/spe \
  Eval.dataset.label_file_list=["./train_data/UniMERNet/test_unimernet_spe.txt"] \
 Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams
 # CPE 测试集评估
 python3 tools/eval.py -c configs/rec/rec_unimernet.yml -o \
  Eval.dataset.data_dir=./train_data/UniMERNet/UniMER-Test/cpe \
  Eval.dataset.label_file_list=["./train_data/UniMERNet/test_unimernet_cpe.txt"] \
 Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams
 # SCE 测试集评估
  python3 tools/eval.py -c configs/rec/rec_unimernet.yml -o \
  Eval.dataset.data_dir=./train_data/UniMERNet/UniMER-Test/sce \
  Eval.dataset.label_file_list=["./train_data/UniMERNet/test_unimernet_sce.txt"] \
 Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams
 # HWE 测试集评估
 python3 tools/eval.py -c configs/rec/rec_unimernet.yml -o \
  Eval.dataset.data_dir=./train_data/UniMERNet/UniMER-Test/hwe \
  Eval.dataset.label_file_list=["./train_data/UniMERNet/test_unimernet_hwe.txt"] \
 Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams

```

### 3.5 预测

使用如下命令进行单张图片预测：
```shell
# 注意将pretrained_model的路径设置为本地路径。
python3 tools/infer_rec.py -c configs/rec/rec_unimernet.yml \
  -o  Global.infer_img='./docs/datasets/images/pme_demo/0000099.png'\
   Global.pretrained_model=./rec_unimernet_train/best_accuracy.pdparams
# 预测文件夹下所有图像时，可修改infer_img为文件夹，如 Global.infer_img='./doc/datasets/pme_demo/'。
```

## 4. FAQ

1. UniMERNet 数据集来自于[UniMERNet源repo](https://github.com/opendatalab/UniMERNet) 。
