# 添加新算法

PaddleOCR将一个算法分解为以下几个部分，并对各部分进行模块化处理，方便快速组合出新的算法。

* [1. 数据加载和处理](#1)
* [2. 网络](#2)
* [3. 后处理](#3)
* [4. 损失函数](#4)
* [5. 指标评估](#5)
* [6. 优化器](#6)

下面将分别对每个部分进行介绍，并介绍如何在该部分里添加新算法所需模块。

<a name="1"></a>

## 1. 数据加载和处理

数据加载和处理由不同的模块(module)组成，其完成了图片的读取、数据增强和label的制作。这一部分在[ppocr/data](../../ppocr/data)下。 各个文件及文件夹作用说明如下:

```bash
ppocr/data/
├── imaug             # 图片的读取、数据增强和label制作相关的文件
│   ├── label_ops.py  # 对label进行变换的modules
│   ├── operators.py  # 对image进行变换的modules
│   ├──.....
├── __init__.py
├── lmdb_dataset.py   # 读取lmdb的数据集的dataset
└── simple_dataset.py # 读取以`image_path\tgt`形式保存的数据集的dataset
```

PaddleOCR内置了大量图像操作相关模块，对于没有没有内置的模块可通过如下步骤添加:

1. 在 [ppocr/data/imaug](../../ppocr/data/imaug) 文件夹下新建文件，如my_module.py。
2. 在 my_module.py 文件内添加相关代码，示例代码如下:

```python
class MyModule:
    def __init__(self, *args, **kwargs):
        # your init code
        pass

    def __call__(self, data):
        img = data['image']
        label = data['label']
        # your process code

        data['image'] = img
        data['label'] = label
        return data
```

3. 在 [ppocr/data/imaug/\__init\__.py](../../ppocr/data/imaug/__init__.py) 文件内导入添加的模块。

数据处理的所有处理步骤由不同的模块顺序执行而成，在config文件中按照列表的形式组合并执行。如:

```yaml
# angle class data process
transforms:
  - DecodeImage: # load image
      img_mode: BGR
      channel_first: False
  - MyModule:
      args1: args1
      args2: args2
  - KeepKeys:
      keep_keys: [ 'image', 'label' ] # dataloader will return list in this order
```

<a name="2"></a>

## 2. 网络

网络部分完成了网络的组网操作，PaddleOCR将网络划分为四部分，这一部分在[ppocr/modeling](../../ppocr/modeling)下。 进入网络的数据将按照顺序(transforms->backbones->
necks->heads)依次通过这四个部分。

```bash
├── architectures # 网络的组网代码
├── transforms    # 网络的图像变换模块
├── backbones     # 网络的特征提取模块
├── necks         # 网络的特征增强模块
└── heads         # 网络的输出模块
```

PaddleOCR内置了DB,EAST,SAST,CRNN和Attention等算法相关的常用模块，对于没有内置的模块可通过如下步骤添加，四个部分添加步骤一致，以backbones为例:

1. 在 [ppocr/modeling/backbones](../../ppocr/modeling/backbones) 文件夹下新建文件，如my_backbone.py。
2. 在 my_backbone.py 文件内添加相关代码，示例代码如下:

```python
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MyBackbone(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(MyBackbone, self).__init__()
        # your init code
        self.conv = nn.xxxx

    def forward(self, inputs):
        # your network forward
        y = self.conv(inputs)
        return y
```

3. 在 [ppocr/modeling/backbones/\__init\__.py](../../ppocr/modeling/backbones/__init__.py)文件内导入添加的模块。

在完成网络的四部分模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
Architecture:
  model_type: rec
  algorithm: CRNN
  Transform:
    name: MyTransform
    args1: args1
    args2: args2
  Backbone:
    name: MyBackbone
    args1: args1
  Neck:
    name: MyNeck
    args1: args1
  Head:
    name: MyHead
    args1: args1
```

<a name="3"></a>

## 3. 后处理

后处理实现解码网络输出获得文本框或者识别到的文字。这一部分在[ppocr/postprocess](../../ppocr/postprocess)下。
PaddleOCR内置了DB,EAST,SAST,CRNN和Attention等算法相关的后处理模块，对于没有内置的组件可通过如下步骤添加:

1. 在 [ppocr/postprocess](../../ppocr/postprocess) 文件夹下新建文件，如 my_postprocess.py。
2. 在 my_postprocess.py 文件内添加相关代码，示例代码如下:

```python
import paddle


class MyPostProcess:
    def __init__(self, *args, **kwargs):
        # your init code
        pass

    def __call__(self, preds, label=None, *args, **kwargs):
        if isinstance(preds, paddle.Tensor):
            preds = preds.numpy()
        # you preds decode code
        preds = self.decode_preds(preds)
        if label is None:
            return preds
        # you label decode code
        label = self.decode_label(label)
        return preds, label

    def decode_preds(self, preds):
        # you preds decode code
        pass

    def decode_label(self, preds):
        # you label decode code
        pass
```

3. 在 [ppocr/postprocess/\__init\__.py](../../ppocr/postprocess/__init__.py)文件内导入添加的模块。

在后处理模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
PostProcess:
  name: MyPostProcess
  args1: args1
  args2: args2
```

<a name="4"></a>

## 4. 损失函数

损失函数用于计算网络输出和label之间的距离。这一部分在[ppocr/losses](../../ppocr/losses)下。
PaddleOCR内置了DB,EAST,SAST,CRNN和Attention等算法相关的损失函数模块，对于没有内置的模块可通过如下步骤添加:

1. 在 [ppocr/losses](../../ppocr/losses) 文件夹下新建文件，如 my_loss.py。
2. 在 my_loss.py 文件内添加相关代码，示例代码如下:

```python
import paddle
from paddle import nn


class MyLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(MyLoss, self).__init__()
        # you init code
        pass

    def __call__(self, predicts, batch):
        label = batch[1]
        # your loss code
        loss = self.loss(input=predicts, label=label)
        return {'loss': loss}
```

3. 在 [ppocr/losses/\__init\__.py](../../ppocr/losses/__init__.py)文件内导入添加的模块。

在损失函数添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
Loss:
  name: MyLoss
  args1: args1
  args2: args2
```

<a name="5"></a>

## 5. 指标评估

指标评估用于计算网络在当前batch上的性能。这一部分在[ppocr/metrics](../../ppocr/metrics)下。 PaddleOCR内置了检测，分类和识别等算法相关的指标评估模块，对于没有内置的模块可通过如下步骤添加:

1. 在 [ppocr/metrics](../../ppocr/metrics) 文件夹下新建文件，如my_metric.py。
2. 在 my_metric.py 文件内添加相关代码，示例代码如下:

```python

class MyMetric(object):
    def __init__(self, main_indicator='acc', **kwargs):
        # main_indicator is used for select best model
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch, *args, **kwargs):
        # preds is out of postprocess
        # batch is out of dataloader
        labels = batch[1]
        cur_correct_num = 0
        cur_all_num = 0
        # you metric code
        self.correct_num += cur_correct_num
        self.all_num += cur_all_num
        return {'acc': cur_correct_num / cur_all_num, }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = self.correct_num / self.all_num
        self.reset()
        return {'acc': acc}

    def reset(self):
        # reset metric
        self.correct_num = 0
        self.all_num = 0

```

3. 在 [ppocr/metrics/\__init\__.py](../../ppocr/metrics/__init__.py)文件内导入添加的模块。

在指标评估模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
Metric:
  name: MyMetric
  main_indicator: acc
```

<a name="6"></a>

## 6. 优化器

优化器用于训练网络。优化器内部还包含了网络正则化和学习率衰减模块。 这一部分在[ppocr/optimizer](../../ppocr/optimizer)下。 PaddleOCR内置了`Momentum`,`Adam`
和`RMSProp`等常用的优化器模块，`Linear`,`Cosine`,`Step`和`Piecewise`等常用的正则化模块与`L1Decay`和`L2Decay`等常用的学习率衰减模块。
对于没有内置的模块可通过如下步骤添加，以`optimizer`为例:

1. 在 [ppocr/optimizer/optimizer.py](../../ppocr/optimizer/optimizer.py) 文件内创建自己的优化器，示例代码如下:

```python
from paddle import optimizer as optim


class MyOptim(object):
    def __init__(self, learning_rate=0.001, *args, **kwargs):
        self.learning_rate = learning_rate

    def __call__(self, parameters):
        # It is recommended to wrap the built-in optimizer of paddle
        opt = optim.XXX(
            learning_rate=self.learning_rate,
            parameters=parameters)
        return opt

```

在优化器模块添加之后，只需要配置文件中进行配置即可使用，如:

```yaml
Optimizer:
  name: MyOptim
  args1: args1
  args2: args2
  lr:
    name: Cosine
    learning_rate: 0.001
  regularizer:
    name: 'L2'
    factor: 0
```
