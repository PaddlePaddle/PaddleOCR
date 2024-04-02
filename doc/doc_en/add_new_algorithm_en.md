# Add New Algorithm

PaddleOCR decomposes an algorithm into the following parts, and modularizes each part to make it more convenient to develop new algorithms.

* Data loading and processing
* Network
* Post-processing
* Loss
* Metric
* Optimizer

The following will introduce each part separately, and introduce how to add the modules required for the new algorithm.


## Data loading and processing

Data loading and processing are composed of different modules, which complete the image reading, data augment and label production. This part is under [ppocr/data](../../ppocr/data). The explanation of each file and folder are as follows:

```bash
ppocr/data/
├── imaug             # Scripts for image reading, data augment and label production
│   ├── label_ops.py  # Modules that transform the label
│   ├── operators.py  # Modules that transform the image
│   ├──.....
├── __init__.py
├── lmdb_dataset.py   # The dataset that reads the lmdb
└── simple_dataset.py # Read the dataset saved in the form of `image_path\tgt`
```

PaddleOCR has a large number of built-in image operation related modules. For modules that are not built-in, you can add them through the following steps:

1. Create a new file under the [ppocr/data/imaug](../../ppocr/data/imaug) folder, such as my_module.py.
2. Add code in the my_module.py file, the sample code is as follows:

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

3. Import the added module in the [ppocr/data/imaug/\__init\__.py](../../ppocr/data/imaug/__init__.py) file.

All different modules of data processing are executed by sequence, combined and executed in the form of a list in the config file. Such as:

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

## Network

The network part completes the construction of the network, and PaddleOCR divides the network into four parts, which are under [ppocr/modeling](../../ppocr/modeling). The data entering the network will pass through these four parts in sequence(transforms->backbones->
necks->heads).

```bash
├── architectures # Code for building network
├── transforms    # Image Transformation Module
├── backbones     # Feature extraction module
├── necks         # Feature enhancement module
└── heads         # Output module
```

PaddleOCR has built-in commonly used modules related to algorithms such as DB, EAST, SAST, CRNN and Attention. For modules that do not have built-in, you can add them through the following steps, the four parts are added in the same steps, take backbones as an example:

1. Create a new file under the [ppocr/modeling/backbones](../../ppocr/modeling/backbones) folder, such as my_backbone.py.
2. Add code in the my_backbone.py file, the sample code is as follows:

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

3. Import the added module in the [ppocr/modeling/backbones/\__init\__.py](../../ppocr/modeling/backbones/__init__.py) file.

After adding the four-part modules of the network, you only need to configure them in the configuration file to use, such as:

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

## Post-processing

Post-processing realizes decoding network output to obtain text box or recognized text. This part is under [ppocr/postprocess](../../ppocr/postprocess).
PaddleOCR has built-in post-processing modules related to algorithms such as DB, EAST, SAST, CRNN and Attention. For components that are not built-in, they can be added through the following steps:

1. Create a new file under the [ppocr/postprocess](../../ppocr/postprocess) folder, such as my_postprocess.py.
2. Add code in the my_postprocess.py file, the sample code is as follows:

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

3. Import the added module in the [ppocr/postprocess/\__init\__.py](../../ppocr/postprocess/__init__.py) file.

After the post-processing module is added, you only need to configure it in the configuration file to use, such as:

```yaml
PostProcess:
  name: MyPostProcess
  args1: args1
  args2: args2
```

## Loss

The loss function is used to calculate the distance between the network output and the label. This part is under [ppocr/losses](../../ppocr/losses).
PaddleOCR has built-in loss function modules related to algorithms such as DB, EAST, SAST, CRNN and Attention. For modules that do not have built-in modules, you can add them through the following steps:

1. Create a new file in the [ppocr/losses](../../ppocr/losses) folder, such as my_loss.py.
2. Add code in the my_loss.py file, the sample code is as follows:

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

3. Import the added module in the [ppocr/losses/\__init\__.py](../../ppocr/losses/__init__.py) file.

After the loss function module is added, you only need to configure it in the configuration file to use it, such as:

```yaml
Loss:
  name: MyLoss
  args1: args1
  args2: args2
```

## Metric

Metric is used to calculate the performance of the network on the current batch. This part is under [ppocr/metrics](../../ppocr/metrics). PaddleOCR has built-in evaluation modules related to algorithms such as detection, classification and recognition. For modules that do not have built-in modules, you can add them through the following steps:

1. Create a new file under the [ppocr/metrics](../../ppocr/metrics) folder, such as my_metric.py.
2. Add code in the my_metric.py file, the sample code is as follows:

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

3. Import the added module in the [ppocr/metrics/\__init\__.py](../../ppocr/metrics/__init__.py) file.

After the metric module is added, you only need to configure it in the configuration file to use it, such as:

```yaml
Metric:
  name: MyMetric
  main_indicator: acc
```

## Optimizer

The optimizer is used to train the network. The optimizer also contains network regularization and learning rate decay modules. This part is under [ppocr/optimizer](../../ppocr/optimizer). PaddleOCR has built-in
Commonly used optimizer modules such as `Momentum`, `Adam` and `RMSProp`, common regularization modules such as `Linear`, `Cosine`, `Step` and `Piecewise`, and common learning rate decay modules such as `L1Decay` and `L2Decay`.
Modules without built-in can be added through the following steps, take `optimizer` as an example:

1. Create your own optimizer in the [ppocr/optimizer/optimizer.py](../../ppocr/optimizer/optimizer.py) file, the sample code is as follows:

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

After the optimizer module is added, you only need to configure it in the configuration file to use, such as:

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
