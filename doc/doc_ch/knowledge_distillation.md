<a name="0"></a>
# 知识蒸馏

+ [知识蒸馏](#0)
  + [1. 简介](#1)
    - [1.1 知识蒸馏介绍](#11)
    - [1.2 PaddleOCR知识蒸馏简介](#12)
  + [2. 配置文件解析](#2)
    + [2.1 识别配置文件解析](#21)
      - [2.1.1 模型结构](#211)
      - [2.1.2 损失函数](#212)
      - [2.1.3 后处理](#213)
      - [2.1.4 指标计算](#214)
      - [2.1.5 蒸馏模型微调](#215)
    + [2.2 检测配置文件解析](#22)
      - [2.2.1 模型结构](#221)
      - [2.2.2 损失函数](#222)
      - [2.2.3 后处理](#223)
      - [2.2.4 蒸馏指标计算](#224)
      - [2.2.5 检测蒸馏模型Fine-tune](#225)

<a name="1"></a>
## 1. 简介
<a name="11"></a>
### 1.1 知识蒸馏介绍

近年来，深度神经网络在计算机视觉、自然语言处理等领域被验证是一种极其有效的解决问题的方法。通过构建合适的神经网络，加以训练，最终网络模型的性能指标基本上都会超过传统算法。

在数据量足够大的情况下，通过合理构建网络模型的方式增加其参数量，可以显著改善模型性能，但是这又带来了模型复杂度急剧提升的问题。大模型在实际场景中使用的成本较高。

深度神经网络一般有较多的参数冗余，目前有几种主要的方法对模型进行压缩，减小其参数量。如裁剪、量化、知识蒸馏等，其中知识蒸馏是指使用教师模型(teacher model)去指导学生模型(student model)学习特定任务，保证小模型在参数量不变的情况下，得到比较大的性能提升。

此外，在知识蒸馏任务中，也衍生出了互学习的模型训练方法，论文[Deep Mutual Learning](https://arxiv.org/abs/1706.00384)中指出，使用两个完全相同的模型在训练的过程中互相监督，可以达到比单个模型训练更好的效果。

<a name="12"></a>
### 1.2 PaddleOCR知识蒸馏简介

无论是大模型蒸馏小模型，还是小模型之间互相学习，更新参数，他们本质上是都是不同模型之间输出或者特征图(feature map)之间的相互监督，区别仅在于 (1) 模型是否需要固定参数。(2) 模型是否需要加载预训练模型。

对于大模型蒸馏小模型的情况，大模型一般需要加载预训练模型并固定参数；对于小模型之间互相蒸馏的情况，小模型一般都不加载预训练模型，参数也都是可学习的状态。

在知识蒸馏任务中，不只有2个模型之间进行蒸馏的情况，多个模型之间互相学习的情况也非常普遍。因此在知识蒸馏代码框架中，也有必要支持该种类别的蒸馏方法。

PaddleOCR中集成了知识蒸馏的算法，具体地，有以下几个主要的特点：
- 支持任意网络的互相学习，不要求子网络结构完全一致或者具有预训练模型；同时子网络数量也没有任何限制，只需要在配置文件中添加即可。
- 支持loss函数通过配置文件任意配置，不仅可以使用某种loss，也可以使用多种loss的组合
- 支持知识蒸馏训练、预测、评估与导出等所有模型相关的环境，方便使用与部署。


通过知识蒸馏，在中英文通用文字识别任务中，不增加任何预测耗时的情况下，可以给模型带来3%以上的精度提升，结合学习率调整策略以及模型结构微调策略，最终提升提升超过5%。


<a name="2"></a>
## 2. 配置文件解析

在知识蒸馏训练的过程中，数据预处理、优化器、学习率、全局的一些属性没有任何变化。模型结构、损失函数、后处理、指标计算等模块的配置文件需要进行微调。

下面以识别与检测的知识蒸馏配置文件为例，对知识蒸馏的训练与配置进行解析。

<a name="21"></a>
### 2.1 识别配置文件解析

配置文件在[ch_PP-OCRv2_rec_distillation.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml)。

<a name="211"></a>
#### 2.1.1 模型结构

知识蒸馏任务中，模型结构配置如下所示。

```yaml
Architecture:
  model_type: &model_type "rec"    # 模型类别，rec、det等，每个子网络的的模型类别都与
  name: DistillationModel          # 结构名称，蒸馏任务中，为DistillationModel，用于构建对应的结构
  algorithm: Distillation          # 算法名称
  Models:                          # 模型，包含子网络的配置信息
    Teacher:                       # 子网络名称，至少需要包含`pretrained`与`freeze_params`信息，其他的参数为子网络的构造参数
      pretrained:                  # 该子网络是否需要加载预训练模型
      freeze_params: false         # 是否需要固定参数
      return_all_feats: true       # 子网络的参数，表示是否需要返回所有的features，如果为False，则只返回最后的输出
      model_type: *model_type      # 模型类别
      algorithm: CRNN              # 子网络的算法名称，该子网络剩余参与均为构造参数，与普通的模型训练配置一致
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
    Student:                       # 另外一个子网络，这里给的是DML的蒸馏示例，两个子网络结构相同，均需要学习参数
      pretrained:                  # 下面的组网参数同上
      freeze_params: false
      return_all_feats: true
      model_type: *model_type
      algorithm: CRNN
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
```

当然，这里如果希望添加更多的子网络进行训练，也可以按照`Student`与`Teacher`的添加方式，在配置文件中添加相应的字段。比如说如果希望有3个模型互相监督，共同训练，那么`Architecture`可以写为如下格式。

```yaml
Architecture:
  model_type: &model_type "rec"
  name: DistillationModel
  algorithm: Distillation
  Models:
    Teacher:
      pretrained:
      freeze_params: false
      return_all_feats: true
      model_type: *model_type
      algorithm: CRNN
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
    Student:
      pretrained:
      freeze_params: false
      return_all_feats: true
      model_type: *model_type
      algorithm: CRNN
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
    Student2:                       # 知识蒸馏任务中引入的新的子网络，其他部分与上述配置相同
      pretrained:
      freeze_params: false
      return_all_feats: true
      model_type: *model_type
      algorithm: CRNN
      Transform:
      Backbone:
        name: MobileNetV1Enhance
        scale: 0.5
      Neck:
        name: SequenceEncoder
        encoder_type: rnn
        hidden_size: 64
      Head:
        name: CTCHead
        mid_channels: 96
        fc_decay: 0.00002
```

最终该模型训练时，包含3个子网络：`Teacher`, `Student`, `Student2`。

蒸馏模型`DistillationModel`类的具体实现代码可以参考[distillation_model.py](../../ppocr/modeling/architectures/distillation_model.py)。

最终模型`forward`输出为一个字典，key为所有的子网络名称，例如这里为`Student`与`Teacher`，value为对应子网络的输出，可以为`Tensor`（只返回该网络的最后一层）和`dict`（也返回了中间的特征信息）。

在识别任务中，为了添加更多损失函数，保证蒸馏方法的可扩展性，将每个子网络的输出保存为`dict`，其中包含子模块输出。以该识别模型为例，每个子网络的输出结果均为`dict`，key包含`backbone_out`,`neck_out`, `head_out`，`value`为对应模块的tensor，最终对于上述配置文件，`DistillationModel`的输出格式如下。

```json
{
  "Teacher": {
    "backbone_out": tensor,
    "neck_out": tensor,
    "head_out": tensor,
  },
  "Student": {
    "backbone_out": tensor,
    "neck_out": tensor,
    "head_out": tensor,
  }
}
```

<a name="212"></a>
#### 2.1.2 损失函数

知识蒸馏任务中，损失函数配置如下所示。

```yaml
Loss:
  name: CombinedLoss                           # 损失函数名称，基于改名称，构建用于损失函数的类
  loss_config_list:                            # 损失函数配置文件列表，为CombinedLoss的必备函数
  - DistillationCTCLoss:                       # 基于蒸馏的CTC损失函数，继承自标准的CTC loss
      weight: 1.0                              # 损失函数的权重，loss_config_list中，每个损失函数的配置都必须包含该字段
      model_name_list: ["Student", "Teacher"]  # 对于蒸馏模型的预测结果，提取这两个子网络的输出，与gt计算CTC loss
      key: head_out                            # 取子网络输出dict中，该key对应的tensor
  - DistillationDMLLoss:                       # 蒸馏的DML损失函数，继承自标准的DMLLoss
      weight: 1.0                              # 权重
      act: "softmax"                           # 激活函数，对输入使用激活函数处理，可以为softmax, sigmoid或者为None，默认为None
      model_name_pairs:                        # 用于计算DML loss的子网络名称对，如果希望计算其他子网络的DML loss，可以在列表下面继续填充
      - ["Student", "Teacher"]
      key: head_out                            # 取子网络输出dict中，该key对应的tensor
  - DistillationDistanceLoss:                  # 蒸馏的距离损失函数
      weight: 1.0                              # 权重
      mode: "l2"                               # 距离计算方法，目前支持l1, l2, smooth_l1
      model_name_pairs:                        # 用于计算distance loss的子网络名称对
      - ["Student", "Teacher"]
      key: backbone_out                        # 取子网络输出dict中，该key对应的tensor
```

上述损失函数中，所有的蒸馏损失函数均继承自标准的损失函数类，主要功能为: 对蒸馏模型的输出进行解析，找到用于计算损失的中间节点(tensor)，再使用标准的损失函数类去计算。

以上述配置为例，最终蒸馏训练的损失函数包含下面3个部分。

- `Student`和`Teacher`的最终输出(`head_out`)与gt的CTC loss，权重为1。在这里因为2个子网络都需要更新参数，因此2者都需要计算与g的loss。
- `Student`和`Teacher`的最终输出(`head_out`)之间的DML loss，权重为1。
- `Student`和`Teacher`的骨干网络输出(`backbone_out`)之间的l2 loss，权重为1。

关于`CombinedLoss`更加具体的实现可以参考: [combined_loss.py](../../ppocr/losses/combined_loss.py#L23)。关于`DistillationCTCLoss`等蒸馏损失函数更加具体的实现可以参考[distillation_loss.py](../../ppocr/losses/distillation_loss.py)。

<a name="213"></a>
#### 2.1.3 后处理

知识蒸馏任务中，后处理配置如下所示。

```yaml
PostProcess:
  name: DistillationCTCLabelDecode       # 蒸馏任务的CTC解码后处理，继承自标准的CTCLabelDecode类
  model_name: ["Student", "Teacher"]     # 对于蒸馏模型的预测结果，提取这两个子网络的输出，进行解码
  key: head_out                          # 取子网络输出dict中，该key对应的tensor
```

以上述配置为例，最终会同时计算`Student`和`Teahcer` 2个子网络的CTC解码输出，返回一个`dict`，`key`为用于处理的子网络名称，`value`为用于处理的子网络列表。

关于`DistillationCTCLabelDecode`更加具体的实现可以参考: [rec_postprocess.py](../../ppocr/postprocess/rec_postprocess.py#L128)

<a name="214"></a>
#### 2.1.4 指标计算

知识蒸馏任务中，指标计算配置如下所示。

```yaml
Metric:
  name: DistillationMetric         # 蒸馏任务的CTC解码后处理，继承自标准的CTCLabelDecode类
  base_metric_name: RecMetric      # 指标计算的基类，对于模型的输出，会基于该类，计算指标
  main_indicator: acc              # 指标的名称
  key: "Student"                   # 选取该子网络的 main_indicator 作为作为保存保存best model的判断标准
```

以上述配置为例，最终会使用`Student`子网络的acc指标作为保存best model的判断指标，同时，日志中也会打印出所有子网络的acc指标。

关于`DistillationMetric`更加具体的实现可以参考: [distillation_metric.py](../../ppocr/metrics/distillation_metric.py#L24)。

<a name="215"></a>
#### 2.1.5 蒸馏模型微调

对蒸馏得到的识别蒸馏进行微调有2种方式。

（1）基于知识蒸馏的微调：这种情况比较简单，下载预训练模型，在[ch_PP-OCRv2_rec_distillation.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml)中配置好预训练模型路径以及自己的数据路径，即可进行模型微调训练。

（2）微调时不使用知识蒸馏：这种情况，需要首先将预训练模型中的学生模型参数提取出来，具体步骤如下。

* 首先下载预训练模型并解压。
```shell
# 下面预训练模型并解压
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar
tar -xf ch_PP-OCRv2_rec_train.tar
```

* 然后使用python，对其中的学生模型参数进行提取

```python
import paddle
# 加载预训练模型
all_params = paddle.load("ch_PP-OCRv2_rec_train/best_accuracy.pdparams")
# 查看权重参数的keys
print(all_params.keys())
# 学生模型的权重提取
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# 查看学生模型权重参数的keys
print(s_params.keys())
# 保存
paddle.save(s_params, "ch_PP-OCRv2_rec_train/student.pdparams")
```

转化完成之后，使用[ch_PP-OCRv2_rec.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec.yml)，修改预训练模型的路径（为导出的`student.pdparams`模型路径）以及自己的数据路径，即可进行模型微调。

<a name="22"></a>
### 2.2 检测配置文件解析

检测模型蒸馏的配置文件在PaddleOCR/configs/det/ch_PP-OCRv3/目录下，包含两个个蒸馏配置文件：
- ch_PP-OCRv3_det_cml.yml，采用cml蒸馏，采用一个大模型蒸馏两个小模型，且两个小模型互相学习的方法
- ch_PP-OCRv3_det_dml.yml，采用DML的蒸馏，两个Student模型互蒸馏的方法

<a name="221"></a>
#### 2.2.1 模型结构

知识蒸馏任务中，模型结构配置如下所示：

```
Architecture:
  name: DistillationModel          # 结构名称，蒸馏任务中，为DistillationModel，用于构建对应的结构
  algorithm: Distillation          # 算法名称
  Models:                          # 模型，包含子网络的配置信息
    Student:                       # 子网络名称，至少需要包含`pretrained`与`freeze_params`信息，其他的参数为子网络的构造参数
      freeze_params: false         # 是否需要固定参数
      return_all_feats: false      # 子网络的参数，表示是否需要返回所有的features，如果为False，则只返回最后的输出
      model_type: det
      algorithm: DB
      Backbone:
        name: ResNet
        in_channels: 3
        layers: 50
      Neck:
        name: LKPAN
        out_channels: 256
      Head:
        name: DBHead
        kernel_list: [7,2,2]
        k: 50
    Teacher:                      # 另外一个子网络，这里给的是DML蒸馏示例，
      freeze_params: true  
      return_all_feats: false
      model_type: det
      algorithm: DB
      Transform:
      Backbone:
        name: ResNet
        in_channels: 3
        layers: 50
      Neck:
        name: LKPAN
        out_channels: 256
      Head:
        name: DBHead
        kernel_list: [7,2,2]
        k: 50

```

如果是采用DML，即两个小模型互相学习的方法，上述配置文件里的Teacher网络结构需要设置为Student模型一样的配置，具体参考配置文件[ch_PP-OCRv3_det_dml.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_dml.yml)。

下面介绍[ch_PP-OCRv3_det_cml.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/configs/det/ch_PP-OCRv3/ch_PP-OCRv3_det_cml.yml)的配置文件参数：

```
Architecture:
  name: DistillationModel  
  algorithm: Distillation
  model_type: det
  Models:
    Teacher:                         # CML蒸馏的Teacher模型配置
      pretrained: ./pretrain_models/ch_ppocr_server_v2.0_det_train/best_accuracy
      freeze_params: true            # Teacher 不训练
      return_all_feats: false
      model_type: det
      algorithm: DB
      Transform:
      Backbone:
        name: ResNet
        in_channels: 3
        layers: 50
      Neck:
        name: LKPAN
        out_channels: 256
      Head:
        name: DBHead
        kernel_list: [7,2,2]
        k: 50
    Student:                         # CML蒸馏的Student模型配置
      pretrained: ./pretrain_models/MobileNetV3_large_x0_5_pretrained  
      freeze_params: false
      return_all_feats: false
      model_type: det
      algorithm: DB
      Backbone:
        name: MobileNetV3
        scale: 0.5
        model_name: large
        disable_se: true
      Neck:
        name: RSEFPN
        out_channels: 96
        shortcut: True
      Head:
        name: DBHead
        k: 50
    Student2:                          # CML蒸馏的Student2模型配置
      pretrained: ./pretrain_models/MobileNetV3_large_x0_5_pretrained  
      freeze_params: false
      return_all_feats: false
      model_type: det
      algorithm: DB
      Transform:
      Backbone:
        name: MobileNetV3
        scale: 0.5
        model_name: large
        disable_se: true
      Neck:
        name: RSEFPN
        out_channels: 96
        shortcut: True
      Head:
        name: DBHead
        k: 50

```


蒸馏模型`DistillationModel`类的具体实现代码可以参考[distillation_model.py](../../ppocr/modeling/architectures/distillation_model.py)。

最终模型`forward`输出为一个字典，key为所有的子网络名称，例如这里为`Student`与`Teacher`，value为对应子网络的输出，可以为`Tensor`（只返回该网络的最后一层）和`dict`（也返回了中间的特征信息）。

在蒸馏任务中，为了方便添加蒸馏损失函数，每个网络的输出保存为`dict`，其中包含子模块输出。每个子网络的输出结果均为`dict`，key包含`backbone_out`,`neck_out`, `head_out`，`value`为对应模块的tensor，最终对于上述配置文件，`DistillationModel`的输出格式如下。

```json
{
  "Teacher": {
    "backbone_out": tensor,
    "neck_out": tensor,
    "head_out": tensor,
  },
  "Student": {
    "backbone_out": tensor,
    "neck_out": tensor,
    "head_out": tensor,
  }
}
```

<a name="222"></a>
#### 2.2.2 损失函数

检测ch_PP-OCRv3_det_cml.yml蒸馏损失函数配置如下所示。相比较于ch_PP-OCRv3_det_distill.yml的损失函数配置，cml蒸馏的损失函数配置做了3个改动：
```yaml
Loss:
  name: CombinedLoss
  loss_config_list:
  - DistillationDilaDBLoss:
      weight: 1.0
      model_name_pairs:
      - ["Student", "Teacher"]
      - ["Student2", "Teacher"]                  # 改动1，计算两个Student和Teacher的损失
      key: maps
      balance_loss: true
      main_loss_type: DiceLoss
      alpha: 5
      beta: 10
      ohem_ratio: 3
  - DistillationDMLLoss:                         # 改动2，增加计算两个Student之间的损失
      model_name_pairs:
      - ["Student", "Student2"]
      maps_name: "thrink_maps"
      weight: 1.0
      # act: None
      key: maps
  - DistillationDBLoss:
      weight: 1.0
      model_name_list: ["Student", "Student2"]   # 改动3，计算两个Student和GT之间的损失
      balance_loss: true
      main_loss_type: DiceLoss
      alpha: 5
      beta: 10
      ohem_ratio: 3

```

关于`DistillationDilaDBLoss`更加具体的实现可以参考: [distillation_loss.py](https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.4/ppocr/losses/distillation_loss.py#L185)。关于`DistillationDBLoss`等蒸馏损失函数更加具体的实现可以参考[distillation_loss.py](https://github.com/PaddlePaddle/PaddleOCR/blob/04c44974b13163450dfb6bd2c327863f8a194b3c/ppocr/losses/distillation_loss.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L148)。

<a name="223"></a>
#### 2.2.3 后处理

知识蒸馏任务中，检测蒸馏后处理配置如下所示。

```yaml
PostProcess:
  name: DistillationDBPostProcess                  # DB检测蒸馏任务的CTC解码后处理，继承自标准的DBPostProcess类
  model_name: ["Student", "Student2", "Teacher"]   # 对于蒸馏模型的预测结果，提取多个子网络的输出，进行解码，不需要后处理的网络可以不在model_name中设置
  thresh: 0.3
  box_thresh: 0.6
  max_candidates: 1000
  unclip_ratio: 1.5
```

以上述配置为例，最终会同时计算`Student`，`Student2`和`Teacher` 3个子网络的输出做后处理计算。同时，由于有多个输入，后处理返回的输出也有多个，

关于`DistillationDBPostProcess`更加具体的实现可以参考: [db_postprocess.py](../../ppocr/postprocess/db_postprocess.py#L195)

<a name="224"></a>
#### 2.2.4 蒸馏指标计算

知识蒸馏任务中，检测蒸馏指标计算配置如下所示。

```yaml
Metric:
  name: DistillationMetric
  base_metric_name: DetMetric
  main_indicator: hmean
  key: "Student"
```

由于蒸馏需要包含多个网络，甚至多个Student网络，在计算指标的时候只需要计算一个Student网络的指标即可，`key`字段设置为`Student`则表示只计算`Student`网络的精度。

<a name="225"></a>
#### 2.2.5 检测蒸馏模型finetune

PP-OCRv3检测蒸馏有两种方式：
- 采用ch_PP-OCRv3_det_cml.yml，采用cml蒸馏，同样Teacher模型设置为PaddleOCR提供的模型或者您训练好的大模型
- 采用ch_PP-OCRv3_det_dml.yml，采用DML的蒸馏，两个Student模型互蒸馏的方法，在PaddleOCR采用的数据集上相比单独训练Student模型有1%-2%的提升。

在具体fine-tune时，需要在网络结构的`pretrained`参数中设置要加载的预训练模型。

在精度提升方面，cml的精度>dml的精度蒸馏方法的精度。当数据量不足或者Teacher模型精度与Student精度相差不大的时候，这个结论或许会改变。


另外，由于PaddleOCR提供的蒸馏预训练模型包含了多个模型的参数，如果您希望提取Student模型的参数，可以参考如下代码：
```
# 下载蒸馏训练模型的参数
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv3_det_distill_train.tar
```

```python
import paddle
# 加载预训练模型
all_params = paddle.load("ch_PP-OCRv3_det_distill_train/best_accuracy.pdparams")
# 查看权重参数的keys
print(all_params.keys())
# 学生模型的权重提取
s_params = {key[len("Student."):]: all_params[key] for key in all_params if "Student." in key}
# 查看学生模型权重参数的keys
print(s_params.keys())
# 保存
paddle.save(s_params, "ch_PP-OCRv3_det_distill_train/student.pdparams")
```

最终`Student`模型的参数将会保存在`ch_PP-OCRv3_det_distill_train/student.pdparams`中，用于模型的fine-tune。
