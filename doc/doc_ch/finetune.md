# 模型微调

## 1. 模型微调背景与意义

PaddleOCR提供的PP-OCR系列模型在通用场景中性能优异，能够解决绝大多数情况下的检测与识别问题。在垂类场景中，如果希望获取更优的模型效果，可以通过模型微调的方法，进一步提升PP-OCR系列检测与识别模型的精度。

本文主要介绍文本检测与识别模型在模型微调时的一些注意事项，最终希望您在自己的场景中，通过模型微调，可以获取精度更高的文本检测与识别模型。

本文核心要点如下所示。

1. PP-OCR提供的预训练模型有较好的泛化能力
2. 加入少量真实数据（检测任务>=500张, 识别任务>=5000张），会大幅提升垂类场景的检测与识别效果
3. 在模型微调时，加入真实通用场景数据，可以进一步提升模型精度与泛化性能
4. 在图像检测任务中，增大图像的预测尺度，能够进一步提升较小文字区域的检测效果
5. 在模型微调时，需要适当调整超参数（学习率，batch size最为重要），以获得更优的微调效果。

更多详细内容，请参考第2章与第3章。

## 2. 文本检测模型微调

### 2.1 数据选择

* 数据量：建议至少准备500张的文本检测数据集用于模型微调。

* 数据标注：单行文本标注格式，建议标注的检测框与实际语义内容一致。如在火车票场景中，姓氏与名字可能离得较远，但是它们在语义上属于同一个检测字段，这里也需要将整个姓名标注为1个检测框。

### 2.2 模型选择

建议选择PP-OCRv2模型（配置文件：[ch_PP-OCRv2_det_student.yml](../../configs/det/ch_PP-OCRv2/ch_PP-OCRv2_det_student.yml)，预训练模型：[ch_PP-OCRv2_det_distill_train.tar](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar)）进行微调，其精度与泛化性能是目前提供的最优预训练模型。

更多PP-OCR系列模型，请参考[PaddleOCR 首页说明文档](../../README_ch.md)。

注意：在使用上述预训练模型的时候，由于保存的模型中包含教师模型，因此需要将其中的学生模型单独提取出来，再加载学生模型即可进行模型微调。

```python
import paddle
# 加载完整的检测预训练模型
a = paddle.load("ch_PP-OCRv2_det_distill_train/best_accuracy.pdparams")
# 提取学生模型的参数
b = {k[len("student_model."):]: a[k] for k in a if "student_model." in k}
# 保存模型，用于后续模型微调
paddle.save(b, "ch_PP-OCRv2_det_student.pdparams")
```


### 2.3 训练超参选择

在模型微调的时候，最重要的超参就是预训练模型路径`pretrained_model`, 学习率`learning_rate`与`batch_size`，部分配置文件如下所示。

```yaml
Global:
  pretrained_model: ./pretrain_models/student.pdparams # 预训练模型路径
Optimizer:
  lr:
    name: Cosine
    learning_rate: 0.001 # 学习率
    warmup_epoch: 2
  regularizer:
    name: 'L2'
    factor: 0

Train:
  loader:
    shuffle: True
    drop_last: False
    batch_size_per_card: 8  # 单卡batch size
    num_workers: 4
```

上述配置文件中，首先需要将`pretrained_model`字段指定为2.2章节中提取出来的`ch_PP-OCRv2_det_student.pdparams`文件路径。

PaddleOCR提供的配置文件是在8卡训练（相当于总的batch size是`8*8=64`）、且没有加载预训练模型情况下的配置文件，因此您的场景中，学习率与总的batch size需要对应线性调整，例如

* 如果您的场景中是单卡训练，单卡batch_size=8，则总的batch_size=8，建议将学习率调整为`1e-4`左右。
* 如果您的场景中是单卡训练，由于显存限制，只能设置单卡batch_size=4，则总的batch_size=4，建议将学习率调整为`5e-5`左右。

### 2.4 预测超参选择

对训练好的模型导出并进行推理时，可以通过进一步调整预测的图像尺度，来提升小面积文本的检测效果，下面是DBNet推理时的一些超参数，可以通过适当调整，提升效果。

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  det_db_thresh | float | 0.3 | DB输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点 |
|  det_db_box_thresh | float | 0.6 | 检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域 |
|  det_db_unclip_ratio | float | 1.5 | `Vatti clipping`算法的扩张系数，使用该方法对文字区域进行扩张 |
|  max_batch_size | int | 10 | 预测的batch size |
|  use_dilation | bool | False | 是否对分割结果进行膨胀以获取更优检测效果 |
|  det_db_score_mode | str | "fast" | DB的检测结果得分计算方法，支持`fast`和`slow`，`fast`是根据polygon的外接矩形边框内的所有像素计算平均得分，`slow`是根据原始polygon内的所有像素计算平均得分，计算速度相对较慢一些，但是更加准确一些。 |


更多关于推理方法的介绍可以参考[Paddle Inference推理教程](./inference.md)。


## 3. 文本识别模型微调


### 3.1 数据选择

* 数据量：不更换字典的情况下，建议至少准备5000张的文本识别数据集用于模型微调；如果更换了字典（不建议），需要的数量更多。

* 数据分布：建议分布与实测场景尽量一致。如果实测场景包含大量短文本，则训练数据中建议也包含较多短文本，如果实测场景对于空格识别效果要求较高，则训练数据中建议也包含较多带空格的文本内容。


* 通用中英文数据：在训练的时候，可以在训练集中添加通用真实数据（如在不更换字典的微调场景中，建议添加LSVT、RCTW、MTWI等真实数据），进一步提升模型的泛化性能。

### 3.2 模型选择

建议选择PP-OCRv2模型（配置文件：[ch_PP-OCRv2_rec_distillation.yml](../../configs/rec/ch_PP-OCRv2/ch_PP-OCRv2_rec_distillation.yml)，预训练模型：[ch_PP-OCRv2_rec_train.tar](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar)）进行微调，其精度与泛化性能是目前提供的最优预训练模型。

更多PP-OCR系列，模型请参考[PaddleOCR 首页说明文档](../../README_ch.md)。


### 3.3 训练超参选择

与文本检测任务微调相同，在识别模型微调的时候，最重要的超参就是预训练模型路径`pretrained_model`, 学习率`learning_rate`与`batch_size`，部分默认配置文件如下所示。

```yaml
Global:
  pretrained_model:  # 预训练模型路径
Optimizer:
  lr:
    name: Piecewise
    decay_epochs : [700, 800]
    values : [0.001, 0.0001]  # 学习率
    warmup_epoch: 5
  regularizer:
    name: 'L2'
    factor: 0

Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
    - ./train_data/train_list.txt
    ratio_list: [1.0] # 采样比例，默认值是[1.0]
  loader:
    shuffle: True
    drop_last: False
    batch_size_per_card: 128 # 单卡batch size
    num_workers: 8

```


上述配置文件中，首先需要将`pretrained_model`字段指定为2.2章节中解压得到的`ch_PP-OCRv2_rec_train/best_accuracy.pdparams`文件路径。

PaddleOCR提供的配置文件是在8卡训练（相当于总的batch size是`8*128=1024`）、且没有加载预训练模型情况下的配置文件，因此您的场景中，学习率与总的batch size需要对应线性调整，例如：

* 如果您的场景中是单卡训练，单卡batch_size=128，则总的batch_size=128，在加载预训练模型的情况下，建议将学习率调整为`[1e-4, 2e-5]`左右（piecewise学习率策略，需设置2个值，下同）。
* 如果您的场景中是单卡训练，因为显存限制，只能设置单卡batch_size=64，则总的batch_size=64，在加载预训练模型的情况下，建议将学习率调整为`[5e-5, 1e-5]`左右。


如果有通用真实场景数据加进来，建议每个epoch中，垂类场景数据与真实场景的数据量保持在1:1左右。

比如：您自己的垂类场景识别数据量为1W，数据标签文件为`vertical.txt`，收集到的通用场景识别数据量为10W，数据标签文件为`general.txt`，


那么，可以设置`label_file_list`和`ratio_list`参数如下所示。每个epoch中，`vertical.txt`中会进行全采样（采样比例为1.0），包含1W条数据；`general.txt`中会按照0.1的采样比例进行采样，包含`10W*0.1=1W`条数据，最终二者的比例为`1:1`。

```yaml
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ./train_data/
    label_file_list:
    - vertical.txt
    - general.txt
    ratio_list: [1.0, 0.1]
```
