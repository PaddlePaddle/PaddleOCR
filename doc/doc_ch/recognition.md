## 文字识别

### 数据准备


PaddleOCR 支持两种数据格式: `lmdb` 用于训练公开数据，调试算法; `通用数据` 训练自己的数据:

请按如下步骤设置数据集：

训练数据的默认存储路径是 `PaddleOCR/train_data`,如果您的磁盘上已有数据集，只需创建软链接至数据集目录：

```
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/dataset
```


* 数据下载

若您本地没有数据集，可以在官网下载 [icdar2015](http://rrc.cvc.uab.es/?ch=4&com=downloads) 数据，用于快速验证。也可以参考[DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)，下载 benchmark 所需的lmdb格式数据集。

如果希望复现SRN的论文指标，需要下载离线[增广数据](https://pan.baidu.com/s/1-HSZ-ZVdqBF2HaBZ5pRAKA),提取码: y3ry。增广数据是由MJSynth和SynthText做旋转和扰动得到的。数据下载完成后请解压到 {your_path}/PaddleOCR/train_data/data_lmdb_release/training/ 路径下。

* 使用自己数据集：

若您希望使用自己的数据进行训练，请参考下文组织您的数据。
- 训练集

首先请将训练图片放入同一个文件夹（train_images），并用一个txt文件（rec_gt_train.txt）记录图片路径和标签。

**注意：** 默认请将图片路径和图片标签用 \t 分割，如用其他方式分割将造成训练报错

```
" 图像文件名                 图像标注信息 "

train_data/train_0001.jpg   简单可依赖
train_data/train_0002.jpg   用科技让复杂的世界更简单
```
PaddleOCR 提供了一份用于训练 icdar2015 数据集的标签文件，通过以下方式下载：

```
# 训练集标签
wget -P ./train_data/ic15_data  https://paddleocr.bj.bcebos.com/dataset/rec_gt_train.txt
# 测试集标签
wget -P ./train_data/ic15_data  https://paddleocr.bj.bcebos.com/dataset/rec_gt_test.txt
```

最终训练集应有如下文件结构：
```
|-train_data
    |-ic15_data
        |- rec_gt_train.txt
        |- train
            |- word_001.png
            |- word_002.jpg
            |- word_003.jpg
            | ...
```

- 测试集

同训练集类似，测试集也需要提供一个包含所有图片的文件夹（test）和一个rec_gt_test.txt，测试集的结构如下所示：

```
|-train_data
    |-ic15_data
        |- rec_gt_test.txt
        |- test
            |- word_001.jpg
            |- word_002.jpg
            |- word_003.jpg
            | ...
```

- 字典

最后需要提供一个字典（{word_dict_name}.txt），使模型在训练时，可以将所有出现的字符映射为字典的索引。

因此字典需要包含所有希望被正确识别的字符，{word_dict_name}.txt需要写成如下格式，并以 `utf-8` 编码格式保存：

```
l
d
a
d
r
n
```

word_dict.txt 每行有一个单字，将字符与数字索引映射在一起，“and” 将被映射成 [2 5 1]

`ppocr/utils/ppocr_keys_v1.txt` 是一个包含6623个字符的中文字典，
`ppocr/utils/ic15_dict.txt` 是一个包含36个字符的英文字典，
您可以按需使用。

- 自定义字典

如需自定义dic文件，请在 `configs/rec/rec_icdar15_train.yml` 中添加 `character_dict_path` 字段, 指向您的字典路径。
并将 `character_type` 设置为 `ch`。

- 添加空格类别

如果希望支持识别"空格"类别, 请将yml文件中的 `use_space_char` 字段设置为 `true`。

**注意：`use_space_char` 仅在 `character_type=ch` 时生效**


### 启动训练

PaddleOCR提供了训练脚本、评估脚本和预测脚本，本节将以 CRNN 识别模型为例：

首先下载pretrain model，您可以下载训练好的模型在 icdar2015 数据上进行finetune

```
cd PaddleOCR/
# 下载MobileNetV3的预训练模型
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/rec_mv3_none_bilstm_ctc.tar
# 解压模型参数
cd pretrain_models
tar -xf rec_mv3_none_bilstm_ctc.tar && rm -rf rec_mv3_none_bilstm_ctc.tar
```

开始训练:

*如果您安装的是cpu版本，请将配置文件中的 `use_gpu` 字段修改为false*

```
# 设置PYTHONPATH路径
export PYTHONPATH=$PYTHONPATH:.
# GPU训练 支持单卡，多卡训练，通过CUDA_VISIBLE_DEVICES指定卡号
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 训练icdar15英文数据
python3 tools/train.py -c configs/rec/rec_icdar15_train.yml
```

- 数据增强

PaddleOCR提供了多种数据增强方式，如果您希望在训练时加入扰动，请在配置文件中设置 `distort: true`。

默认的扰动方式有：颜色空间转换(cvtColor)、模糊(blur)、抖动(jitter)、噪声(Gasuss noise)、随机切割(random crop)、透视(perspective)、颜色反转(reverse)。

训练过程中每种扰动方式以50%的概率被选择，具体代码实现请参考：[img_tools.py](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/data/rec/img_tools.py)

*由于OpenCV的兼容性问题，扰动操作暂时只支持GPU*

- 训练

PaddleOCR支持训练和评估交替进行, 可以在 `configs/rec/rec_icdar15_train.yml` 中修改 `eval_batch_step` 设置评估频率，默认每500个iter评估一次。评估过程中默认将最佳acc模型，保存为 `output/rec_CRNN/best_accuracy` 。

如果验证集很大，测试将会比较耗时，建议减少评估次数，或训练完再进行评估。

**提示：** 可通过 -c 参数选择 `configs/rec/` 路径下的多种模型配置进行训练，PaddleOCR支持的识别算法有：


| 配置文件 |  算法名称 |   backbone |   trans   |   seq      |     pred     |
| :--------: |  :-------:   | :-------:  |   :-------:   |   :-----:   |  :-----:   |
| rec_chinese_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  |
| rec_icdar15_train.yml |  CRNN |   Mobilenet_v3 large 0.5 |  None   |  BiLSTM |  ctc  |
| rec_mv3_none_bilstm_ctc.yml |  CRNN |   Mobilenet_v3 large 0.5 |  None   |  BiLSTM |  ctc  |
| rec_mv3_none_none_ctc.yml |  Rosetta |   Mobilenet_v3 large 0.5 |  None   |  None |  ctc  |
| rec_mv3_tps_bilstm_ctc.yml |  STARNet |   Mobilenet_v3 large 0.5 |  tps   |  BiLSTM |  ctc  |
| rec_mv3_tps_bilstm_attn.yml |  RARE |   Mobilenet_v3 large 0.5 |  tps   |  BiLSTM |  attention  |
| rec_r34_vd_none_bilstm_ctc.yml |  CRNN |   Resnet34_vd |  None   |  BiLSTM |  ctc  |
| rec_r34_vd_none_none_ctc.yml |  Rosetta |   Resnet34_vd |  None   |  None |  ctc  |
| rec_r34_vd_tps_bilstm_attn.yml | RARE | Resnet34_vd | tps | BiLSTM | attention |
| rec_r34_vd_tps_bilstm_ctc.yml | STARNet | Resnet34_vd | tps | BiLSTM | ctc |
| rec_r50fpn_vd_none_srn.yml | SRN | Resnet50_fpn_vd | None | rnn | srn |

训练中文数据，推荐使用`rec_chinese_lite_train.yml`，如您希望尝试其他算法在中文数据集上的效果，请参考下列说明修改配置文件：

以 `rec_mv3_none_none_ctc.yml` 为例：
```
Global:
  ...
  # 修改 image_shape 以适应长文本
  image_shape: [3, 32, 320]
  ...
  # 修改字符类型
  character_type: ch
  # 添加自定义字典，如修改字典请将路径指向新字典
  character_dict_path: ./ppocr/utils/ppocr_keys_v1.txt
  # 训练时添加数据增强
  distort: true
  # 识别空格
  use_space_char: true
  ...
  # 修改reader类型
  reader_yml: ./configs/rec/rec_chinese_reader.yml
  ...

...

Optimizer:
  ...
  # 添加学习率衰减策略
  decay:
    function: cosine_decay
    # 每个 epoch 包含 iter 数
    step_each_epoch: 20
    # 总共训练epoch数
    total_epoch: 1000
```
**注意，预测/评估时的配置文件请务必与训练一致。**



### 评估

评估数据集可以通过 `configs/rec/rec_icdar15_reader.yml`  修改EvalReader中的 `label_file_path` 设置。

*注意* 评估时必须确保配置文件中 infer_img 字段为空
```
export CUDA_VISIBLE_DEVICES=0
# GPU 评估， Global.checkpoints 为待测权重
python3 tools/eval.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints={path/to/weights}/best_accuracy
```

### 预测

* 训练引擎的预测

使用 PaddleOCR 训练好的模型，可以通过以下脚本进行快速预测。

默认预测图片存储在 `infer_img` 里，通过 `-o Global.checkpoints` 指定权重：

```
# 预测英文结果
python3 tools/infer_rec.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

预测图片：

![](../imgs_words/en/word_1.png)

得到输入图像的预测结果：

```
infer_img: doc/imgs_words/en/word_1.png
     index: [19 24 18 23 29]
     word : joint
```

预测使用的配置文件必须与训练一致，如您通过 `python3 tools/train.py -c configs/rec/rec_chinese_lite_train.yml` 完成了中文模型的训练，
您可以使用如下命令进行中文模型预测。

```
# 预测中文结果
python3 tools/infer_rec.py -c configs/rec/rec_chinese_lite_train.yml -o Global.checkpoints={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/ch/word_1.jpg
```

预测图片：

![](../imgs_words/ch/word_1.jpg)

得到输入图像的预测结果：

```
infer_img: doc/imgs_words/ch/word_1.jpg
     index: [2092  177  312 2503]
     word : 韩国小馆
```
