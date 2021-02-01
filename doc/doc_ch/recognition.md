## 文字识别


- [一、数据准备](#数据准备)
    - [数据下载](#数据下载)
    - [自定义数据集](#自定义数据集)  
    - [字典](#字典)  
    - [支持空格](#支持空格)

- [二、启动训练](#启动训练)
    - [1. 数据增强](#数据增强)
    - [2. 训练](#训练)
    - [3. 小语种](#小语种)

- [三、评估](#评估)

- [四、预测](#预测)
    - [1. 训练引擎预测](#训练引擎预测)


<a name="数据准备"></a>
### 数据准备


PaddleOCR 支持两种数据格式: `lmdb` 用于训练公开数据，调试算法; `通用数据` 训练自己的数据:

请按如下步骤设置数据集：

训练数据的默认存储路径是 `PaddleOCR/train_data`,如果您的磁盘上已有数据集，只需创建软链接至数据集目录：

```
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/dataset
```

<a name="数据下载"></a>
* 数据下载

若您本地没有数据集，可以在官网下载 [icdar2015](http://rrc.cvc.uab.es/?ch=4&com=downloads) 数据，用于快速验证。也可以参考[DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)，下载 benchmark 所需的lmdb格式数据集。
如果希望复现SRN的论文指标，需要下载离线[增广数据](https://pan.baidu.com/s/1-HSZ-ZVdqBF2HaBZ5pRAKA),提取码: y3ry。增广数据是由MJSynth和SynthText做旋转和扰动得到的。数据下载完成后请解压到 {your_path}/PaddleOCR/train_data/data_lmdb_release/training/ 路径下。

<a name="自定义数据集"></a>
* 使用自己数据集

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

PaddleOCR 也提供了数据格式转换脚本，可以将官网 label 转换支持的数据格式。 数据转换工具在 `ppocr/utils/gen_label.py`, 这里以训练集为例：

```
# 将官网下载的标签文件转换为 rec_gt_label.txt
python gen_label.py --mode="rec" --input_path="{path/of/origin/label}" --output_label="rec_gt_label.txt"
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
<a name="字典"></a>
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

`ppocr/utils/ppocr_keys_v1.txt` 是一个包含6623个字符的中文字典

`ppocr/utils/ic15_dict.txt` 是一个包含36个字符的英文字典

`ppocr/utils/dict/french_dict.txt` 是一个包含118个字符的法文字典

`ppocr/utils/dict/japan_dict.txt` 是一个包含4399个字符的日文字典

`ppocr/utils/dict/korean_dict.txt` 是一个包含3636个字符的韩文字典

`ppocr/utils/dict/german_dict.txt` 是一个包含131个字符的德文字典

`ppocr/utils/dict/en_dict.txt` 是一个包含63个字符的英文字典


您可以按需使用。

目前的多语言模型仍处在demo阶段，会持续优化模型并补充语种，**非常欢迎您为我们提供其他语言的字典和字体**，
如您愿意可将字典文件提交至 [dict](../../ppocr/utils/dict) 将语料文件提交至[corpus](../../ppocr/utils/corpus)，我们会在Repo中感谢您。

- 自定义字典

如需自定义dic文件，请在 `configs/rec/rec_icdar15_train.yml` 中添加 `character_dict_path` 字段, 指向您的字典路径。
并将 `character_type` 设置为 `ch`。

<a name="支持空格"></a>
- 添加空格类别

如果希望支持识别"空格"类别, 请将yml文件中的 `use_space_char` 字段设置为 `True`。


<a name="启动训练"></a>
### 启动训练

PaddleOCR提供了训练脚本、评估脚本和预测脚本，本节将以 CRNN 识别模型为例：

首先下载pretrain model，您可以下载训练好的模型在 icdar2015 数据上进行finetune

```
cd PaddleOCR/
# 下载MobileNetV3的预训练模型
wget -P ./pretrain_models/ https://paddleocr.bj.bcebos.com/dygraph_v2.0/en/rec_mv3_none_bilstm_ctc_v2.0_train.tar
# 解压模型参数
cd pretrain_models
tar -xf rec_mv3_none_bilstm_ctc_v2.0_train.tar && rm -rf rec_mv3_none_bilstm_ctc_v2.0_train.tar
```

开始训练:

*如果您安装的是cpu版本，请将配置文件中的 `use_gpu` 字段修改为false*

```
# GPU训练 支持单卡，多卡训练，通过--gpus参数指定卡号
# 训练icdar15英文数据 训练日志会自动保存为 "{save_model_dir}" 下的train.log
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/rec_icdar15_train.yml
```
<a name="数据增强"></a>
- 数据增强

PaddleOCR提供了多种数据增强方式，如果您希望在训练时加入扰动，请在配置文件中设置 `distort: true`。

默认的扰动方式有：颜色空间转换(cvtColor)、模糊(blur)、抖动(jitter)、噪声(Gasuss noise)、随机切割(random crop)、透视(perspective)、颜色反转(reverse)。

训练过程中每种扰动方式以50%的概率被选择，具体代码实现请参考：[img_tools.py](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/data/rec/img_tools.py)

*由于OpenCV的兼容性问题，扰动操作暂时只支持Linux*

<a name="训练"></a>
- 训练

PaddleOCR支持训练和评估交替进行, 可以在 `configs/rec/rec_icdar15_train.yml` 中修改 `eval_batch_step` 设置评估频率，默认每500个iter评估一次。评估过程中默认将最佳acc模型，保存为 `output/rec_CRNN/best_accuracy` 。

如果验证集很大，测试将会比较耗时，建议减少评估次数，或训练完再进行评估。

**提示：** 可通过 -c 参数选择 `configs/rec/` 路径下的多种模型配置进行训练，PaddleOCR支持的识别算法有：


| 配置文件 |  算法名称 |   backbone |   trans   |   seq      |     pred     |
| :--------: |  :-------:   | :-------:  |   :-------:   |   :-----:   |  :-----:   |
| [rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml) |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  |
| [rec_chinese_common_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_common_train_v2.0.yml) |  CRNN | ResNet34_vd |  None   |  BiLSTM |  ctc  |
| rec_icdar15_train.yml |  CRNN |   Mobilenet_v3 large 0.5 |  None   |  BiLSTM |  ctc  |
| rec_mv3_none_bilstm_ctc.yml |  CRNN |   Mobilenet_v3 large 0.5 |  None   |  BiLSTM |  ctc  |
| rec_mv3_none_none_ctc.yml |  Rosetta |   Mobilenet_v3 large 0.5 |  None   |  None |  ctc  |
| rec_r34_vd_none_bilstm_ctc.yml |  CRNN |   Resnet34_vd |  None   |  BiLSTM |  ctc  |
| rec_r34_vd_none_none_ctc.yml |  Rosetta |   Resnet34_vd |  None   |  None |  ctc  |
| rec_r50fpn_vd_none_srn.yml    | SRN | Resnet50_fpn_vd    | None    | rnn | srn |

训练中文数据，推荐使用[rec_chinese_lite_train_v2.0.yml](../../configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml)，如您希望尝试其他算法在中文数据集上的效果，请参考下列说明修改配置文件：

以 `rec_chinese_lite_train_v2.0.yml` 为例：
```
Global:
  ...
  # 添加自定义字典，如修改字典请将路径指向新字典
  character_dict_path: ppocr/utils/ppocr_keys_v1.txt
  # 修改字符类型
  character_type: ch
  ...
  # 识别空格
  use_space_char: True


Optimizer:
  ...
  # 添加学习率衰减策略
  lr:
    name: Cosine
    learning_rate: 0.001
  ...

...

Train:
  dataset:
    # 数据集格式，支持LMDBDateSet以及SimpleDataSet
    name: SimpleDataSet
    # 数据集路径
    data_dir: ./train_data/
    # 训练集标签文件
    label_file_list: ["./train_data/train_list.txt"]
    transforms:
      ...
      - RecResizeImg:
          # 修改 image_shape 以适应长文本
          image_shape: [3, 32, 320]
      ...
  loader:
    ...
    # 单卡训练的batch_size
    batch_size_per_card: 256
    ...

Eval:
  dataset:
    # 数据集格式，支持LMDBDateSet以及SimpleDataSet
    name: SimpleDataSet
    # 数据集路径
    data_dir: ./train_data
    # 验证集标签文件
    label_file_list: ["./train_data/val_list.txt"]
    transforms:
      ...
      - RecResizeImg:
          # 修改 image_shape 以适应长文本
          image_shape: [3, 32, 320]
      ...
  loader:
    # 单卡验证的batch_size
    batch_size_per_card: 256
    ...
```
**注意，预测/评估时的配置文件请务必与训练一致。**

<a name="小语种"></a>
- 小语种

PaddleOCR目前已支持26种（除中文外）语种识别，`configs/rec/multi_languages` 路径下提供了一个多语言的配置文件模版: [rec_multi_language_lite_train.yml](../../configs/rec/multi_language/rec_multi_language_lite_train.yml)。

您有两种方式创建所需的配置文件：

1. 通过脚本自动生成

[generate_multi_language_configs.py](../../configs/rec/multi_language/generate_multi_language_configs.py) 可以帮助您生成多语言模型的配置文件

- 以意大利语为例，如果您的数据是按如下格式准备的：
    ```
    |-train_data
        |- it_train.txt # 训练集标签
        |- it_val.txt # 验证集标签
        |- data
            |- word_001.jpg
            |- word_002.jpg
            |- word_003.jpg
            | ...
    ```

    可以使用默认参数，生成配置文件：

    ```bash
    # 该代码需要在指定目录运行
    cd PaddleOCR/configs/rec/multi_language/
    # 通过-l或者--language参数设置需要生成的语种的配置文件，该命令会将默认参数写入配置文件
    python3 generate_multi_language_configs.py -l it
    ```

- 如果您的数据放置在其他位置，或希望使用自己的字典，可以通过指定相关参数来生成配置文件:

    ```bash
    # -l或者--language字段是必须的
    # --train修改训练集，--val修改验证集，--data_dir修改数据集目录，--dict修改字典路径， -o修改对应默认参数
    cd PaddleOCR/configs/rec/multi_language/
    python3 generate_multi_language_configs.py -l it \  # 语种
    --train {path/of/train_label.txt} \ # 训练标签文件的路径
    --val {path/of/val_label.txt} \     # 验证集标签文件的路径
    --data_dir {train_data/path} \      # 训练数据的根目录
    --dict {path/of/dict} \             # 字典文件路径
    -o Global.use_gpu=False             # 是否使用gpu
    ...

    ```

2. 手动修改配置文件

   您也可以手动修改模版中的以下几个字段:

   ```
    Global:
      use_gpu: True
      epoch_num: 500
      ...
      character_type: it  # 需要识别的语种
      character_dict_path:  {path/of/dict} # 字典文件所在路径

   Train:
      dataset:
        name: SimpleDataSet
        data_dir: train_data/ # 数据存放根目录
        label_file_list: ["./train_data/train_list.txt"] # 训练集label路径
      ...

   Eval:
      dataset:
        name: SimpleDataSet
        data_dir: train_data/ # 数据存放根目录
        label_file_list: ["./train_data/val_list.txt"] # 验证集label路径
      ...

   ```

目前PaddleOCR支持的多语言算法有：

| 配置文件 |  算法名称 |   backbone |   trans   |   seq      |     pred     |  language | character_type |
| :--------: |  :-------:   | :-------:  |   :-------:   |   :-----:   |  :-----:   | :-----:  | :-----:  |
| rec_chinese_cht_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 中文繁体  | chinese_cht|
| rec_en_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 英语（区分大小写）   | EN |
| rec_french_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 法语 |  french |
| rec_ger_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 德语   | german |
| rec_japan_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 日语  | japan |
| rec_korean_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 韩语  | korean |
| rec_it_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 意大利语  | it |
| rec_xi_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 西班牙语 |  xi |
| rec_pu_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 葡萄牙语   | pu |
| rec_ru_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 俄罗斯语  | ru |
| rec_ar_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 阿拉伯语  | ar |
| rec_hi_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 印地语 |  hi |
| rec_ug_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 维吾尔语  | ug |
| rec_fa_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 波斯语  | fa |
| rec_ur_ite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 乌尔都语  | ur |
| rec_rs_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 塞尔维亚(latin)语 | rs |
| rec_oc_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 欧西坦语   | oc |
| rec_mr_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 马拉地语  | mr |
| rec_ne_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 尼泊尔语  | ne |
| rec_rsc_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 塞尔维亚(cyrillic)语 |  rsc |
| rec_bg_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 保加利亚语  | bg |
| rec_uk_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 乌克兰语  | uk |
| rec_be_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 白俄罗斯语   | be |
| rec_te_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 泰卢固语  | te |
| rec_ka_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 卡纳达语  | ka |
| rec_ta_lite_train.yml |  CRNN |   Mobilenet_v3 small 0.5 |  None   |  BiLSTM |  ctc  | 泰米尔语 |  ta |

多语言模型训练方式与中文模型一致，训练数据集均为100w的合成数据，少量的字体可以在 [百度网盘](https://pan.baidu.com/s/1bS_u207Rm7YbY33wOECKDA) 上下载，提取码：frgi。

如您希望在现有模型效果的基础上调优，请参考下列说明修改配置文件：

以 `rec_french_lite_train` 为例：
```
Global:
  ...
  # 添加自定义字典，如修改字典请将路径指向新字典
  character_dict_path: ./ppocr/utils/dict/french_dict.txt
  ...
  # 识别空格
  use_space_char: True

...

Train:
  dataset:
    # 数据集格式，支持LMDBDateSet以及SimpleDataSet
    name: SimpleDataSet
    # 数据集路径
    data_dir: ./train_data/
    # 训练集标签文件
    label_file_list: ["./train_data/french_train.txt"]
    ...

Eval:
  dataset:
    # 数据集格式，支持LMDBDateSet以及SimpleDataSet
    name: SimpleDataSet
    # 数据集路径
    data_dir: ./train_data
    # 验证集标签文件
    label_file_list: ["./train_data/french_val.txt"]
    ...
```
<a name="评估"></a>
### 评估

评估数据集可以通过 `configs/rec/rec_icdar15_train.yml`  修改Eval中的 `label_file_path` 设置。

```
# GPU 评估， Global.checkpoints 为待测权重
python3 -m paddle.distributed.launch --gpus '0' tools/eval.py -c configs/rec/rec_icdar15_train.yml -o Global.checkpoints={path/to/weights}/best_accuracy
```

<a name="预测"></a>
### 预测

<a name="训练引擎预测"></a>
* 训练引擎的预测

使用 PaddleOCR 训练好的模型，可以通过以下脚本进行快速预测。

默认预测图片存储在 `infer_img` 里，通过 `-o Global.checkpoints` 指定权重：

```
# 预测英文结果
python3 tools/infer_rec.py -c configs/rec/rec_icdar15_train.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.load_static_weights=false Global.infer_img=doc/imgs_words/en/word_1.png
```

预测图片：

![](../imgs_words/en/word_1.png)

得到输入图像的预测结果：

```
infer_img: doc/imgs_words/en/word_1.png
        result: ('joint', 0.9998967)
```

预测使用的配置文件必须与训练一致，如您通过 `python3 tools/train.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml` 完成了中文模型的训练，
您可以使用如下命令进行中文模型预测。

```
# 预测中文结果
python3 tools/infer_rec.py -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0.yml -o Global.pretrained_model={path/to/weights}/best_accuracy Global.load_static_weights=false Global.infer_img=doc/imgs_words/ch/word_1.jpg
```

预测图片：

![](../imgs_words/ch/word_1.jpg)

得到输入图像的预测结果：

```
infer_img: doc/imgs_words/ch/word_1.jpg
        result: ('韩国小馆', 0.997218)
```
