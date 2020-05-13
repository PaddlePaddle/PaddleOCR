## 文字识别

### 数据准备


PaddleOCR 支持 `lmdb` 和 `通用数据` 两种数据格式，请按如下步骤设置数据集：

训练数据的默认存储路径是 `PaddleOCR/train_data`,如果您的磁盘上已有数据集，只需创建软链接至数据集目录：

```
ln -sf <path/to/dataset> <path/to/paddle_detection>/train_data/dataset
```


* 数据下载

若您本地没有数据集，可以参考[DTRB](https://github.com/clovaai/deep-text-recognition-benchmark#download-lmdb-dataset-for-traininig-and-evaluation-from-here)，下载 benchmark 所需的lmdb格式数据集。也可在官网下载 [icdar2015](http://rrc.cvc.uab.es/?ch=4&com=downloads) 数据，用于快速验证。

* 使用自己数据集

若您希望使用自己的数据进行训练，请参考下文组织您的数据。

- 训练集

首先请将训练图片放入同一个文件夹（train_images），并用一个txt文件（rec_gt_train.txt）记录图片路径和标签。

* 注意： 默认请将图片路径和图片标签用 \t 分割，如用其他方式分割将造成训练报错

```
" 图像文件名                 图像标注信息 "

train_data/train_0001.jpg   简单可依赖
train_data/train_0002.jpg   用科技让复杂的世界更简单
```

最终训练集应有如下文件结构：

|-train_data
    |- rec_gt_train.txt
    |- train_imags
        |- train_001.jpg
        |- train_002.jpg
        |- train_003.jpg
        | ...

- 评估集

同训练集类似，评估集也需要提供一个包含所有图片的文件夹（eval_images）和一个rec_gt_eval.txt，评估集的结构如下所示：

|-train_data
    |- rec_gt_eval.txt
    |- eval_imags
        |- eval_001.jpg
        |- eval_002.jpg
        |- eval_003.jpg
        | ...


- 字典

最后需要提供一个字典（{word_dict_name}.txt），使模型在训练时，可以将所有出现的字符映射为字典的索引。

因此字典需要包含所有希望被正确识别的字符，{word_dict_name}.txt需要写成如下格式：


L
d
a
D
R
n

word_dict.txt 每行有一个单字，将字符与数字索引映射在一起，“and” 将被映射成 [2 5 1]

`ppocr/utils/ppocr_keys_v1.txt` 是一个包含6623个字符的中文字典，
`ppocr/utils/ic15_dict.txt` 是一个包含36个字符的英文字典，
您可以按需使用。如需自定义dic文件，请修改 `configs/rec/rec_icdar15_train.yml` 中的 `character_dict_path` 字段。


### 启动训练

PaddleOCR提供了训练脚本、评估脚本和预测脚本，本节将以RCNN中文识别模型为例：

```
# 设置PYTHONPATH路径
export PYTHONPATH=$PYTHONPATH:.
# GPU训练 支持单卡，多卡训练，通过CUDA_VISIBLE_DEVICES指定卡号
export CUDA_VISIBLE_DEVICES=0,1,2,3
python tools/train.py -c configs/rec/rec_icdar15_train.yml
```

PaddleOCR支持训练和评估交替进行, 可以在 `configs/rec/rec_icdar15_train.yml` 中修改 `eval_batch_step` 设置评估频率，默认每2000个iter评估一次。评估过程中默认将最佳acc模型，保存为 `output/rec/best_accuracy` 。

如果验证集很大，测试将会比较耗时，建议减少评估次数，或训练完再进行评估。

### 评估

评估数据集可以通过 `configs/rec/rec_icdar15_reader.yml`  修改EvalReader中的 `label_file_path` 设置。

```
export CUDA_VISIBLE_DEVICES=0
# GPU 评估， Global.pretrain_weights 为待测权重
python tools/eval.py -c configs/rec/rec_chinese_lite_train.yml -o Global.pretrain_weights={path/to/weights}/best_accuracy
```

### 测试

* 训练引擎的预测

PaddleOCR 提供了训练好的中文模型，可以使用

默认预测图片存储在 `infer_img` 里，通过 Global.pretrain_weights 指定权重：

```
python tools/infer_rec.py -c configs/rec/rec_chinese_lite_train.yml -o Global.pretrain_weights={path/to/weights}/best_accuracy
```

得到输入图像的预测结果：

```
infer_img: infer_img/328_4.jpg
# 字符在字典中的索引
[1863  921   55  155 1863 4209 3344  486  914 1863 4918]
# 预测结果
冷库专用冷冻液/载冷剂
```

得到预测结果后，脚本会自动将权重转换为inference model 并保存在 rec_inference 下：

|-rec_inference
    |- model
    |- params


