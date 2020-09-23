## 文字角度分类

### 数据准备

请按如下步骤设置数据集：

训练数据的默认存储路径是 `PaddleOCR/train_data/cls`,如果您的磁盘上已有数据集，只需创建软链接至数据集目录：

```
ln -sf <path/to/dataset> <path/to/paddle_ocr>/train_data/cls/dataset
```

请参考下文组织您的数据。
- 训练集

首先请将训练图片放入同一个文件夹（train_images），并用一个txt文件（cls_gt_train.txt）记录图片路径和标签。

**注意：** 默认请将图片路径和图片标签用 `\t` 分割，如用其他方式分割将造成训练报错

0和180分别表示图片的角度为0度和180度

```
" 图像文件名                 图像标注信息 "

train_data/cls/word_001.jpg   0
train_data/cls/word_002.jpg   180
```

最终训练集应有如下文件结构：
```
|-train_data
    |-cls
        |- cls_gt_train.txt
        |- train
            |- word_001.png
            |- word_002.jpg
            |- word_003.jpg
            | ...
```

- 测试集

同训练集类似，测试集也需要提供一个包含所有图片的文件夹（test）和一个cls_gt_test.txt，测试集的结构如下所示：

```
|-train_data
    |-cls
        |- 和一个cls_gt_test.txt
        |- test
            |- word_001.jpg
            |- word_002.jpg
            |- word_003.jpg
            | ...
```

### 启动训练

PaddleOCR提供了训练脚本、评估脚本和预测脚本。

开始训练:

*如果您安装的是cpu版本，请将配置文件中的 `use_gpu` 字段修改为false*

```
# 设置PYTHONPATH路径
export PYTHONPATH=$PYTHONPATH:.
# GPU训练 支持单卡，多卡训练，通过CUDA_VISIBLE_DEVICES指定卡号
export CUDA_VISIBLE_DEVICES=0,1,2,3
# 启动训练
python3 tools/train.py -c configs/cls/cls_mv3.yml
```

- 数据增强

PaddleOCR提供了多种数据增强方式，如果您希望在训练时加入扰动，请在配置文件中设置 `distort: true`。

默认的扰动方式有：颜色空间转换(cvtColor)、模糊(blur)、抖动(jitter)、噪声(Gasuss noise)、随机切割(random crop)、透视(perspective)、颜色反转(reverse),随机数据增强(RandAugment)。

训练过程中除随机数据增强外每种扰动方式以50%的概率被选择，具体代码实现请参考：
[randaugment.py](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/data/cls/randaugment.py)
[img_tools.py](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/data/rec/img_tools.py)

*由于OpenCV的兼容性问题，扰动操作暂时只支持linux*

### 训练

PaddleOCR支持训练和评估交替进行, 可以在 `configs/cls/cls_mv3.yml` 中修改 `eval_batch_step` 设置评估频率，默认每500个iter评估一次。评估过程中默认将最佳acc模型，保存为 `output/cls_mv3/best_accuracy` 。

如果验证集很大，测试将会比较耗时，建议减少评估次数，或训练完再进行评估。

**注意，预测/评估时的配置文件请务必与训练一致。**

### 评估

评估数据集可以通过`configs/cls/cls_reader.yml`  修改EvalReader中的 `label_file_path` 设置。

*注意* 评估时必须确保配置文件中 infer_img 字段为空
```
export CUDA_VISIBLE_DEVICES=0
# GPU 评估， Global.checkpoints 为待测权重
python3 tools/eval.py -c configs/cls/cls_mv3.yml -o Global.checkpoints={path/to/weights}/best_accuracy
```

### 预测

* 训练引擎的预测

使用 PaddleOCR 训练好的模型，可以通过以下脚本进行快速预测。

默认预测图片存储在 `infer_img` 里，通过 `-o Global.checkpoints` 指定权重：

```
# 预测分类结果
python3 tools/infer_cls.py -c configs/cls/cls_mv3.yml -o Global.checkpoints={path/to/weights}/best_accuracy Global.infer_img=doc/imgs_words/en/word_1.png
```

预测图片：

![](../imgs_words/en/word_1.png)

得到输入图像的预测结果：

```
infer_img: doc/imgs_words/en/word_1.png
    scores: [[0.93161047 0.06838956]]
    label: [0]
```
