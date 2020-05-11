# 文字检测

本节以icdar15数据集为例，介绍PaddleOCR中检测模型的使用方式。

## 3.1 数据准备
icdar2015数据集可以从[官网](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载到，首次下载需注册。

将下载到的数据集解压到工作目录下，假设解压在/PaddleOCR/train_data/ 下。另外，PaddleOCR将零散的标注文件整理成单独的标注文件
，您可以通过wget的方式进行下载。
```
wget -P /PaddleOCR/train_data/  训练标注文件链接
wget -P /PaddleOCR/train_data/  测试标注文件链接
```

解压数据集和下载标注文件后，/PaddleOCR/train_data/ 有两个文件夹和两个文件，分别是：
```
/PaddleOCR/train_data/  
  └─ icdar_c4_train_imgs/         icdar数据集的训练数据
  └─ ch4_test_images/             icdar数据集的测试数据
  └─ train_icdar2015_label.txt    icdar数据集的训练标注
  └─ test_icdar2015_label.txt     icdar数据集的测试标注
```

提供的标注文件格式为：
```
" 图像文件名                    json.dumps编码的图像标注信息"
ch4_test_images/img_61.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]], ...}]
```
json.dumps编码前的图像标注信息是包含多个字典的list，字典中的points表示文本框的位置，如果您想在其他数据集上训练PaddleOCR,
可以按照上述形式构建标注文件。


## 3.2 快速启动训练

首先下载pretrain model，目前支持两种backbone，分别是MobileNetV3、ResNet50，您可以根据需求使用PaddleClas中的模型更换
backbone。
```
# 下载MobileNetV3的预训练模型
wget -P /PaddleOCR/pretrained_model/ 模型链接
# 下载ResNet50的预训练模型
wget -P /PaddleOCR/pretrained_model/ 模型链接
```

**启动训练**
```
cd PaddleOCR/
python3 tools/train.py -c configs/det/det_db_mv3.yml
```

上述指令中，通过-c 选择训练使用configs/det/det_db_mv3.yml配置文件。
有关配置文件的详细解释，请参考[链接]()。

您也可以通过-o参数在不需要修改yml文件的情况下，改变训练的参数，比如，调整训练的学习率为0.0001
```
python3 tools/train.py -c configs/det/det_db_mv3.yml -o Optimizer.base_lr=0.0001
```


## 3.3 指标评估

PaddleOCR计算三个OCR检测相关的指标，分别是：Precision、Recall、Hmean。

运行如下代码，根据配置文件det_db_mv3.yml中save_res_path指定的测试集检测结果文件，计算评估指标。

```
python3 tools/eval.py -c configs/det/det_db_mv3.yml  -o checkpoints ./output/best_accuracy
```

## 3.4 测试检测效果
