# 文字检测

本节以icdar15数据集为例，介绍PaddleOCR中检测模型的训练、评估与测试。

## 数据准备
icdar2015数据集可以从[官网](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载到，首次下载需注册。

将下载到的数据集解压到工作目录下，假设解压在 PaddleOCR/train_data/ 下。另外，PaddleOCR将零散的标注文件整理成单独的标注文件
，您可以通过wget的方式进行下载。
```
# 在PaddleOCR路径下
cd PaddleOCR/
wget -P ./train_data/  https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt
wget -P ./train_data/  https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt
```

解压数据集和下载标注文件后，PaddleOCR/train_data/ 有两个文件夹和两个文件，分别是：
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
json.dumps编码前的图像标注信息是包含多个字典的list，字典中的 `points` 表示文本框的四个点的坐标(x, y)，从左上角的点开始顺时针排列。
`transcription` 表示当前文本框的文字，在文本检测任务中并不需要这个信息。
如果您想在其他数据集上训练PaddleOCR，可以按照上述形式构建标注文件。


## 快速启动训练

首先下载pretrain model，PaddleOCR的检测模型目前支持两种backbone，分别是MobileNetV3、ResNet50_vd，
您可以根据需求使用[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/master/ppcls/modeling/architectures)中的模型更换backbone。
```
cd PaddleOCR/
# 下载MobileNetV3的预训练模型
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x0_5_pretrained.tar
# 下载ResNet50的预训练模型
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar
```

**启动训练**
```
python3 tools/train.py -c configs/det/det_mv3_db.yml
```

上述指令中，通过-c 选择训练使用configs/det/det_db_mv3.yml配置文件。
有关配置文件的详细解释，请参考[链接](./doc/config.md)。

您也可以通过-o参数在不需要修改yml文件的情况下，改变训练的参数，比如，调整训练的学习率为0.0001
```
python3 tools/train.py -c configs/det/det_mv3_db.yml -o Optimizer.base_lr=0.0001
```

## 指标评估

PaddleOCR计算三个OCR检测相关的指标，分别是：Precision、Recall、Hmean。

运行如下代码，根据配置文件det_db_mv3.yml中save_res_path指定的测试集检测结果文件，计算评估指标。

```
python3 tools/eval.py -c configs/det/det_mv3_db.yml  -o Global.checkpoints="{path/to/weights}/best_accuracy"
```
训练中模型参数默认保存在Global.save_model_dir目录下。在评估指标时，需要设置Global.checkpoints指向保存的参数文件。

比如：
```
python3 tools/eval.py -c configs/det/det_mv3_db.yml  -o Global.checkpoints="./output/det_db/best_accuracy"
```


## 测试检测效果

测试单张图像的检测效果
```
python3 tools/infer_det.py -c config/det/det_mv3_db.yml -o TestReader.single_img_path="./doc/imgs_en/img_10.jpg" Global.checkpoints="./output/det_db/best_accuracy"
```

测试文件夹下所有图像的检测效果
```
python3 tools/infer_det.py -c config/det/det_mv3_db.yml -o TestReader.single_img_path="./doc/imgs_en/" Global.checkpoints="./output/det_db/best_accuracy"
```
