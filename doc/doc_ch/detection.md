# 文字检测

本节以icdar2015数据集为例，介绍PaddleOCR中检测模型的训练、评估与测试。

## 数据准备
icdar2015数据集可以从[官网](https://rrc.cvc.uab.es/?ch=4&com=downloads)下载到，首次下载需注册。

将下载到的数据集解压到工作目录下，假设解压在 PaddleOCR/train_data/ 下。另外，PaddleOCR将零散的标注文件整理成单独的标注文件
，您可以通过wget的方式进行下载。
```shell
# 在PaddleOCR路径下
cd PaddleOCR/
wget -P ./train_data/  https://paddleocr.bj.bcebos.com/dataset/train_icdar2015_label.txt
wget -P ./train_data/  https://paddleocr.bj.bcebos.com/dataset/test_icdar2015_label.txt
```

PaddleOCR 也提供了数据格式转换脚本，可以将官网 label 转换支持的数据格式。 数据转换工具在 `ppocr/utils/gen_label.py`, 这里以训练集为例：

```
# 将官网下载的标签文件转换为 train_icdar2015_label.txt
python gen_label.py --mode="det" --root_path="icdar_c4_train_imgs/"  \
                    --input_path="ch4_training_localization_transcription_gt" \
                    --output_label="train_icdar2015_label.txt"
```

解压数据集和下载标注文件后，PaddleOCR/train_data/ 有两个文件夹和两个文件，分别是：
```
/PaddleOCR/train_data/icdar2015/text_localization/
  └─ icdar_c4_train_imgs/         icdar数据集的训练数据
  └─ ch4_test_images/             icdar数据集的测试数据
  └─ train_icdar2015_label.txt    icdar数据集的训练标注
  └─ test_icdar2015_label.txt     icdar数据集的测试标注
```

提供的标注文件格式如下，中间用"\t"分隔：
```
" 图像文件名                    json.dumps编码的图像标注信息"
ch4_test_images/img_61.jpg    [{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]
```
json.dumps编码前的图像标注信息是包含多个字典的list，字典中的 `points` 表示文本框的四个点的坐标(x, y)，从左上角的点开始顺时针排列。
`transcription` 表示当前文本框的文字，**当其内容为“###”时，表示该文本框无效，在训练时会跳过。**

如果您想在其他数据集上训练，可以按照上述形式构建标注文件。

## 快速启动训练

首先下载模型backbone的pretrain model，PaddleOCR的检测模型目前支持两种backbone，分别是MobileNetV3、ResNet_vd系列，
您可以根据需求使用[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/master/ppcls/modeling/architectures)中的模型更换backbone。
```shell
cd PaddleOCR/
# 下载MobileNetV3的预训练模型
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x0_5_pretrained.tar
# 或，下载ResNet18_vd的预训练模型
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/ResNet18_vd_pretrained.tar
# 或，下载ResNet50_vd的预训练模型
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar

# 解压预训练模型文件，以MobileNetV3为例
tar -xf ./pretrain_models/MobileNetV3_large_x0_5_pretrained.tar ./pretrain_models/

# 注：正确解压backbone预训练权重文件后，文件夹下包含众多以网络层命名的权重文件，格式如下：
./pretrain_models/MobileNetV3_large_x0_5_pretrained/
  └─ conv_last_bn_mean
  └─ conv_last_bn_offset
  └─ conv_last_bn_scale
  └─ conv_last_bn_variance
  └─ ......

```

#### 启动训练

*如果您安装的是cpu版本，请将配置文件中的 `use_gpu` 字段修改为false*

```shell
# 单机单卡训练 mv3_db 模型
python3 tools/train.py -c configs/det/det_mv3_db.yml \
     -o Global.pretrain_weights=./pretrain_models/MobileNetV3_large_x0_5_pretrained/
# 单机多卡训练，通过 --gpus 参数设置使用的GPU ID；如果使用的paddle版本小于2.0rc1，请使用'--select_gpus'参数选择要使用的GPU
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/det/det_mv3_db.yml \
     -o Global.pretrain_weights=./pretrain_models/MobileNetV3_large_x0_5_pretrained/
```


上述指令中，通过-c 选择训练使用configs/det/det_db_mv3.yml配置文件。
有关配置文件的详细解释，请参考[链接](./config.md)。

您也可以通过-o参数在不需要修改yml文件的情况下，改变训练的参数，比如，调整训练的学习率为0.0001
```shell
python3 tools/train.py -c configs/det/det_mv3_db.yml -o Optimizer.base_lr=0.0001
```

#### 断点训练

如果训练程序中断，如果希望加载训练中断的模型从而恢复训练，可以通过指定Global.checkpoints指定要加载的模型路径：
```shell
python3 tools/train.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=./your/trained/model


```

**注意**：`Global.checkpoints`的优先级高于`Global.pretrain_weights`的优先级，即同时指定两个参数时，优先加载`Global.checkpoints`指定的模型，如果`Global.checkpoints`指定的模型路径有误，会加载`Global.pretrain_weights`指定的模型。

## 指标评估

PaddleOCR计算三个OCR检测相关的指标，分别是：Precision、Recall、Hmean。

运行如下代码，根据配置文件`det_db_mv3.yml`中`save_res_path`指定的测试集检测结果文件，计算评估指标。

评估时设置后处理参数`box_thresh=0.6`，`unclip_ratio=1.5`，使用不同数据集、不同模型训练，可调整这两个参数进行优化
```shell
python3 tools/eval.py -c configs/det/det_mv3_db.yml  -o Global.checkpoints="{path/to/weights}/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```
训练中模型参数默认保存在`Global.save_model_dir`目录下。在评估指标时，需要设置`Global.checkpoints`指向保存的参数文件。

比如：
```shell
python3 tools/eval.py -c configs/det/det_mv3_db.yml  -o Global.checkpoints="./output/det_db/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```

* 注：`box_thresh`、`unclip_ratio`是DB后处理所需要的参数，在评估EAST模型时不需要设置

## 测试检测效果

测试单张图像的检测效果
```shell
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml -o Global.infer_img="./doc/imgs_en/img_10.jpg" Global.checkpoints="./output/det_db/best_accuracy"
```

测试DB模型时，调整后处理阈值，
```shell
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml -o Global.infer_img="./doc/imgs_en/img_10.jpg" Global.checkpoints="./output/det_db/best_accuracy" PostProcess.box_thresh=0.6 PostProcess.unclip_ratio=1.5
```


测试文件夹下所有图像的检测效果
```shell
python3 tools/infer_det.py -c configs/det/det_mv3_db.yml -o Global.infer_img="./doc/imgs_en/" Global.checkpoints="./output/det_db/best_accuracy"
```
