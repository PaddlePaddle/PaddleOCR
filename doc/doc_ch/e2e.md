# 端到端文字识别

本节以partvgg/totaltext数据集为例，介绍PaddleOCR中端到端模型的训练、评估与测试。

## 数据准备
支持两种不同的数据形式textnet / icdar ，分别为四点标注数据和十四点标注数据，十四点标注数据效果要比四点标注效果好
###数据形式为textnet

解压数据集和下载标注文件后，PaddleOCR/train_data/part_vgg_synth/train/ 有一个文件夹和一个文件，分别是：
```
/PaddleOCR/train_data/part_vgg_synth/train/
  |- image/         partvgg数据集的训练数据
      |- 119_nile_110_31.png
      | ...
  |- train_annotation_info.txt     partvgg数据集的测试标注
```

提供的标注文件格式如下，中间用"\t"分隔：
```
" 图像文件名      图像标注信息--四点标注                                         图像标注信息--识别标注  
119_nile_110_31    140.2    222.5    266.0    194.6    278.7    251.8    152.9    279.7    Path:    32.9    133.1    106.0    130.8    106.4    143.8    33.3    146.1    were    21.8    81.9    106.9    80.4    107.7    123.2    22.6    124.7    why
```
标注文件txt当中，其中每一行代表一组数据，以第一行为例。第一个代表同级目录image/下面的文件名前缀， 后面每9个代表一组标注信息，前8个代表文本框的四个点坐标（x,y)，从左上角的点开始顺时针排列。
最后一个代表文字的识别结果，**当其内容为“###”时，表示该文本框无效，在训练时会跳过。**


###数据形式为icdar
解压数据集和下载标注文件后，PaddleOCR/train_data/total_text/train/ 有两个文件夹，分别是：
```
/PaddleOCR/train_data/total_text/train/
  |- rgb/           total_text数据集的训练数据
      |- gt_0.png
      | ...  
  |- poly/           total_text数据集的测试标注
      |- gt_0.txt
      | ...
```

提供的标注文件格式如下，中间用"\t"分隔：
```
" 图像标注信息--十四点标注数据                                                                                                                                                              图像标注信息--识别标注  
1004.0,689.0,1019.0,698.0,1034.0,708.0,1049.0,718.0,1064.0,728.0,1079.0,738.0,1095.0,748.0,1094.0,774.0,1079.0,765.0,1065.0,756.0,1050.0,747.0,1036.0,738.0,1021.0,729.0,1007.0,721.0    EST
1102.0,755.0,1116.0,764.0,1131.0,773.0,1146.0,783.0,1161.0,792.0,1176.0,801.0,1191.0,811.0,1193.0,837.0,1178.0,828.0,1164.0,819.0,1150.0,810.0,1135.0,801.0,1121.0,792.0,1107.0,784.0    1972
```
标注文件当中，其中每一个txt文件代表一组数据，文件名就是同级目录rgb/下面的文件名。以第一行为例，前面28个代表文本框的十四个点坐标（x,y)，从左上角的点开始顺时针排列。
最后一个代表文字的识别结果，**当其内容为“###”时，表示该文本框无效，在训练时会跳过。**
如果您想在其他数据集上训练，可以按照上述形式构建标注文件。

## 快速启动训练

首先下载模型backbone的pretrain model，PaddleOCR的检测模型目前支持两种backbone，分别是MobileNetV3、ResNet_vd系列，
您可以根据需求使用[PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/master/ppcls/modeling/architectures)中的模型更换backbone。
```shell
cd PaddleOCR/
下载ResNet50_vd的动态图预训练模型
wget -P ./pretrain_models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ResNet50_vd_ssld_pretrained.pdparams

./pretrain_models/
  └─ ResNet50_vd_ssld_pretrained.pdparams

```

#### 启动训练

*如果您安装的是cpu版本，请将配置文件中的 `use_gpu` 字段修改为false*

```shell
# 单机单卡训练 e2e 模型
python3 tools/train.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.pretrained_model=./pretrain_models/ResNet50_vd_ssld_pretrained Global.load_static_weights=False
# 单机多卡训练，通过 --gpus 参数设置使用的GPU ID
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.pretrained_model=./pretrain_models/ResNet50_vd_ssld_pretrained  Global.load_static_weights=False
```


上述指令中，通过-c 选择训练使用configs/e2e/e2e_r50_vd_pg.yml配置文件。
有关配置文件的详细解释，请参考[链接](./config.md)。

您也可以通过-o参数在不需要修改yml文件的情况下，改变训练的参数，比如，调整训练的学习率为0.0001
```shell
python3 tools/train.py -c configs/e2e/e2e_r50_vd_pg.yml -o Optimizer.base_lr=0.0001
```

#### 断点训练

如果训练程序中断，如果希望加载训练中断的模型从而恢复训练，可以通过指定Global.checkpoints指定要加载的模型路径：
```shell
python3 tools/train.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.checkpoints=./your/trained/model
```

**注意**：`Global.checkpoints`的优先级高于`Global.pretrain_weights`的优先级，即同时指定两个参数时，优先加载`Global.checkpoints`指定的模型，如果`Global.checkpoints`指定的模型路径有误，会加载`Global.pretrain_weights`指定的模型。

## 指标评估

PaddleOCR计算三个OCR端到端相关的指标，分别是：Precision、Recall、Hmean。

运行如下代码，根据配置文件`e2e_r50_vd_pg.yml`中`save_res_path`指定的测试集检测结果文件，计算评估指标。

评估时设置后处理参数`max_side_len=768`，使用不同数据集、不同模型训练，可调整参数进行优化
训练中模型参数默认保存在`Global.save_model_dir`目录下。在评估指标时，需要设置`Global.checkpoints`指向保存的参数文件。
```shell
python3 tools/eval.py -c configs/e2e/e2e_r50_vd_pg.yml  -o Global.checkpoints="{path/to/weights}/best_accuracy"
```



## 测试端到端效果

测试单张图像的端到端识别效果
```shell
python3 tools/infer_e2e.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.infer_img="./doc/imgs_en/img_10.jpg" Global.pretrained_model="./output/det_db/best_accuracy" Global.load_static_weights=false
```

测试文件夹下所有图像的端到端识别效果
```shell
python3 tools/infer_e2e.py -c configs/e2e/e2e_r50_vd_pg.yml -o Global.infer_img="./doc/imgs_en/" Global.pretrained_model="./output/det_db/best_accuracy" Global.load_static_weights=false
```
