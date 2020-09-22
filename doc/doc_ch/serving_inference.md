# 使用Paddle Serving预测推理

阅读本文档之前，请先阅读文档 [基于Python预测引擎推理](./inference.md)

同本地执行预测一样，我们需要保存一份可以用于Paddle Serving的模型。

接下来首先介绍如何将训练的模型转换成Paddle Serving模型，然后将依次介绍文本检测、文本识别以及两者串联基于预测引擎推理。



## 一、训练模型转Serving模型

### 检测模型转Serving模型

下载超轻量级中文检测模型：

```
wget -P ./ch_lite/ https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar && tar xf ./ch_lite/ch_det_mv3_db.tar -C ./ch_lite/
```

上述模型是以MobileNetV3为backbone训练的DB算法，将训练好的模型转换成Serving模型只需要运行如下命令：

```
# -c后面设置训练算法的yml配置文件
# -o配置可选参数
# Global.checkpoints参数设置待转换的训练模型地址，不用添加文件后缀.pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python tools/export_serving_model.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=./ch_lite/det_mv3_db/best_accuracy Global.save_inference_dir=./inference/det_db/
```

转Serving模型时，使用的配置文件和训练时使用的配置文件相同。另外，还需要设置配置文件中的`Global.checkpoints`、`Global.save_inference_dir`参数。 其中`Global.checkpoints`指向训练中保存的模型参数文件，`Global.save_inference_dir`是生成的inference模型要保存的目录。 转换成功后，在`save_inference_dir`目录下有两个文件：

```
inference/det_db/
├── serving_client_dir # 客户端配置文件夹
└── serving_server_dir # 服务端配置文件夹

```

### 识别模型转Serving模型

下载超轻量中文识别模型：

```
wget -P ./ch_lite/ https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar && tar xf ./ch_lite/ch_rec_mv3_crnn.tar -C ./ch_lite/
```

识别模型转inference模型与检测的方式相同，如下：

```
# -c后面设置训练算法的yml配置文件
# -o配置可选参数
# Global.checkpoints参数设置待转换的训练模型地址，不用添加文件后缀.pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3 tools/export_serving_model.py -c configs/rec/rec_chinese_lite_train.yml -o Global.checkpoints=./ch_lite/rec_mv3_crnn/best_accuracy \
        Global.save_inference_dir=./inference/rec_crnn/
```

**注意：**如果您是在自己的数据集上训练的模型，并且调整了中文字符的字典文件，请注意修改配置文件中的`character_dict_path`是否是所需要的字典文件。

转换成功后，在目录下有两个文件：

```
/inference/rec_crnn/
├── serving_client_dir # 客户端配置文件夹
└── serving_server_dir # 服务端配置文件夹
```

### 方向分类模型转Serving模型

下载方向分类模型：

```
wget -P ./ch_lite/ https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile-v1.1.cls_pre.tar && tar xf ./ch_lite/ch_ppocr_mobile-v1.1.cls_pre.tar -C ./ch_lite/
```

方向分类模型转inference模型与检测的方式相同，如下：

```
# -c后面设置训练算法的yml配置文件
# -o配置可选参数
# Global.checkpoints参数设置待转换的训练模型地址，不用添加文件后缀.pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3 tools/export_serving_model.py -c configs/cls/cls_mv3.yml -o Global.checkpoints=./ch_lite/cls_model/best_accuracy \
        Global.save_inference_dir=./inference/cls/
```

转换成功后，在目录下有两个文件：

```
/inference/cls/
├── serving_client_dir # 客户端配置文件夹
└── serving_server_dir # 服务端配置文件夹
```

在接下来的教程中，我们将给出推理的demo模型下载链接。

```
wget --no-check-certificate https://paddleocr.bj.bcebos.com/deploy/pdserving/ocr_pdserving_suite.tar.gz
tar zxf ocr_pdserving_suite.tar.gz
```



## 二、文本检测模型Serving推理

文本检测模型推理，默认使用DB模型的配置参数。当不使用DB模型时，在推理时，需要通过传入相应的参数进行算法适配，细节参考下文。

与本地预测不同的是，Serving预测需要一个客户端和一个服务端，因此接下来的教程都是两行代码。所有的

### 1. 超轻量中文检测模型推理

超轻量中文检测模型推理，可以执行如下命令启动服务端：

```
#根据环境只需要启动其中一个就可以
python det_rpc_server.py --use_pdserving True --det_model_dir det_mv_server #标准版，Linux用户
python det_local_server.py --use_pdserving True --det_model_dir det_mv_server #快速版，Windows/Linux用户
```
如果需要使用CPU版本，还需增加 `--use_gpu False`。

客户端

```
python det_web_client.py
```



Serving的推测和本地预测不同点在于，客户端发送请求到服务端，服务端需要检测到文字框之后返回框的坐标，此处没有后处理的图片，只能看到坐标值。

### 2. DB文本检测模型推理

首先将DB文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在ICDAR2015英文数据集训练的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/det_r50_vd_db.tar))，可以使用如下命令进行转换：

```
# -c后面设置训练算法的yml配置文件
# Global.checkpoints参数设置待转换的训练模型地址，不用添加文件后缀.pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3 tools/export_serving_model.py -c configs/det/det_r50_vd_db.yml -o Global.checkpoints="./models/det_r50_vd_db/best_accuracy" Global.save_inference_dir="./inference/det_db"
```

经过转换之后，会在`./inference/det_db` 目录下出现`serving_server_dir`和`serving_client_dir`，然后指定`det_model_dir` 。

## 三、文本识别模型Serving推理

下面将介绍超轻量中文识别模型推理、基于CTC损失的识别模型推理和基于Attention损失的识别模型推理。对于中文文本识别，建议优先选择基于CTC损失的识别模型，实践中也发现基于Attention损失的效果不如基于CTC损失的识别模型。此外，如果训练时修改了文本的字典，请参考下面的自定义文本识别字典的推理。

### 1. 超轻量中文识别模型推理

超轻量中文识别模型推理，可以执行如下命令启动服务端：

```
#根据环境只需要启动其中一个就可以
python rec_rpc_server.py --use_pdserving True --rec_model_dir ocr_rec_server #标准版，Linux用户
python rec_local_server.py --use_pdserving True --rec_model_dir ocr_rec_server #快速版，Windows/Linux用户
```
如果需要使用CPU版本，还需增加 `--use_gpu False`。

客户端

```
python rec_web_client.py
```

![](../imgs_words/ch/word_4.jpg)

执行命令后，上面图像的预测结果（识别的文本和得分）会打印到屏幕上，示例如下：

```
{u'result': {u'score': [u'0.89547354'], u'pred_text': ['实力活力']}}
```



## 四、方向分类模型推理

下面将介绍方向分类模型推理。



### 1. 方向分类模型推理

方向分类模型推理， 可以执行如下命令启动服务端：

```
#根据环境只需要启动其中一个就可以
python clas_rpc_server.py --use_pdserving True --cls_model_dir ocr_clas_server #标准版，Linux用户
python clas_local_server.py --use_pdserving True --cls_model_dir ocr_clas_server #快速版，Windows/Linux用户
```
如果需要使用CPU版本，还需增加 `--use_gpu False`。

客户端

```
python rec_web_client.py
```

![](../imgs_words/ch/word_4.jpg)

执行命令后，上面图像的预测结果（分类的方向和得分）会打印到屏幕上，示例如下：

```
{u'result': {u'direction': [u'0'], u'score': [u'0.9999963']}}
```


## 五、文本检测、方向分类和文字识别串联Serving推理

### 1. 超轻量中文OCR模型推理

在执行预测时，需要通过参数`image_dir`指定单张图像或者图像集合的路径、参数`det_model_dir`,`cls_model_dir`和`rec_model_dir`分别指定检测，方向分类和识别的inference模型路径。参数`use_angle_cls`用于控制是否启用方向分类模型。与本地预测不同的是，为了减少网络传输耗时，可视化识别结果目前不做处理，用户收到的是推理得到的文字字段。

执行如下命令启动服务端：

```
#标准版，Linux用户
#GPU用户
python -m paddle_serving_server_gpu.serve --model det_mv_server --port 9293 --gpu_id 0
python -m paddle_serving_server_gpu.serve --model ocr_cls_server --port 9294 --gpu_id 0
python ocr_rpc_server.py --use_pdserving True --use_gpu True --rec_model_dir ocr_rec_server
#CPU用户
python -m paddle_serving_server.serve --model det_mv_server --port 9293
python -m paddle_serving_server.serve --model ocr_cls_server --port 9294
python ocr_rpc_server.py --use_pdserving True --use_gpu False --rec_model_dir ocr_rec_server

#快速版，Windows/Linux用户
python ocr_local_server.py --use_gpu False --use_pdserving True --rec_model_dir ocr_rec_server/ --det_model_dir det_mv_server/ --cls_model_dir ocr_clas_server/ --rec_char_dict_path ppocr_keys_v1.txt  --use_angle_cls True
```

客户端

```
python rec_web_client.py
```
