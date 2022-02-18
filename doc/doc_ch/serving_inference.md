# 使用Paddle Serving预测推理

阅读本文档之前，请先阅读文档 [基于Python预测引擎推理](./inference.md)

同本地执行预测一样，我们需要保存一份可以用于Paddle Serving的模型。

接下来首先介绍如何将训练的模型转换成Paddle Serving模型，然后将依次介绍文本检测、文本识别以及两者串联基于预测引擎推理。

### 一、 准备环境
我们先安装Paddle Serving相关组件
我们推荐用户使用GPU来做Paddle Serving的OCR服务部署

**CUDA版本：9.X/10.X**

**CUDNN版本：7.X**

**操作系统版本：Linux/Windows**

**Python版本： 2.7/3.5/3.6/3.7**

**Python操作指南：**

目前Serving用于OCR的部分功能还在测试当中，因此在这里我们给出[Servnig latest package](https://github.com/PaddlePaddle/Serving/blob/develop/doc/Latest_Packages_CN.md)
大家根据自己的环境选择需要安装的whl包即可，例如以Python 3.5为例，执行下列命令
```
#CPU/GPU版本选择一个
#GPU版本服务端
#CUDA 9
python -m pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server_gpu-0.0.0.post9-py3-none-any.whl
#CUDA 10
python -m pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server_gpu-0.0.0.post10-py3-none-any.whl
#CPU版本服务端
python -m pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server-0.0.0-py3-none-any.whl
#客户端和App包使用以下链接（CPU，GPU通用）
python -m pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_client-0.0.0-cp36-none-any.whl https://paddle-serving.bj.bcebos.com/whl/paddle_serving_app-0.0.0-py3-none-any.whl
```

## 二、训练模型转Serving模型

在前序文档 [基于Python预测引擎推理](./inference.md) 中，我们提供了如何把训练的checkpoint转换成Paddle模型。Paddle模型通常由一个文件夹构成，内含模型结构描述文件`model`和模型参数文件`params`。Serving模型由两个文件夹构成，用于存放客户端和服务端的配置。

我们以`ch_rec_r34_vd_crnn`模型作为例子，下载链接在：

```
wget --no-check-certificate https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar
tar xf ch_rec_r34_vd_crnn_infer.tar
```
因此我们按照Serving模型转换教程，运行下列python文件。
```
python tools/inference_to_serving.py --model_dir ch_rec_r34_vd_crnn
```
最终会在`serving_client_dir`和`serving_server_dir`生成客户端和服务端的模型配置。其中`serving_server_dir`和`serving_client_dir`的名字可以自定义。最终文件结构如下

```
/ch_rec_r34_vd_crnn/
├── serving_client_dir # 客户端配置文件夹
└── serving_server_dir # 服务端配置文件夹
```

## 三、文本检测模型Serving推理

启动服务可以根据实际需求选择启动`标准版`或者`快速版`，两种方式的对比如下表：  

|版本|特点|适用场景|
|-|-|-|
|标准版|稳定性高，分布式部署|适用于吞吐量大，需要跨机房部署的情况|
|快速版|部署方便，预测速度快|适用于对预测速度要求高，迭代速度快的场景，Windows用户只能选择快速版|

接下来的命令中，我们会指定快速版和标准版的命令。需要说明的是，标准版只能用Linux平台，快速版可以支持Linux/Windows。
文本检测模型推理，默认使用DB模型的配置参数，识别默认为CRNN。

配置文件在`params.py`中，我们贴出配置部分，如果需要做改动，也在这个文件内部进行修改。

```
def read_params():
    cfg = Config()
    #use gpu
    cfg.use_gpu = False # 是否使用GPU
    cfg.use_pdserving = True # 是否使用paddleserving，必须为True

    #params for text detector
    cfg.det_algorithm = "DB" # 检测算法， DB/EAST等
    cfg.det_model_dir = "./det_mv_server/" # 检测算法模型路径
    cfg.det_max_side_len = 960

    #DB params
    cfg.det_db_thresh =0.3
    cfg.det_db_box_thresh =0.5
    cfg.det_db_unclip_ratio =2.0

    #EAST params
    cfg.det_east_score_thresh = 0.8
    cfg.det_east_cover_thresh = 0.1
    cfg.det_east_nms_thresh = 0.2

    #params for text recognizer
    cfg.rec_algorithm = "CRNN" # 识别算法， CRNN/RARE等
    cfg.rec_model_dir = "./ocr_rec_server/" # 识别算法模型路径

    cfg.rec_image_shape = "3, 32, 320"
    cfg.rec_char_type = 'ch'
    cfg.rec_batch_num = 30
    cfg.max_text_length = 25

    cfg.rec_char_dict_path = "./ppocr_keys_v1.txt" # 识别算法字典文件
    cfg.use_space_char = True

    #params for text classifier
    cfg.use_angle_cls = True # 是否启用分类算法
    cfg.cls_model_dir = "./ocr_clas_server/" # 分类算法模型路径
    cfg.cls_image_shape = "3, 48, 192"
    cfg.label_list = ['0', '180']
    cfg.cls_batch_num = 30
    cfg.cls_thresh = 0.9

    return cfg
```
与本地预测不同的是，Serving预测需要一个客户端和一个服务端，因此接下来的教程都是两行代码。

在正式执行服务端启动命令之前，先export PYTHONPATH到工程主目录下。
```
export PYTHONPATH=$PWD:$PYTHONPATH
cd deploy/pdserving
```
为了方便用户复现Demo程序，我们提供了Chinese and English ultra-lightweight OCR model (8.1M)版本的Serving模型
```
wget --no-check-certificate https://paddleocr.bj.bcebos.com/deploy/pdserving/ocr_pdserving_suite.tar.gz
tar xf ocr_pdserving_suite.tar.gz
```

### 1. 超轻量中文检测模型推理

超轻量中文检测模型推理，可以执行如下命令启动服务端：

```
#根据环境只需要启动其中一个就可以
python det_rpc_server.py #标准版，Linux用户
python det_local_server.py #快速版，Windows/Linux用户
```

客户端

```
python det_web_client.py
```


Serving的推测和本地预测不同点在于，客户端发送请求到服务端，服务端需要检测到文字框之后返回框的坐标，此处没有后处理的图片，只能看到坐标值。

## 四、文本识别模型Serving推理

下面将介绍超轻量中文识别模型推理、基于CTC损失的识别模型推理和基于Attention损失的识别模型推理。对于中文文本识别，建议优先选择基于CTC损失的识别模型，实践中也发现基于Attention损失的效果不如基于CTC损失的识别模型。此外，如果训练时修改了文本的字典，请参考下面的自定义文本识别字典的推理。

### 1. 超轻量中文识别模型推理

超轻量中文识别模型推理，可以执行如下命令启动服务端：
需要注意params.py中的`--use_gpu`的值
```
#根据环境只需要启动其中一个就可以
python rec_rpc_server.py #标准版，Linux用户
python rec_local_server.py #快速版，Windows/Linux用户
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



## 五、方向分类模型推理

下面将介绍方向分类模型推理。



### 1. 方向分类模型推理

方向分类模型推理， 可以执行如下命令启动服务端：
需要注意params.py中的`--use_gpu`的值
```
#根据环境只需要启动其中一个就可以
python clas_rpc_server.py #标准版，Linux用户
python clas_local_server.py #快速版，Windows/Linux用户
```

客户端

```
python rec_web_client.py
```

![](../imgs_words/ch/word_4.jpg)

执行命令后，上面图像的预测结果（分类的方向和得分）会打印到屏幕上，示例如下：

```
{u'result': {u'direction': [u'0'], u'score': [u'0.9999963']}}
```


## 六、文本检测、方向分类和文字识别串联Serving推理

### 1. 超轻量中文OCR模型推理

在执行预测时，需要通过参数`image_dir`指定单张图像或者图像集合的路径、参数`det_model_dir`,`cls_model_dir`和`rec_model_dir`分别指定检测，方向分类和识别的inference模型路径。参数`use_angle_cls`用于控制是否启用方向分类模型。与本地预测不同的是，为了减少网络传输耗时，可视化识别结果目前不做处理，用户收到的是推理得到的文字字段。

执行如下命令启动服务端：
需要注意params.py中的`--use_gpu`的值
```
#标准版，Linux用户
#GPU用户
python -m paddle_serving_server_gpu.serve --model det_infer_server --port 9293 --gpu_id 0
python -m paddle_serving_server_gpu.serve --model cls_infer_server --port 9294 --gpu_id 0
python ocr_rpc_server.py
#CPU用户
python -m paddle_serving_server.serve --model det_infer_server --port 9293
python -m paddle_serving_server.serve --model cls_infer_server --port 9294
python ocr_rpc_server.py

#快速版，Windows/Linux用户
python ocr_local_server.py
```

客户端

```
python rec_web_client.py
```
