[English](readme_en.md) | 简体中文	

PaddleOCR提供2种服务部署方式：	
- 基于PaddleHub Serving的部署：代码路径为"`./deploy/hubserving`"，使用方法参考[文档](../hubserving/readme.md)。		
- 基于PaddleServing的部署：代码路径为"`./deploy/pdserving`"，按照本教程使用。	

# Paddle Serving 服务部署	
本教程将介绍基于[Paddle Serving](https://github.com/PaddlePaddle/Serving)部署PaddleOCR在线预测服务的详细步骤。
- [快速启动服务](#快速启动服务)
    - [1. 准备环境](#准备环境)
    - [2. 转换模型](#转换模型)
    - [3. 启动服务](#启动服务)
- [发送预测请求](#发送预测请求)

pdserving服务部署目录下包括`检测`、`方向分类器`、`识别`、`串联`四种服务部署工具，请根据需求选择相应的服务。目录结构如下：
```
deploy/pdserving/
  └─  det_local_server.py     快速版 检测 服务端
  └─  det_rpc_server.py       标准版 检测 服务端
  └─  clas_local_server.py    快速版 方向分类器 服务端
  └─  clas_rpc_server.py      标准版 方向分类器 服务端
  └─  rec_local_server.py     快速版 识别 服务端
  └─  rec_rpc_server.py       标准版 识别 服务端
  └─  ocr_local_server.py     快速版 串联 服务端
  └─  ocr_rpc_server.py       标准版 串联 服务端
  └─  pdserving_client.py     客户端
  └─  params.py               配置文件
```

<a name="快速启动服务"></a>
## 快速启动服务

<a name="准备环境"></a>
### 1. 准备环境
环境版本要求：  
- **CUDA版本：9.X/10.X**  
- **CUDNN版本：7.X**  
- **操作系统版本：Linux/Windows**  
- **Python版本： 2.7/3.5/3.6/3.7** 

**Python操作指南：**

目前Serving用于OCR的部分功能还在测试当中，因此在这里我们给出[Servnig latest package](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md)
大家根据自己的环境选择需要安装的whl包即可，例如以Python 3.5为例，执行下列命令：
```
# 安装服务端，CPU/GPU版本选择一个
# GPU版本服务端
# CUDA 9
python -m pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server_gpu-0.0.0.post9-py3-none-any.whl 
# CUDA 10
python -m pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server_gpu-0.0.0.post10-py3-none-any.whl
# CPU版本服务端
python -m pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_server-0.0.0-py3-none-any.whl

# 安装客户端和App包，CPU、GPU通用
python -m pip install -U https://paddle-serving.bj.bcebos.com/whl/paddle_serving_client-0.0.0-cp35-none-any.whl https://paddle-serving.bj.bcebos.com/whl/paddle_serving_app-0.0.0-py3-none-any.whl

# 安装其他依赖
pip3.5 install func-timeout
```

<a name="转换模型"></a>
## 2. 转换模型

Paddle Serving无法直接用训练模型（checkpoints 模型）或推理模型（inference 模型）进行部署。Serving模型由两个文件夹构成，用于存放客户端和服务端的配置。本节介绍如何将推理模型转换为Paddle Serving可部署的模型。

**以文本检测模型`ch_ppocr_mobile_v1.1_det_infer`为例，文本识别模型和方向分类器的转换同理。**

首先下载推理模型：
```shell
wget -P ./inference/ https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_infer.tar && tar xf ./inference/ch_ppocr_mobile_v1.1_det_infer.tar -C ./inference/
```
然后运行如下python脚本进行转换，其中，使用参数`model_dir`指定待转换的推理模型路径：
```
python deploy/pdserving/inference_to_serving.py --model_dir ./inference/ch_ppocr_mobile_v1.1_det_infer
```
最终会在`ch_ppocr_mobile_v1.1_det_infer`目录下生成客户端和服务端的模型配置，结构如下：
```
/ch_ppocr_mobile_v1.1_det_infer/
├── serving_client_dir # 客户端配置文件夹
└── serving_server_dir # 服务端配置文件夹
```

<a name="启动服务"></a>
## 3. 启动服务

启动服务可以根据实际需求选择启动`标准版`或者`快速版`，两种方式的对比如下表：  

|版本|特点|适用场景|
|-|-|-|
|标准版|稳定性高，分布式部署|适用于吞吐量大，需要跨机房部署的情况，只能用于Linux平台|
|快速版|部署方便，预测速度快|适用于对预测速度要求高，迭代速度快的场景，可以支持Linux/Windows|


**step 1. 配置环境变量**

```
# 以下两步的顺序不能反
export PYTHONPATH=$PWD:$PYTHONPATH
cd deploy/pdserving
```

**step 2. 修改配置参数**

配置参数在`params.py`中，具体内容如下所示，可根据需要修改相关参数，如修改模型路径、修改后处理参数等。

```
def read_params():
    cfg = Config()
    #use gpu
    cfg.use_gpu = False #是否使用GPU，False代表使用CPU
    cfg.use_pdserving = True  #使用paddle serving部署时必须为True

    #params for text detector
    cfg.det_algorithm = "DB"
    cfg.det_server_dir = "../../inference/ch_ppocr_mobile_v1.1_det_infer/serving_server_dir"
    cfg.det_client_dir = "../../inference/ch_ppocr_mobile_v1.1_det_infer/serving_client_dir"
    cfg.det_max_side_len = 960

    #DB parmas
    cfg.det_db_thresh =0.3
    cfg.det_db_box_thresh =0.5
    cfg.det_db_unclip_ratio =2.0

    #EAST parmas
    cfg.det_east_score_thresh = 0.8
    cfg.det_east_cover_thresh = 0.1
    cfg.det_east_nms_thresh = 0.2

    #params for text recognizer
    cfg.rec_algorithm = "CRNN"
    cfg.rec_server_dir = "../../inference/ch_ppocr_mobile_v1.1_rec_infer/serving_server_dir"
    cfg.rec_client_dir = "../../inference/ch_ppocr_mobile_v1.1_rec_infer/serving_client_dir"

    cfg.rec_image_shape = "3, 32, 320"
    cfg.rec_char_type = 'ch'
    cfg.rec_batch_num = 30
    cfg.max_text_length = 25

    cfg.rec_char_dict_path = "../../ppocr/utils/ppocr_keys_v1.txt"
    cfg.use_space_char = True

    #params for text classifier
    cfg.use_angle_cls = True
    cfg.cls_server_dir = "../../inference/ch_ppocr_mobile_v1.1_cls_infer/serving_server_dir"
    cfg.cls_client_dir = "../../inference/ch_ppocr_mobile_v1.1_cls_infer/serving_client_dir"
    cfg.cls_image_shape = "3, 48, 192"
    cfg.label_list = ['0', '180']
    cfg.cls_batch_num = 30
    cfg.cls_thresh = 0.9

    return cfg
```

**step 3_1. 启动独立的检测服务或识别服务**

如果只需要搭建检测服务或识别服务，一行命令即可，检测服务的启动方式如下，识别同理。检测+识别的串联服务请直接跳至step 3_2。

```
# 启动文本检测服务，标准版/快速版 二选一
python det_rpc_server.py #标准版，Linux用户
python det_local_server.py #快速版，Windows/Linux用户
```

**step 3_2. 启动文本检测、识别串联的服务**

如果需要搭建检测+识别的串联服务，快速版与step 3_1中的独立服务启动方式相同，但标准版略有不同，具体步骤如下：

```
# 标准版，Linux用户
# GPU用户
# 启动检测服务
python -m paddle_serving_server_gpu.serve --model ../../inference/ch_ppocr_mobile_v1.1_det_infer/serving_server_dir/ --port 9293 --gpu_id 0
# 启动方向分类器服务
python -m paddle_serving_server_gpu.serve --model ../../inference/ch_ppocr_mobile_v1.1_cls_infer/serving_server_dir/ --port 9294 --gpu_id 0
# 启动串联服务
python ocr_rpc_server.py 

# CPU用户
# 启动检测服务
python -m paddle_serving_server.serve --model ../../inference/ch_ppocr_mobile_v1.1_det_infer/serving_server_dir/ --port 9293
# 启动方向分类器服务
python -m paddle_serving_server.serve --model ../../inference/ch_ppocr_mobile_v1.1_cls_infer/serving_server_dir/ --port 9294
# 启动串联服务
python ocr_rpc_server.py

# 快速版，Windows/Linux用户
python ocr_local_server.py 
```

<a name="发送预测请求"></a>
## 发送预测请求
以上所有单独或串联的服务均可使用如下客户端进行访问：
```
python pdserving_client.py image_path
```

