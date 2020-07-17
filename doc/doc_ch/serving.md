# 服务部署

PaddleOCR提供2种服务部署方式：
- 基于HubServing的部署：已集成到PaddleOCR中（[code](https://github.com/PaddlePaddle/PaddleOCR/tree/develop/deploy/hubserving)），按照本教程使用；
- 基于PaddleServing的部署：详见PaddleServing官网[demo](https://github.com/PaddlePaddle/Serving/tree/develop/python/examples/ocr)，后续也将集成到PaddleOCR。  

服务部署目录下包括检测、识别、2阶段串联三种服务包，根据需求选择相应的服务包进行安装和启动。目录如下：
```
deploy/hubserving/
  └─  ocr_det     检测模块服务包
  └─  ocr_rec     识别模块服务包
  └─  ocr_system  检测+识别串联服务包
```

每个服务包下包含3个文件。以2阶段串联服务包为例，目录如下：
```
deploy/hubserving/ocr_system/
  └─  __init__.py    空文件，必选
  └─  config.json    配置文件，可选，使用配置启动服务时作为参数传入
  └─  module.py      主模块，必选，包含服务的完整逻辑
  └─  params.py      参数文件，必选，包含模型路径、前后处理参数等参数
```

## 快速启动服务
以下步骤以检测+识别2阶段串联服务为例，如果只需要检测服务或识别服务，替换相应文件路径即可。
### 1. 准备环境
```shell
# 安装paddlehub  
pip3 install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置环境变量  
export PYTHONPATH=.
```   

### 2. 安装服务模块
PaddleOCR提供3种服务模块，根据需要安装所需模块。如： 

安装检测服务模块：  
```hub install deploy/hubserving/ocr_det/```  

或，安装识别服务模块：    
```hub install deploy/hubserving/ocr_rec/```  

或，安装检测+识别串联服务模块：  
```hub install deploy/hubserving/ocr_system/```  

### 3. 启动服务
#### 方式1. 命令行命令启动（仅支持CPU）
**启动命令：**  
```shell
$ hub serving start --modules [Module1==Version1, Module2==Version2, ...] \
                    --port XXXX \
                    --use_multiprocess \
                    --workers \
```  

**参数：**  

|参数|用途|  
|-|-|  
|--modules/-m|PaddleHub Serving预安装模型，以多个Module==Version键值对的形式列出<br>*`当不指定Version时，默认选择最新版本`*|  
|--port/-p|服务端口，默认为8866|  
|--use_multiprocess|是否启用并发方式，默认为单进程方式，推荐多核CPU机器使用此方式<br>*`Windows操作系统只支持单进程方式`*|
|--workers|在并发方式下指定的并发任务数，默认为`2*cpu_count-1`，其中`cpu_count`为CPU核数|  

如启动串联服务：  ```hub serving start -m ocr_system```  

这样就完成了一个服务化API的部署，使用默认端口号8866。

#### 方式2. 配置文件启动（支持CPU、GPU）
**启动命令：**  
```hub serving start --config/-c config.json```  

其中，`config.json`格式如下： 
```python
{
    "modules_info": {
        "ocr_system": {
            "init_args": {
                "version": "1.0.0",
                "use_gpu": true
            },
            "predict_args": {
            }
        }
    },
    "port": 8868,
    "use_multiprocess": false,
    "workers": 2
}
```

- `init_args`中的可配参数与`module.py`中的`_initialize`函数接口一致。其中，**当`use_gpu`为`true`时，表示使用GPU启动服务**。  
- `predict_args`中的可配参数与`module.py`中的`predict`函数接口一致。

**注意:**  
- 使用配置文件启动服务时，其他参数会被忽略。
- 如果使用GPU预测(即，`use_gpu`置为`true`)，则需要在启动服务之前，设置CUDA_VISIBLE_DEVICES环境变量，如：```export CUDA_VISIBLE_DEVICES=0```，否则不用设置。
- **`use_gpu`不可与`use_multiprocess`同时为`true`**。

如，使用GPU 3号卡启动串联服务：  
```shell
export CUDA_VISIBLE_DEVICES=3
hub serving start -c deploy/hubserving/ocr_system/config.json
```  

## 发送预测请求
配置好服务端，可使用以下命令发送预测请求，获取预测结果:  

```python tools/test_hubserving.py server_url image_path```  

需要给脚本传递2个参数：  
- **server_url**：服务地址，格式为  
`http://[ip_address]:[port]/predict/[module_name]`  
例如，如果使用配置文件启动检测、识别、检测+识别2阶段服务，那么发送请求的url将分别是：  
`http://127.0.0.1:8866/predict/ocr_det`  
`http://127.0.0.1:8867/predict/ocr_rec`  
`http://127.0.0.1:8868/predict/ocr_system`  
- **image_path**：测试图像路径，可以是单张图片路径，也可以是图像集合目录路径  

访问示例：  
```python tools/test_hubserving.py http://127.0.0.1:8868/predict/ocr_system ./doc/imgs/```

## 返回结果格式说明
返回结果为列表（list），列表中的每一项为词典（dict），词典一共可能包含3种字段，信息如下：

|字段名称|数据类型|意义| 
|-|-|-|
|text|str|文本内容|
|confidence|float| 文本识别置信度|
|text_region|list|文本位置坐标|

不同模块返回的字段不同，如，文本识别服务模块返回结果不含`text_region`字段，具体信息如下：

|字段名/模块名|ocr_det|ocr_rec|ocr_system|
|-|-|-|-|  
|text||✔|✔| 
|confidence||✔|✔| 
|text_region|✔||✔| 

**说明：** 如果需要增加、删除、修改返回字段，可在相应模块的`module.py`文件中进行修改，完整流程参考下一节自定义修改服务模块。

## 自定义修改服务模块
如果需要修改服务逻辑，你一般需要操作以下步骤（以修改`ocr_system`为例）：  

- 1、 停止服务  
```hub serving stop --port/-p XXXX```  

- 2、 到相应的`module.py`和`params.py`等文件中根据实际需求修改代码。  
例如，如果需要替换部署服务所用模型，则需要到`params.py`中修改模型路径参数`det_model_dir`和`rec_model_dir`，当然，同时可能还需要修改其他相关参数，请根据实际情况修改调试。 建议修改后先直接运行`module.py`调试，能正确运行预测后再启动服务测试。

- 3、 卸载旧服务包  
```hub uninstall ocr_system```  

- 4、 安装修改后的新服务包  
```hub install deploy/hubserving/ocr_system/```  

- 5、重新启动服务  
```hub serving start -m ocr_system```  

