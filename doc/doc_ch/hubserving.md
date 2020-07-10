# 服务部署

PaddleOCR提供2种服务部署方式：
- 基于HubServing的部署：已集成到PaddleOCR中（[code](https://github.com/PaddlePaddle/PaddleOCR/tree/develop/deploy/ocr_hubserving)），按照本教程使用；
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
  └─  __init__.py    空文件
  └─  config.json    配置文件，启动服务时作为参数传入
  └─  module.py      主模块，包含服务的完整逻辑
```

## 启动服务
以下步骤以检测+识别2阶段串联服务为例，如果只需要检测服务或识别服务，替换相应文件路径即可。
### 1. 安装paddlehub
```pip3 install paddlehub --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple```

### 2. 安装服务模块
PaddleOCR提供3种服务模块，根据需要安装所需模块。如： 

安装检测服务模块：  
```hub install deploy/hubserving/ocr_det/```  

或，安装识别服务模块：  
```hub install deploy/hubserving/ocr_rec/```  

或，安装检测+识别串联服务模块：  
```hub install deploy/hubserving/ocr_system/```  

### 3. 修改配置文件
在config.json中指定模型路径、是否使用GPU、是否对结果做可视化等参数，如，串联服务ocr_system的配置：
```python
{
    "modules_info": {
        "ocr_system": {
            "init_args": {
                "version": "1.0.0",
                "det_model_dir": "./inference/det/",
                "rec_model_dir": "./inference/rec/",
                "use_gpu": true
            },
            "predict_args": {
                "visualization": false
            }
        }
    }
}
```
其中，模型路径对应的模型为```inference模型```。

### 4. 运行启动命令
```hub serving start -m ocr_system --config hubserving/ocr_det/config.json```  

这样就完成了一个服务化API的部署，默认端口号为8866。

**NOTE:** 如使用GPU预测(即，config中use_gpu置为true)，则需要在启动服务之前，设置CUDA_VISIBLE_DEVICES环境变量，如：```export CUDA_VISIBLE_DEVICES=0```，否则不用设置。

## 发送预测请求
配置好服务端，以下数行代码即可实现发送预测请求，获取预测结果:

```python
import requests
import json
import cv2
import base64

def cv2_to_base64(image):
    return base64.b64encode(image).decode('utf8')

# 发送HTTP请求
data = {'images':[cv2_to_base64(open("./doc/imgs/11.jpg", 'rb').read())]}
headers = {"Content-type": "application/json"}
# url = "http://127.0.0.1:8866/predict/ocr_det"
# url = "http://127.0.0.1:8866/predict/ocr_rec"
url = "http://127.0.0.1:8866/predict/ocr_system"
r = requests.post(url=url, headers=headers, data=json.dumps(data))

# 打印预测结果
print(r.json()["results"])
```

你可能需要根据实际情况修改```url```字符串中的端口号和服务模块名称。  

上面所示代码都已写入测试脚本，可直接运行命令：```python tools/test_hubserving.py```

## 自定义修改服务模块
如果需要修改服务逻辑，你一般需要操作以下步骤：  

1、 停止服务  
```hub serving stop -m ocr_system```  

2、 到相应的module.py文件中根据实际需求修改代码  

3、 卸载旧服务包  
```hub uninstall ocr_system```  

4、 安装修改后的新服务包  
```hub install deploy/hubserving/ocr_system/```  

