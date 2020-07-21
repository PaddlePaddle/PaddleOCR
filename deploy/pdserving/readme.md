# Paddle Serving 服务部署

本教程将介绍基于[Paddle Serving](https://github.com/PaddlePaddle/Serving)部署PaddleOCR在线预测服务的详细步骤。

## 快速启动服务

### 1. 准备环境
我们先安装Paddle Serving相关组件
我们推荐用户使用GPU来做Paddle Serving的OCR服务部署 

**CUDA版本：9.0**
**CUDNN版本：7.0**
**操作系统版本：CentOS 6以上**

```
python -m pip install paddle_serving_server_gpu paddle_serving_client paddle_serving_app
```

### 2. 模型转换
可以使用`paddle_serving_app`提供的模型，执行下列命令
```
python -m paddle_serving_app.package --get_model ocr_rec
tar -xzvf ocr_rec.tar.gz
python -m paddle_serving_app.package --get_model ocr_det
tar -xzvf ocr_det.tar.gz 
```
执行上述命令会下载`db_crnn_mobile`的模型，如果想要下载规模更大的`db_crnn_server`模型，可以在下载预测模型并解压之后。参考[如何从Paddle保存的预测模型转为Paddle Serving格式可部署的模型](https://github.com/PaddlePaddle/Serving/blob/develop/doc/INFERENCE_TO_SERVING_CN.md)。

### 3. 启动服务
启动服务可以根据实际需求选择启动`标准版`或者`快速版`，两种方式的对比如下表：  

|版本|特点|适用场景|
|-|-|-|
|标准版|稳定性高，分布式部署|适用于吞吐量大，需要跨机房部署的情况|
|快速版|部署方便，预测速度快|适用于对预测速度要求高，迭代速度快的场景|

#### 方式1. 启动标准版服务

```
python -m paddle_serving_server_gpu.serve --model ocr_det_model --port 9293 --gpu_id 0
python ocr_web_server.py
```

#### 方式2. 启动快速版服务

```
python ocr_local_server.py
```

## 发送预测请求

```
python ocr_web_client.py
```

## 返回结果格式说明

返回结果是json格式
```
{u'result': {u'res': [u'\u571f\u5730\u6574\u6cbb\u4e0e\u571f\u58e4\u4fee\u590d\u7814\u7a76\u4e2d\u5fc3', u'\u534e\u5357\u519c\u4e1a\u5927\u5b661\u7d20\u56fe']}}
```
我们也可以打印结果json串中`res`字段的每一句话
```
土地整治与土壤修复研究中心
华南农业大学1素图
```

## 自定义修改服务逻辑

在`ocr_web_server.py`或是`ocr_local_server.py`当中的`preprocess`函数里面做了检测服务和识别服务的前处理，`postprocess`函数里面做了识别的后处理服务，可以在相应的函数中做修改。调用了`paddle_serving_app`库提供的常见CV模型的前处理/后处理库。

如果想要单独启动Paddle Serving的检测服务和识别服务，参见下列表格, 执行对应的脚本即可。

| 模型 | 标准版         | 快速版           |
| ---- | ----------------- | ------------------- |
| 检测 | det_web_server.py | det_local_server.py |
| 识别 | rec_web_server.py | rec_local_server.py |

更多信息参见[Paddle Serving](https://github.com/PaddlePaddle/Serving)
