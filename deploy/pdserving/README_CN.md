# PPOCR 服务化部署

([English](./README.md)|简体中文)

PaddleOCR提供2种服务部署方式：
- 基于PaddleHub Serving的部署：代码路径为"`./deploy/hubserving`"，使用方法参考[文档](../../deploy/hubserving/readme.md)；
- 基于PaddleServing的部署：代码路径为"`./deploy/pdserving`"，按照本教程使用。

# 基于PaddleServing的服务部署

本文档将介绍如何使用[PaddleServing](https://github.com/PaddlePaddle/Serving/blob/develop/README_CN.md)工具部署PPOCR
动态图模型的pipeline在线服务。

相比较于hubserving部署，PaddleServing具备以下优点：
- 支持客户端和服务端之间高并发和高效通信
- 支持 工业级的服务能力 例如模型管理，在线加载，在线A/B测试等
- 支持 多种编程语言 开发客户端，例如C++, Python和Java

更多有关PaddleServing服务化部署框架介绍和使用教程参考[文档](https://github.com/PaddlePaddle/Serving/blob/develop/README_CN.md)。

## 目录
- [环境准备](#环境准备)
- [模型转换](#模型转换)
- [Paddle Serving pipeline部署](#部署)
- [FAQ](#FAQ)

<a name="环境准备"></a>
## 环境准备

需要准备PaddleOCR的运行环境和Paddle Serving的运行环境。

- 准备PaddleOCR的运行环境[链接](../../doc/doc_ch/installation.md)
  根据环境下载对应的paddle whl包，推荐安装2.0.1版本

- 准备PaddleServing的运行环境，步骤如下

1. 安装serving，用于启动服务
    ```
    pip3 install paddle-serving-server==0.5.0 # for CPU
    pip3 install paddle-serving-server-gpu==0.5.0 # for GPU
    # 其他GPU环境需要确认环境再选择执行如下命令
    pip3 install paddle-serving-server-gpu==0.5.0.post9 # GPU with CUDA9.0
    pip3 install paddle-serving-server-gpu==0.5.0.post10 # GPU with CUDA10.0
    pip3 install paddle-serving-server-gpu==0.5.0.post101 # GPU with CUDA10.1 + TensorRT6
    pip3 install paddle-serving-server-gpu==0.5.0.post11 # GPU with CUDA10.1 + TensorRT7
    ```

2. 安装client，用于向服务发送请求
    在[下载链接](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md)中找到对应python版本的client安装包，这里推荐python3.7版本：

    ```
    wget https://paddle-serving.bj.bcebos.com/whl/paddle_serving_client-0.0.0-cp37-none-any.whl
    pip3 install paddle_serving_client-0.0.0-cp37-none-any.whl
    ```

3. 安装serving-app
    ```
    pip3 install paddle-serving-app==0.3.1
    ```

    **Note:** 如果要安装最新版本的PaddleServing参考[链接](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md)。

<a name="模型转换"></a>
## 模型转换

使用PaddleServing做服务化部署时，需要将保存的inference模型转换为serving易于部署的模型。

首先，下载PPOCR的[inference模型](https://github.com/PaddlePaddle/PaddleOCR#pp-ocr-20-series-model-listupdate-on-dec-15)
```
# 下载并解压 OCR 文本检测模型
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_det_infer.tar
# 下载并解压 OCR 文本识别模型
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar
```

接下来，用安装的paddle_serving_client把下载的inference模型转换成易于server部署的模型格式。

```
# 转换检测模型
python3 -m paddle_serving_client.convert --dirname ./ch_ppocr_mobile_v2.0_det_infer/ \
                                         --model_filename inference.pdmodel          \
                                         --params_filename inference.pdiparams       \
                                         --serving_server ./ppocr_det_mobile_2.0_serving/ \
                                         --serving_client ./ppocr_det_mobile_2.0_client/

# 转换识别模型
python3 -m paddle_serving_client.convert --dirname ./ch_ppocr_mobile_v2.0_rec_infer/ \
                                         --model_filename inference.pdmodel          \
                                         --params_filename inference.pdiparams       \
                                         --serving_server ./ppocr_rec_mobile_2.0_serving/  \
                                         --serving_client ./ppocr_rec_mobile_2.0_client/
```

检测模型转换完成后，会在当前文件夹多出`ppocr_det_mobile_2.0_serving` 和`ppocr_det_mobile_2.0_client`的文件夹，具备如下格式：
```
|- ppocr_det_mobile_2.0_serving/
  |- __model__  
  |- __params__
  |- serving_server_conf.prototxt  
  |- serving_server_conf.stream.prototxt

|- ppocr_det_mobile_2.0_client
  |- serving_client_conf.prototxt  
  |- serving_client_conf.stream.prototxt

```
识别模型同理。

<a name="部署"></a>
## Paddle Serving pipeline部署

1. 下载PaddleOCR代码，若已下载可跳过此步骤
    ```
    git clone https://github.com/PaddlePaddle/PaddleOCR

    # 进入到工作目录
    cd PaddleOCR/deploy/pdserver/
    ```
    pdserver目录包含启动pipeline服务和发送预测请求的代码，包括：
    ```
    __init__.py
    config.yml            # 启动服务的配置文件
    ocr_reader.py         # OCR模型预处理和后处理的代码实现
    pipeline_http_client.py   # 发送pipeline预测请求的脚本
    web_service.py        # 启动pipeline服务端的脚本
    ```

2. 启动服务可运行如下命令：
    ```
    # 启动服务，运行日志保存在log.txt
    python3 web_service.py &>log.txt &
    ```
    成功启动服务后，log.txt中会打印类似如下日志
    ![](./imgs/start_server.png)

3. 发送服务请求：
    ```
    python3 pipeline_http_client.py
    ```
    成功运行后，模型预测的结果会打印在cmd窗口中，结果示例为：
    ![](./imgs/results.png)

    调整 config.yml 中的并发个数获得最大的QPS, 一般检测和识别的并发数为2：1
    ```
    det:
        #并发数，is_thread_op=True时，为线程并发；否则为进程并发
        concurrency: 8
        ...
    rec:
        #并发数，is_thread_op=True时，为线程并发；否则为进程并发
        concurrency: 4
        ...
    ```
    有需要的话可以同时发送多个服务请求

    预测性能数据会被自动写入 `PipelineServingLogs/pipeline.tracer` 文件中：

    ```
    2021-05-12 10:03:24,812 ==================== TRACER ======================
    2021-05-12 10:03:24,904 Op(rec):
    2021-05-12 10:03:24,904         in[51.5634921875 ms]
    2021-05-12 10:03:24,904         prep[215.310859375 ms]
    2021-05-12 10:03:24,904         midp[33.1617109375 ms]
    2021-05-12 10:03:24,905         postp[10.451234375 ms]
    2021-05-12 10:03:24,905         out[9.736765625 ms]
    2021-05-12 10:03:24,905         idle[0.1914292677880819]
    2021-05-12 10:03:24,905 Op(det):
    2021-05-12 10:03:24,905         in[218.63487096774193 ms]
    2021-05-12 10:03:24,906         prep[357.35925 ms]
    2021-05-12 10:03:24,906         midp[31.47598387096774 ms]
    2021-05-12 10:03:24,906         postp[15.274870967741936 ms]
    2021-05-12 10:03:24,906         out[16.245693548387095 ms]
    2021-05-12 10:03:24,906         idle[0.3675805857279226]
    2021-05-12 10:03:24,906 DAGExecutor:
    2021-05-12 10:03:24,906         Query count[128]
    2021-05-12 10:03:24,906         QPS[12.8 q/s]
    2021-05-12 10:03:24,906         Succ[1.0]
    2021-05-12 10:03:24,907         Error req[]
    2021-05-12 10:03:24,907         Latency:
    2021-05-12 10:03:24,907                 ave[798.6557734374998 ms]
    2021-05-12 10:03:24,907                 .50[867.936 ms]
    2021-05-12 10:03:24,907                 .60[914.507 ms]
    2021-05-12 10:03:24,907                 .70[961.064 ms]
    2021-05-12 10:03:24,907                 .80[1043.264 ms]
    2021-05-12 10:03:24,907                 .90[1117.923 ms]
    2021-05-12 10:03:24,907                 .95[1207.056 ms]
    2021-05-12 10:03:24,908                 .99[1325.008 ms]
    2021-05-12 10:03:24,908 Channel (server worker num[10]):
    2021-05-12 10:03:24,909         chl0(In: ['@DAGExecutor'], Out: ['det']) size[0/0]
    2021-05-12 10:03:24,909         chl1(In: ['det'], Out: ['rec']) size[1/0]
    2021-05-12 10:03:24,910         chl2(In: ['rec'], Out: ['@DAGExecutor']) size[0/0]
    ```

<a name="FAQ"></a>
## FAQ
**Q1**： 发送请求后没有结果返回或者提示输出解码报错

**A1**： 启动服务和发送请求时不要设置代理，可以在启动服务前和发送请求前关闭代理，关闭代理的命令是：
```
unset https_proxy
unset http_proxy
```
