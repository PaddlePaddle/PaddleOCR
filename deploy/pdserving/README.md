# OCR Pipeline WebService

(English|[简体中文](./README_CN.md))

PaddleOCR provides two service deployment methods:
- Based on **PaddleHub Serving**: Code path is "`./deploy/hubserving`". Please refer to the [tutorial](../../deploy/hubserving/readme_en.md)
- Based on **PaddleServing**: Code path is "`./deploy/pdserving`". Please follow this tutorial.

# Service deployment based on PaddleServing  

This document will introduce how to use the [PaddleServing](https://github.com/PaddlePaddle/Serving/blob/develop/README.md) to deploy the PPOCR dynamic graph model as a pipeline online service.

Some Key Features of Paddle Serving:
- Integrate with Paddle training pipeline seamlessly, most paddle models can be deployed with one line command.
- Industrial serving features supported, such as models management, online loading, online A/B testing etc.
- Highly concurrent and efficient communication between clients and servers supported.

The introduction and tutorial of Paddle Serving service deployment framework reference [document](https://github.com/PaddlePaddle/Serving/blob/develop/README.md).


## Contents
- [Environmental preparation](#environmental-preparation)
- [Model conversion](#model-conversion)
- [Paddle Serving pipeline deployment](#paddle-serving-pipeline-deployment)
- [FAQ](#faq)

<a name="environmental-preparation"></a>
## Environmental preparation

PaddleOCR operating environment and Paddle Serving operating environment are needed.

1. Please prepare PaddleOCR operating environment reference [link](../../doc/doc_ch/installation.md).
   Download the corresponding paddle whl package according to the environment, it is recommended to install version 2.0.1.


2. The steps of PaddleServing operating environment prepare are as follows:

    Install serving which used to start the service
    ```
    pip3 install paddle-serving-server==0.5.0 # for CPU
    pip3 install paddle-serving-server-gpu==0.5.0 # for GPU
    # Other GPU environments need to confirm the environment and then choose to execute the following commands
    pip3 install paddle-serving-server-gpu==0.5.0.post9 # GPU with CUDA9.0
    pip3 install paddle-serving-server-gpu==0.5.0.post10 # GPU with CUDA10.0
    pip3 install paddle-serving-server-gpu==0.5.0.post101 # GPU with CUDA10.1 + TensorRT6
    pip3 install paddle-serving-server-gpu==0.5.0.post11 # GPU with CUDA10.1 + TensorRT7
    ```

3. Install the client to send requests to the service
    In [download link](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md) find the client installation package corresponding to the python version.
    The python3.7 version is recommended here:

    ```
    wget https://paddle-serving.bj.bcebos.com/whl/paddle_serving_client-0.0.0-cp37-none-any.whl
    pip3 install paddle_serving_client-0.0.0-cp37-none-any.whl
    ```

4. Install serving-app
    ```
    pip3 install paddle-serving-app==0.3.1
    ```

   **note:** If you want to install the latest version of PaddleServing, refer to [link](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md).


<a name="model-conversion"></a>
## Model conversion
When using PaddleServing for service deployment, you need to convert the saved inference model into a serving model that is easy to deploy.

Firstly, download the [inference model](https://github.com/PaddlePaddle/PaddleOCR#pp-ocr-20-series-model-listupdate-on-dec-15) of PPOCR
```
# Download and unzip the OCR text detection model
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_det_infer.tar
# Download and unzip the OCR text recognition model
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar

```
Then, you can use installed paddle_serving_client tool to convert inference model to mobile model.
```
#  Detection model conversion
python3 -m paddle_serving_client.convert --dirname ./ch_ppocr_mobile_v2.0_det_infer/ \
                                         --model_filename inference.pdmodel          \
                                         --params_filename inference.pdiparams       \
                                         --serving_server ./ppocr_det_mobile_2.0_serving/ \
                                         --serving_client ./ppocr_det_mobile_2.0_client/

#  Recognition model conversion
python3 -m paddle_serving_client.convert --dirname ./ch_ppocr_mobile_v2.0_rec_infer/ \
                                         --model_filename inference.pdmodel          \
                                         --params_filename inference.pdiparams       \
                                         --serving_server ./ppocr_rec_mobile_2.0_serving/  \
                                         --serving_client ./ppocr_rec_mobile_2.0_client/

```

After the detection model is converted, there will be additional folders of `ppocr_det_mobile_2.0_serving` and `ppocr_det_mobile_2.0_client` in the current folder, with the following format:
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
The recognition model is the same.

<a name="paddle-serving-pipeline-deployment"></a>
## Paddle Serving pipeline deployment

1. Download the PaddleOCR code, if you have already downloaded it, you can skip this step.
    ```
    git clone https://github.com/PaddlePaddle/PaddleOCR

    # Enter the working directory  
    cd PaddleOCR/deploy/pdserver/
    ```

    The pdserver directory contains the code to start the pipeline service and send prediction requests, including:
    ```
    __init__.py
    config.yml # Start the service configuration file
    ocr_reader.py # OCR model pre-processing and post-processing code implementation
    pipeline_http_client.py # Script to send pipeline prediction request
    web_service.py # Start the script of the pipeline server
    ```

2. Run the following command to start the service.
    ```
    # Start the service and save the running log in log.txt
    python3 web_service.py &>log.txt &
    ```
    After the service is successfully started, a log similar to the following will be printed in log.txt
    ![](./imgs/start_server.png)

3. Send service request
    ```
    python3 pipeline_http_client.py
    ```
    After successfully running, the predicted result of the model will be printed in the cmd window. An example of the result is:
    ![](./imgs/results.png)  

    Adjust the number of concurrency in config.yml to get the largest QPS. Generally, the number of concurrent detection and recognition is 2:1

    ```
    det:
        concurrency: 8
        ...
    rec:
        concurrency: 4
        ...
    ```

    Multiple service requests can be sent at the same time if necessary.

    The predicted performance data will be automatically written into the `PipelineServingLogs/pipeline.tracer` file:

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


<a name="faq"></a>
## FAQ
**Q1**: No result return after sending the request.

**A1**: Do not set the proxy when starting the service and sending the request. You can close the proxy before starting the service and before sending the request. The command to close the proxy is:
```
unset https_proxy
unset http_proxy
```  
