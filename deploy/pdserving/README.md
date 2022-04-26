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
- [OCR Pipeline WebService](#ocr-pipeline-webservice)
- [Service deployment based on PaddleServing](#service-deployment-based-on-paddleserving)
  - [Contents](#contents)
  - [Environmental preparation](#environmental-preparation)
  - [Model conversion](#model-conversion)
  - [Paddle Serving pipeline deployment](#paddle-serving-pipeline-deployment)
  - [WINDOWS Users](#windows-users)
  - [FAQ](#faq)

<a name="environmental-preparation"></a>
## Environmental preparation

PaddleOCR operating environment and Paddle Serving operating environment are needed.

1. Please prepare PaddleOCR operating environment reference [link](../../doc/doc_ch/installation.md).
   Download the corresponding paddlepaddle whl package according to the environment, it is recommended to install version 2.2.2.

2. The steps of PaddleServing operating environment prepare are as follows:


```bash
# Install serving which used to start the service
wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl
pip3 install paddle_serving_server_gpu-0.7.0.post102-py3-none-any.whl

# Install paddle-serving-server for cuda10.1
# wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_server_gpu-0.7.0.post101-py3-none-any.whl
# pip3 install paddle_serving_server_gpu-0.7.0.post101-py3-none-any.whl

# Install serving which used to start the service
wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_client-0.7.0-cp37-none-any.whl
pip3 install paddle_serving_client-0.7.0-cp37-none-any.whl

# Install serving-app
wget https://paddle-serving.bj.bcebos.com/test-dev/whl/paddle_serving_app-0.7.0-py3-none-any.whl
pip3 install paddle_serving_app-0.7.0-py3-none-any.whl
```

   **note:** If you want to install the latest version of PaddleServing, refer to [link](https://github.com/PaddlePaddle/Serving/blob/v0.7.0/doc/Latest_Packages_CN.md).


<a name="model-conversion"></a>
## Model conversion
When using PaddleServing for service deployment, you need to convert the saved inference model into a serving model that is easy to deploy.

Firstly, download the [inference model](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/README_ch.md#pp-ocr%E7%B3%BB%E5%88%97%E6%A8%A1%E5%9E%8B%E5%88%97%E8%A1%A8%E6%9B%B4%E6%96%B0%E4%B8%AD) of PPOCR
```
# Download and unzip the OCR text detection model
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar -O ch_PP-OCRv2_det_infer.tar && tar -xf ch_PP-OCRv2_det_infer.tar
# Download and unzip the OCR text recognition model
wget https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar -O ch_PP-OCRv2_rec_infer.tar &&  tar -xf ch_PP-OCRv2_rec_infer.tar
```
Then, you can use installed paddle_serving_client tool to convert inference model to mobile model.
```
#  Detection model conversion
python3 -m paddle_serving_client.convert --dirname ./ch_PP-OCRv2_det_infer/ \
                                         --model_filename inference.pdmodel          \
                                         --params_filename inference.pdiparams       \
                                         --serving_server ./ppocr_det_mobile_2.0_serving/ \
                                         --serving_client ./ppocr_det_mobile_2.0_client/

#  Recognition model conversion
python3 -m paddle_serving_client.convert --dirname ./ch_PP-OCRv2_rec_infer/ \
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
    cd PaddleOCR/deploy/pdserving/
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

    The predicted performance data will be automatically written into the `PipelineServingLogs/pipeline.tracer` file.

    Tested on 200 real pictures, and limited the detection long side to 960. The average QPS on T4 GPU can reach around 23:

    ```

    2021-05-13 03:42:36,895 ==================== TRACER ======================
    2021-05-13 03:42:36,975 Op(rec):
    2021-05-13 03:42:36,976         in[14.472382882882883 ms]
    2021-05-13 03:42:36,976         prep[9.556855855855856 ms]
    2021-05-13 03:42:36,976         midp[59.921905405405404 ms]
    2021-05-13 03:42:36,976         postp[15.345945945945946 ms]
    2021-05-13 03:42:36,976         out[1.9921216216216215 ms]
    2021-05-13 03:42:36,976         idle[0.16254943864471572]
    2021-05-13 03:42:36,976 Op(det):
    2021-05-13 03:42:36,976         in[315.4468035714286 ms]
    2021-05-13 03:42:36,976         prep[69.5980625 ms]
    2021-05-13 03:42:36,976         midp[18.989535714285715 ms]
    2021-05-13 03:42:36,976         postp[18.857803571428573 ms]
    2021-05-13 03:42:36,977         out[3.1337544642857145 ms]
    2021-05-13 03:42:36,977         idle[0.7477961159203756]
    2021-05-13 03:42:36,977 DAGExecutor:
    2021-05-13 03:42:36,977         Query count[224]
    2021-05-13 03:42:36,977         QPS[22.4 q/s]
    2021-05-13 03:42:36,977         Succ[0.9910714285714286]
    2021-05-13 03:42:36,977         Error req[169, 170]
    2021-05-13 03:42:36,977         Latency:
    2021-05-13 03:42:36,977                 ave[535.1678348214285 ms]
    2021-05-13 03:42:36,977                 .50[172.651 ms]
    2021-05-13 03:42:36,977                 .60[187.904 ms]
    2021-05-13 03:42:36,977                 .70[245.675 ms]
    2021-05-13 03:42:36,977                 .80[526.684 ms]
    2021-05-13 03:42:36,977                 .90[854.596 ms]
    2021-05-13 03:42:36,977                 .95[1722.728 ms]
    2021-05-13 03:42:36,977                 .99[3990.292 ms]
    2021-05-13 03:42:36,978 Channel (server worker num[10]):
    2021-05-13 03:42:36,978         chl0(In: ['@DAGExecutor'], Out: ['det']) size[0/0]
    2021-05-13 03:42:36,979         chl1(In: ['det'], Out: ['rec']) size[6/0]
    2021-05-13 03:42:36,979         chl2(In: ['rec'], Out: ['@DAGExecutor']) size[0/0]
    ```

## C++ Serving

Service deployment based on python obviously has the advantage of convenient secondary development. However, the real application often needs to pursue better performance. PaddleServing also provides a more performant C++ deployment version.

The C++ service deployment is the same as python in the environment setup and data preparation stages, the difference is when the service is started and the client sends requests.

| Language | Speed ​​| Secondary development | Do you need to compile |
|-----|-----|---------|------------|
| C++ | fast | Slightly difficult | Single model prediction does not need to be compiled, multi-model concatenation needs to be compiled |
| python | general | easy | single-model/multi-model no compilation required |

1. Compile Serving

   To improve predictive performance, C++ services also provide multiple model concatenation services. Unlike Python Pipeline services, multiple model concatenation requires the pre - and post-model processing code to be written on the server side, so local recompilation is required to generate serving. Specific may refer to the official document: [how to compile Serving](https://github.com/PaddlePaddle/Serving/blob/v0.8.3/doc/Compile_EN.md)

2. Run the following command to start the service.
    ```
    # Start the service and save the running log in log.txt
    python3 -m paddle_serving_server.serve --model ppocrv2_det_serving ppocrv2_rec_serving --op GeneralDetectionOp GeneralInferOp --port 9293 &>log.txt &
    ```
    After the service is successfully started, a log similar to the following will be printed in log.txt
    ![](./imgs/start_server.png)

3. Send service request

   Due to the need for pre and post-processing in the C++Server part, in order to speed up the input to the C++Server is only the base64 encoded string of the picture, it needs to be manually modified
   Change the feed_type field and shape field in ppocrv2_det_client/serving_client_conf.prototxt to the following:

   ```
    feed_var {
    name: "x"
    alias_name: "x"
    is_lod_tensor: false
    feed_type: 20
    shape: 1
    }
   ```

   start the client:

    ```
    python3 ocr_cpp_client.py ppocrv2_det_client ppocrv2_rec_client
    ```
    After successfully running, the predicted result of the model will be printed in the cmd window. An example of the result is:
    ![](./imgs/results.png)  

## WINDOWS Users

Windows does not support Pipeline Serving, if we want to lauch paddle serving on Windows, we should use Web Service, for more infomation please refer to [Paddle Serving for Windows Users](https://github.com/PaddlePaddle/Serving/blob/develop/doc/Windows_Tutorial_EN.md)


**WINDOWS user can only use version 0.5.0 CPU Mode**

**Prepare Stage:**

```
pip3 install paddle-serving-server==0.5.0
pip3 install paddle-serving-app==0.3.1
```

1. Start Server

```
cd win
python3 ocr_web_server.py gpu(for gpu user)
or
python3 ocr_web_server.py cpu(for cpu user)
```

2. Client Send Requests

```
python3 ocr_web_client.py
```

<a name="faq"></a>
## FAQ
**Q1**: No result return after sending the request.

**A1**: Do not set the proxy when starting the service and sending the request. You can close the proxy before starting the service and before sending the request. The command to close the proxy is:
```
unset https_proxy
unset http_proxy
```  
