# OCR Pipeline WebService

(English|[简体中文](./README_CN.md))

PaddleOCR provides 2 service deployment methods:
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

Need to prepare PaddleOCR operating environment and Paddle Serving operating environment.

1. Prepare PaddleOCR operating environment reference [link](../../doc/doc_ch/installation.md).

2. Prepare the operating environment of PaddleServing, the steps are as follows

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
    ```
    pip3 install paddle-serving-client==0.5.0 # for CPU

    pip3 install paddle-serving-client-gpu==0.5.0 # for GPU
    ```

4. Install serving-app
    ```
    pip3 install paddle-serving-app==0.3.0
    # fix local_predict to support load dynamic model
    # find the install directoory of paddle_serving_app
    vim /usr/local/lib/python3.7/site-packages/paddle_serving_app/local_predict.py
    # replace line 85 of local_predict.py config = AnalysisConfig(model_path) with:
    if os.path.exists(os.path.join(model_path, "__params__")):
        config = AnalysisConfig(os.path.join(model_path, "__model__"), os.path.join(model_path, "__params__"))
    else:
        config = AnalysisConfig(model_path)
    ```


   **note:** If you want to install the latest version of PaddleServing, refer to [link](https://github.com/PaddlePaddle/Serving/blob/develop/doc/LATEST_PACKAGES.md).


<a name="model-conversion"></a>
## Model conversion
When using PaddleServing for service deployment, you need to convert the saved inference model into a serving model that is easy to deploy.

Firstly, download the [inference model](https://github.com/PaddlePaddle/PaddleOCR#pp-ocr-20-series-model-listupdate-on-dec-15) of PPOCR
```
# Download and unzip the OCR text detection model
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar && tar xf ch_ppocr_server_v2.0_det_infer.tar
# Download and unzip the OCR text recognition model
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar && tar xf ch_ppocr_server_v2.0_rec_infer.tar

# Conversion detection model
python3 -m paddle_serving_client.convert --dirname ./ch_ppocr_server_v2.0_det_infer/ \
                                         --model_filename inference.pdmodel          \
                                         --params_filename inference.pdiparams       \
                                         --serving_server ./ppocr_det_server_2.0_serving/ \
                                         --serving_client ./ppocr_det_server_2.0_client/

# Conversion recognition model
python3 -m paddle_serving_client.convert --dirname ./ch_ppocr_server_v2.0_rec_infer/ \
                                         --model_filename inference.pdmodel          \
                                         --params_filename inference.pdiparams       \
                                         --serving_server ./ppocr_rec_server_2.0_serving/  \
                                         --serving_client ./ppocr_rec_server_2.0_client/

```

After the detection model is converted, there will be additional folders of `ppocr_det_server_2.0_serving` and `ppocr_det_server_2.0_client` in the current folder, with the following format:
```
|- ppocr_det_server_2.0_serving/
   |- __model__
   |- __params__
   |- serving_server_conf.prototxt
   |- serving_server_conf.stream.prototxt

|- ppocr_det_server_2.0_client
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

<a name="faq"></a>
## FAQ
** Q1**: No result return after sending the request.

** A1**: Do not set the proxy when starting the service and sending the request. You can close the proxy before starting the service and before sending the request. The command to close the proxy is:
```
unset https_proxy
unset http_proxy
```  
