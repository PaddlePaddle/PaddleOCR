English | [简体中文](readme.md)

PaddleOCR provides 2 service deployment methods: 
- Based on **PaddleHub Serving**: Code path is "`./deploy/hubserving`". Please refer to the [tutorial](../hubserving/readme_en.md) for usage.
- Based on **PaddleServing**: Code path is "`./deploy/pdserving`". Please follow this tutorial.

# Service deployment based on Paddle Serving

This tutorial will introduce the detail steps of deploying PaddleOCR online prediction service based on [Paddle Serving](https://github.com/PaddlePaddle/Serving).

## Quick start service

### 1. Prepare the environment
Let's first install the relevant components of Paddle Serving. GPU is recommended for service deployment with Paddle Serving.

**Requirements:**
- **CUDA version: 9.0**
- **CUDNN version: 7.0**
- **Operating system version: >= CentOS 6**
- **Python version： 2.7/3.6/3.7**

**Installation：**
```
# install GPU server
python -m pip install paddle_serving_server_gpu

# or, install CPU server
python -m pip install paddle_serving_server

# install client and App package (CPU/GPU)
python -m pip install paddle_serving_app paddle_serving_client
```

### 2. Model transformation
You can directly use converted model provided by `paddle_serving_app` for convenience. Execute the following command to obtain:
```
python -m paddle_serving_app.package --get_model ocr_rec
tar -xzvf ocr_rec.tar.gz
python -m paddle_serving_app.package --get_model ocr_det
tar -xzvf ocr_det.tar.gz 
```
Executing the above command will download the `db_crnn_mobile` model, which is in different format with inference model. If you want to use other models for deployment, you can refer to the [tutorial](https://github.com/PaddlePaddle/Serving/blob/develop/doc/INFERENCE_TO_SERVING_CN.md) to convert your inference model to a model which is deployable for Paddle Serving.

We take `ch_rec_r34_vd_crnn` model as example. Download the inference model by executing the following command:
```
wget --no-check-certificate https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar
tar xf ch_rec_r34_vd_crnn_infer.tar
```

Convert the downloaded model by executing the following python script:
```
from paddle_serving_client.io import inference_model_to_serving
inference_model_dir = "ch_rec_r34_vd_crnn"
serving_client_dir = "serving_client_dir"
serving_server_dir = "serving_server_dir"
feed_var_names, fetch_var_names = inference_model_to_serving(
        inference_model_dir, serving_client_dir, serving_server_dir, model_filename="model", params_filename="params")
```

Finally, model configuration of client and server will be generated in `serving_client_dir` and `serving_server_dir`.

### 3. Start service  
Start the standard version or the fast version service according to your actual needs. The comparison of the two versions is shown in the table below:

|version|characteristics|recommended scenarios|
|-|-|-|
|standard version|High stability, suitable for distributed deployment|Large throughput and cross regional deployment|
|fast version|Easy to deploy and fast to predict|Suitable for scenarios which requires high prediction speed and fast iteration speed|

#### Mode 1. Start the standard mode service

```
# start with CPU
python -m paddle_serving_server.serve --model ocr_det_model --port 9293 
python ocr_web_server.py cpu

# or, with GPU
python -m paddle_serving_server_gpu.serve --model ocr_det_model --port 9293 --gpu_id 0
python ocr_web_server.py gpu
```

#### Mode 2. Start the fast mode service

```
# start with CPU
python ocr_local_server.py cpu

# or, with GPU
python ocr_local_server.py gpu
```

## Send prediction requests

```
python ocr_web_client.py
```

## Returned result format

The returned result is a JSON string, eg.
```
{u'result': {u'res': [u'\u571f\u5730\u6574\u6cbb\u4e0e\u571f\u58e4\u4fee\u590d\u7814\u7a76\u4e2d\u5fc3', u'\u534e\u5357\u519c\u4e1a\u5927\u5b661\u7d20\u56fe']}}
```

You can also print the readable result in `res`:
```
土地整治与土壤修复研究中心
华南农业大学1素图
```

## User defined service module modification

The pre-processing and post-processing process, can be found in the `preprocess` and `postprocess` function in `ocr_web_server.py` or `ocr_local_server.py`. The pre-processing/post-processing library for common CV models provided by `paddle_serving_app` is called.
You can modify the corresponding code as actual needs.

If you only want to start the detection service or the recognition service, execute the corresponding script reffering to the following table. Indicate the CPU or GPU is used in the start command parameters.

| task | standard         | fast           |
| ---- | ----------------- | ------------------- |
| detection | det_web_server.py | det_local_server.py |
| recognition | rec_web_server.py | rec_local_server.py |

More info can be found in [Paddle Serving](https://github.com/PaddlePaddle/Serving).
