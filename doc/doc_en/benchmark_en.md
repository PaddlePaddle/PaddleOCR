# BENCHMARK

This document gives the prediction time-consuming benchmark of PaddleOCR Ultra Lightweight Chinese Model (8.6M) on each platform.

## TEST DATA
* 500 images were randomly sampled from the Chinese public data set [ICDAR2017-RCTW](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/datasets.md#ICDAR2017-RCTW-17).
  Most of the pictures in the set were collected in the wild through mobile phone cameras.
  Some are screenshots.
  These pictures show various scenes, including street scenes, posters, menus, indoor scenes and screenshots of mobile applications.

## MEASUREMENT
The predicted time-consuming indicators on the four platforms are as follows:

| Long size(px) | T4(s) | V100(s) | Intel Xeon 6148(s) | Snapdragon 855(s) |
| :---------: | :-----: | :-------: | :------------------: | :-----------------: |
| 960       | 0.092 | 0.057   | 0.319              | 0.354             |
| 640       | 0.067 | 0.045   | 0.198              | 0.236             |
| 480       | 0.057 | 0.043   | 0.151              | 0.175             |

Explanation:
* The evaluation time-consuming stage is the complete stage from image input to result output, including image
pre-processing and post-processing.
* ```Intel Xeon 6148``` is the server-side CPU model. Intel MKL-DNN is used in the test to accelerate the CPU prediction speed.
To use this operation, you need to:
    * Update to the latest version of PaddlePaddle: https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-dev
        Please select the corresponding mkl version wheel package according to the CUDA version and Python version of your environment,
        for example, CUDA10, Python3.7 environment, you should:

    ```
    # Obtain the installation package
    wget https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
    # Installation
    pip3.7 install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
    ```
    * Use parameters ```--enable_mkldnn True``` to turn on the acceleration switch when making predictions
* ```Snapdragon 855``` is a mobile processing platform model.
