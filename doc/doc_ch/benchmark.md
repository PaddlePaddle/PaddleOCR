# Benchmark

本文给出了PaddleOCR超轻量中文模型（8.6M）在各平台的预测耗时benchmark。

## 测试数据  
- 从中文公开数据集[ICDAR2017-RCTW](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/datasets.md#ICDAR2017-RCTW-17)中随机采样**500**张图像。  
该集合大部分图片是通过手机摄像头在野外采集的。有些是截图。这些图片展示了各种各样的场景，包括街景、海报、菜单、室内场景和手机应用程序的截图。

## 评估指标  
在四种平台上的预测耗时指标如下：  

|长边尺寸(px)|T4(s)|V100(s)|Intel至强6148(s)|骁龙855(s)|
|-|-|-|-|-|
|960|0.092|0.057|0.319|0.354|
|640|0.067|0.045|0.198| 0.236|
|480|0.057|0.043|0.151| 0.175| 

说明： 
- 评估耗时阶段为图像输入到结果输出的完整阶段，包括了图像的预处理和后处理。  
- `Intel至强6148`为服务器端CPU型号，测试中使用Intel MKL-DNN 加速CPU预测速度，使用该操作需要：  
    - 更新到飞桨latest版本：https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html#whl-dev ，请根据自己环境的CUDA版本和Python版本选择相应的mkl版wheel包，如，CUDA10、Python3.7环境，应操作：
    ```shell
    # 获取安装包
    wget https://paddle-wheel.bj.bcebos.com/0.0.0-gpu-cuda10-cudnn7-mkl/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
    # 安装
    pip3.7 install paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl
    ```
    - 预测时使用参数打开加速开关： `--enable_mkldnn True`  
- `骁龙855`为移动端处理平台型号。  
