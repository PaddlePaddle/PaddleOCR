
# Jeston

本节介绍PaddleOCR在Jeston NX、TX2、nano、AGX等系列硬件的部署。


## 1. 环境准备

需要准备一台Jeston开发板，如果需要TensorRT预测，需准备好TensorRT环境，建议使用7.1.3版本的TensorRT；

1. jeston安装paddlepaddle

paddlepaddle下载[链接](https://www.paddlepaddle.org.cn/inference/user_guides/download_lib.html#python)
请选择适合的您Jetpack版本、cuda版本、trt版本的安装包。

安装命令：
```shell
pip3 install -U paddlepaddle_gpu-*-cp36-cp36m-linux_aarch64.whl
```


2. 下载PaddleOCR代码并安装依赖

首先 clone PaddleOCR 代码：
```
git clone https://github.com/PaddlePaddle/PaddleOCR
```

其次，安装依赖：
```
cd PaddleOCR
pip3 install -r requirements.txt
```

*注：jeston硬件CPU较差，依赖安装较慢，请耐心等待*


## 2. 执行预测

从[文档](../../doc/doc_ch/ppocr_introduction.md) 模型库中获取PPOCR模型，下面以PP-OCRv3模型为例，介绍在PPOCR模型在jeston上的使用方式：

下载并解压PP-OCRv3模型
```
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xf ch_PP-OCRv3_det_infer.tar
tar xf ch_PP-OCRv3_rec_infer.tar
```

执行文本检测预测：
```
cd PaddleOCR
python3 tools/infer/predict_det.py --det_model_dir=./inference/ch_PP-OCRv2_det_infer/  --image_dir=./doc/imgs/  --use_gpu=True
```

执行文本识别预测：
```
python3 tools/infer/predict_det.py --rec_model_dir=./inference/ch_PP-OCRv2_rec_infer/  --image_dir=./doc/imgs_words/ch/  --use_gpu=True
```

执行文本检测+文本识别串联预测：

```
python3 tools/infer/predict_system.py --det_model_dir=./inference/ch_PP-OCRv2_det_infer/ --rec_model_dir=./inference/ch_PP-OCRv2_rec_infer/ --image_dir=./doc/imgs/ --use_gpu=True
```

开启TRT预测只需要在以上命令基础上设置`--use_tensorrt=True`即可：
```
python3 tools/infer/predict_system.py --det_model_dir=./inference/ch_PP-OCRv2_det_infer/ --rec_model_dir=./inference/ch_PP-OCRv2_rec_infer/ --image_dir=./doc/imgs/ --use_gpu=True --use_tensorrt=True
```

更多ppocr模型预测请参考[文档](../../doc/doc_ch/inference_ppocr.md)
