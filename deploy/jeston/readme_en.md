
# Jeston

This section introduces the deployment of PaddleOCR on Jeston NX, TX2, nano, AGX and other series of hardware.


## 1. 环境准备

You need to prepare a Jeston development hardware. If you need TensorRT, you need to prepare the TensorRT environment. It is recommended to use TensorRT version 7.1.3;

1. jeston install paddlepaddle

paddlepaddle download [link](https://www.paddlepaddle.org.cn/inference/user_guides/download_lib.html#python)
Please select the appropriate installation package for your Jetpack version, cuda version, and trt version.

Install paddlepaddle：
```shell
pip3 install -U paddlepaddle_gpu-*-cp36-cp36m-linux_aarch64.whl
```


2. Download PaddleOCR code and install dependencies

Clone the PaddleOCR code:
```
git clone https://github.com/PaddlePaddle/PaddleOCR
```

and install dependencies：
```
cd PaddleOCR
pip3 install -r requirements.txt
```

*Note: Jeston hardware CPU is poor, dependency installation is slow, please wait patiently*

## 2. Perform prediction

Obtain the PPOCR model from the [document](../../doc/doc_en/ppocr_introduction_en.md) model library. The following takes the PP-OCRv3 model as an example to introduce the use of the PPOCR model on jeston:

Download and unzip the PP-OCRv3 models.
```
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar xf ch_PP-OCRv3_det_infer.tar
tar xf ch_PP-OCRv3_rec_infer.tar
```

The text detection inference:
```
cd PaddleOCR
python3 tools/infer/predict_det.py --det_model_dir=./inference/ch_PP-OCRv2_det_infer/  --image_dir=./doc/imgs/  --use_gpu=True
```

The text recognition inference:
```
python3 tools/infer/predict_det.py --rec_model_dir=./inference/ch_PP-OCRv2_rec_infer/  --image_dir=./doc/imgs_words/ch/  --use_gpu=True
```

The text  detection and text recognition inference:

```
python3 tools/infer/predict_system.py --det_model_dir=./inference/ch_PP-OCRv2_det_infer/ --rec_model_dir=./inference/ch_PP-OCRv2_rec_infer/ --image_dir=./doc/imgs/ --use_gpu=True
```

To enable TRT prediction, you only need to set `--use_tensorrt=True` on the basis of the above command:
```
python3 tools/infer/predict_system.py --det_model_dir=./inference/ch_PP-OCRv2_det_infer/ --rec_model_dir=./inference/ch_PP-OCRv2_rec_infer/ --image_dir=./doc/imgs/ --use_gpu=True --use_tensorrt=True
```

For more ppocr model predictions, please refer to[document](../../doc/doc_en/inference_ppocr_en.md)
