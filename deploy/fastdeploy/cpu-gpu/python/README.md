[English](README.md) | 简体中文  
# PaddleOCR CPU-GPU Python部署示例
本目录下提供`infer.py`快速完成PP-OCRv3在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例。执行如下脚本即可完成

## 1. 说明  
PaddleOCR支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署OCR模型

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 3. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleOCR模型列表](../README.md)中下载所需模型.

## 4. 运行部署示例
```bash
# 安装FastDpeloy python包（详细文档请参考`部署环境准备`）
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/fastdeploy/cpu-gpu/python

# 下载PP-OCRv3模型,图片和字典文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xvf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# 运行部署示例
# CPU推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device cpu
# GPU推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu
# GPU上使用TensorRT推理
python infer.py --det_model ch_PP-OCRv3_det_infer --cls_model ch_ppocr_mobile_v2.0_cls_infer --rec_model ch_PP-OCRv3_rec_infer --rec_label_file ppocr_keys_v1.txt --image 12.jpg --device gpu --backend trt
```

运行完成可视化结果如下图所示
<div  align="center">  
<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">
</div>


## 5. 部署示例选项说明  

|参数|含义|默认值
|---|---|---|  
|--det_model|指定检测模型文件夹所在的路径|None|
|--cls_model|指定分类模型文件夹所在的路径|None|
|--rec_model|指定识别模型文件夹所在的路径|None|
|--rec_label_file|识别模型所需label所在的路径|None|
|--image|指定测试图片所在的路径|None|  
|--device|指定即将运行的硬件类型，支持的值为`[cpu, gpu]`，当设置为cpu时，可运行在x86 cpu/arm cpu等cpu上|cpu|
|--use_trt|是否使用trt，该项只在device为gpu时有效|False|  

关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

## 6. 更多指南
- [PP-OCR系列 Python API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/ocr.html)
- [FastDeploy部署PaddleOCR模型概览](../../)
- [PPOCRv3 C++部署](../cpp)

## 7. 常见问题
- [如何将模型预测结果转为numpy格式](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/vision_result_related_problems.md)
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)
