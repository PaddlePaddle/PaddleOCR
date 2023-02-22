[English](README.md) | 简体中文
# PaddleOCR CPU-GPU C++部署示例

本目录下提供`infer.cc`快速完成PP-OCRv3在CPU/GPU，以及GPU上通过Paddle-TensorRT加速部署的示例。

## 1. 说明  
PaddleOCR支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署OCR模型。

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库。

## 3. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleOCR模型列表](../README.md)中下载所需模型.

## 4. 运行部署示例
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.0以上(x.x.x>=1.0.0)

```bash  
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz

# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/fastdeploy/cpu-gpu/cpp

# 编译部署示例
mkdir build && cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

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
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# GPU推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
# GPU上TensorRT推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 2
```

运行完成可视化结果如下图所示
<div  align="center">  
<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">
</div>

- 注意，以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考文档: [如何在Windows中使用FastDeploy C++ SDK](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_sdk_on_windows.md)  
- 关于如何通过FastDeploy使用更多不同的推理后端，以及如何使用不同的硬件，请参考文档：[如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)

## 6. 更多指南
- [PP-OCR系列 C++ API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1ocr.html)
- [FastDeploy部署PaddleOCR模型概览](../../)
- [PPOCRv3 Python部署](../python)

## 7. 常见问题
- [如何切换模型推理后端引擎](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/how_to_change_backend.md)
- [Intel GPU(独立显卡/集成显卡)的使用](https://github.com/PaddlePaddle/FastDeploy/blob/develop/tutorials/intel_gpu/README.md)
- [编译CPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/cpu.md)
- [编译GPU部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/gpu.md)
- [编译Jetson部署库](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install/jetson.md)
