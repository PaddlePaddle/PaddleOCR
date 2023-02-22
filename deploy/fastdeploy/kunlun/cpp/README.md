[English](README.md) | 简体中文
# PPOCRv3 昆仑芯XPU C++部署示例

本目录下提供`infer.cc`, 供用户完成PPOCRv3在昆仑芯XPU上的部署。

## 1. 部署环境准备
在部署前，需自行编译基于昆仑芯XPU的预测库，参考文档[昆仑芯XPU部署环境编译安装](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#自行编译安装)

## 2.部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleOCR模型列表](../README.md)中下载所需模型.

## 3.运行部署示例
```
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR/deploy/fastdeploy/PP-OCRv3/kunlunxin/cpp

mkdir build
cd build
# 使用编译完成的FastDeploy库编译infer_demo
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-kunlunxin
make -j

# 下载模型,图片和字典文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xvf ch_PP-OCRv3_rec_infer.tar

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg

```

运行完成可视化结果如下图所示

<div  align="center">  
<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">
</div>

## 4. 更多指南
- [PP-OCR系列 C++ API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1ocr.html)
- [FastDeploy部署PaddleOCR模型概览](../../)
- [PPOCRv3 Python部署](../python)
