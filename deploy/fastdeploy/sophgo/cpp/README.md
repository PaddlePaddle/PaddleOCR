[English](README_CN.md) | 简体中文
# PP-OCRv3 SOPHGO C++部署示例
本目录下提供`infer.cc`快速完成PPOCRv3模型在SOPHGO BM1684x板子上加速部署的示例。

## 1. 部署环境准备
在部署前，需自行编译基于SOPHGO硬件的预测库，参考文档[SOPHGO硬件部署环境](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#算能硬件部署环境)

## 2. 生成基本目录文件

该例程由以下几个部分组成
```text
.
├── CMakeLists.txt
├── fastdeploy-sophgo  # 编译好的SDK文件夹
├── image  # 存放图片的文件夹
├── infer.cc
└── model  # 存放模型文件的文件夹
```

## 3.部署示例

### 3.1 下载部署示例代码
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/ocr/PP-OCR/sophgo/cpp

# 如果您希望从PaddleOCR下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleOCR.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到dygraph分支
git checkout dygraph
cd PaddleOCR/deploy/fastdeploy/sophgo/cpp
```

### 3.2 拷贝bmodel模型文至model文件夹
将Paddle模型转换为SOPHGO bmodel模型，转换步骤参考[文档](../README.md). 将转换后的SOPHGO bmodel模型文件拷贝至model中.

### 3.3 准备测试图片至image文件夹，以及字典文件
```bash
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
cp 12.jpg image/

wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt
```

### 3.4 编译example

```bash
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-0.0.3
make
```

### 3.5 运行例程

```bash
./infer_demo model ./ppocr_keys_v1.txt image/12.jpeg
```


## 4. 更多指南

- [PP-OCR系列 C++ API查阅](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/namespacefastdeploy_1_1vision_1_1ocr.html)
- [FastDeploy部署PaddleOCR模型概览](../../)
- [PP-OCRv3 Python部署](../python)
- 如果用户想要调整前后处理超参数、单独使用文字检测识别模型、使用其他模型等，更多详细文档与说明请参考[PP-OCR系列在CPU/GPU上的部署](../../cpu-gpu/cpp/README.md)
