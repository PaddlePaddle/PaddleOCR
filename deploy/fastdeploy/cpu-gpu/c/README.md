[English](README.md) | 简体中文
# PaddleOCR CPU-GPU C部署示例

本目录下提供`infer.c`来调用C API快速完成PP-OCRv3模型在CPU/GPU上部署的示例。

## 1. 说明  
PaddleOCR支持利用FastDeploy在NVIDIA GPU、X86 CPU、飞腾CPU、ARM CPU、Intel GPU(独立显卡/集成显卡)硬件上快速部署OCR模型.

## 2. 部署环境准备  
在部署前，需确认软硬件环境，同时下载预编译部署库，参考[FastDeploy安装文档](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/build_and_install#FastDeploy预编译库安装)安装FastDeploy预编译库.
以Linux上推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.4以上(x.x.x>=1.0.4)

## 3. 部署模型准备
在部署前, 请准备好您所需要运行的推理模型, 您可以在[FastDeploy支持的PaddleOCR模型列表](../README.md)中下载所需模型.

## 4.运行部署示例
```bash
# 下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/ocr/PP-OCR/cpu-gpu/c

# 如果您希望从PaddleOCR下载示例代码，请运行
git clone https://github.com/PaddlePaddle/PaddleOCR.git
# 注意：如果当前分支找不到下面的fastdeploy测试代码，请切换到dygraph分支
git checkout dygraph
cd PaddleOCR/deploy/fastdeploy/cpu-gpu/c

mkdir build
cd build

# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz

# 编译Demo
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载PP-OCRv3文字检测模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar
# 下载文字方向分类器模型
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar
# 下载PP-OCRv3文字识别模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xvf ch_PP-OCRv3_rec_infer.tar

# 下载预测图片与字典文件
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg
wget https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/ppocr/utils/ppocr_keys_v1.txt

# 在CPU上使用Paddle Inference推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 0
# 在GPU上使用Paddle Inference推理
./infer_demo ./ch_PP-OCRv3_det_infer ./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv3_rec_infer ./ppocr_keys_v1.txt ./12.jpg 1
```
以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)


运行完成可视化结果如下图所示
<img width="640" src="https://user-images.githubusercontent.com/109218879/185826024-f7593a0c-1bd2-4a60-b76c-15588484fa08.jpg">


## 5. PP-OCRv3 C API接口简介
下面提供了PP-OCRv3的C API简介

- 如果用户想要更换部署后端或进行其他定制化操作, 请查看[C Runtime API](https://baidu-paddle.github.io/fastdeploy-api/c/html/runtime__option_8h.html).
- 更多 PP-OCR C API 请查看 [C PP-OCR API](https://github.com/PaddlePaddle/FastDeploy/blob/develop/c_api/fastdeploy_capi/vision/ocr/ppocr/model.h)

### 配置

```c
FD_C_RuntimeOptionWrapper* FD_C_CreateRuntimeOptionWrapper()
```

> 创建一个RuntimeOption的配置对象，并且返回操作它的指针。
>
> **返回**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption对象的指针


```c
void FD_C_RuntimeOptionWrapperUseCpu(
     FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper)
```

> 开启CPU推理
>
> **参数**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption对象的指针

```c
void FD_C_RuntimeOptionWrapperUseGpu(
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    int gpu_id)
```
> 开启GPU推理
>
> **参数**
>
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption对象的指针
> * **gpu_id**(int): 显卡号


### 模型

```c
FD_C_DBDetectorWrapper* FD_C_CreateDBDetectorWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)
```

> 创建一个DBDetector的模型，并且返回操作它的指针。
>
> **参数**
>
> * **model_file**(const char*): 模型文件路径
> * **params_file**(const char*): 参数文件路径
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption的指针，表示后端推理配置
> * **model_format**(FD_C_ModelFormat): 模型格式
>
> **返回**
> * **fd_c_dbdetector_wrapper**(FD_C_DBDetectorWrapper*): 指向DBDetector模型对象的指针

```c
FD_C_ClassifierWrapper* FD_C_CreateClassifierWrapper(
    const char* model_file, const char* params_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)
```
> 创建一个Classifier的模型，并且返回操作它的指针。
>
> **参数**
>
> * **model_file**(const char*): 模型文件路径
> * **params_file**(const char*): 参数文件路径
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption的指针，表示后端推理配置
> * **model_format**(FD_C_ModelFormat): 模型格式
>
> **返回**
>
> * **fd_c_classifier_wrapper**(FD_C_ClassifierWrapper*): 指向Classifier模型对象的指针

```c
FD_C_RecognizerWrapper* FD_C_CreateRecognizerWrapper(
    const char* model_file, const char* params_file, const char* label_path,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format
)
```
> 创建一个Recognizer的模型，并且返回操作它的指针。
>
> **参数**
>
> * **model_file**(const char*): 模型文件路径
> * **params_file**(const char*): 参数文件路径
> * **label_path**(const char*): 标签文件路径
> * **fd_c_runtime_option_wrapper**(FD_C_RuntimeOptionWrapper*): 指向RuntimeOption的指针，表示后端推理配置
> * **model_format**(FD_C_ModelFormat): 模型格式
>
> **返回**
> * **fd_c_recognizer_wrapper**(FD_C_RecognizerWrapper*): 指向Recognizer模型对象的指针

```c
FD_C_PPOCRv3Wrapper* FD_C_CreatePPOCRv3Wrapper(
    FD_C_DBDetectorWrapper* det_model,
    FD_C_ClassifierWrapper* cls_model,
    FD_C_RecognizerWrapper* rec_model
)
```
> 创建一个PP-OCRv3的模型，并且返回操作它的指针。
>
> **参数**
>
> * **det_model**(FD_C_DBDetectorWrapper*): DBDetector模型
> * **cls_model**(FD_C_ClassifierWrapper*): Classifier模型
> * **rec_model**(FD_C_RecognizerWrapper*): Recognizer模型
>
> **返回**
>
> * **fd_c_ppocrv3_wrapper**(FD_C_PPOCRv3Wrapper*): 指向PP-OCRv3模型对象的指针



### 读写图像

```c
FD_C_Mat FD_C_Imread(const char* imgpath)
```

> 读取一个图像，并且返回cv::Mat的指针。
>
> **参数**
>
> * **imgpath**(const char*): 图像文件路径
>
> **返回**
>
> * **imgmat**(FD_C_Mat): 指向图像数据cv::Mat的指针。


```c
FD_C_Bool FD_C_Imwrite(const char* savepath,  FD_C_Mat img);
```

> 将图像写入文件中。
>
> **参数**
>
> * **savepath**(const char*): 保存图像的路径
> * **img**(FD_C_Mat): 指向图像数据的指针
>
> **返回**
>
> * **result**(FD_C_Bool): 表示操作是否成功


### Predict函数

```c
FD_C_Bool FD_C_PPOCRv3WrapperPredict(
    FD_C_PPOCRv3Wrapper* fd_c_ppocrv3_wrapper,
    FD_C_Mat img,
    FD_C_OCRResult* result)
```
>
> 模型预测接口，输入图像直接并生成结果。
>
> **参数**
> * **fd_c_ppocrv3_wrapper**(FD_C_PPOCRv3Wrapper*): 指向PP-OCRv3模型的指针
> * **img**（FD_C_Mat）: 输入图像的指针，指向cv::Mat对象，可以调用FD_C_Imread读取图像获取
> * **result**(FD_C_OCRResult*): OCR预测结果,包括由检测模型输出的检测框位置,分类模型输出的方向分类,以及识别模型输出的识别结果, OCRResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)


### Predict结果

```c
FD_C_Mat FD_C_VisOcr(FD_C_Mat im, FD_C_OCRResult* ocr_result)
```
>
> 对结果进行可视化，返回可视化的图像。
>
> **参数**
> * **im**(FD_C_Mat): 指向输入图像的指针
> * **ocr_result**(FD_C_OCRResult*): 指向 FD_C_OCRResult结构的指针
>
> **返回**
> * **vis_im**(FD_C_Mat): 指向可视化图像的指针


## 6. 其它文档

- [FastDeploy部署PaddleOCR模型概览](../../)
- [PP-OCRv3 Python部署](../python)
- [PP-OCRv3 C++ 部署](../cpp)
- [PP-OCRv3 C# 部署](../csharp)
