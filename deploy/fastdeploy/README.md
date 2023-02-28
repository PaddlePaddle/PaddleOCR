# PaddleOCR高性能全场景模型部署方案—FastDeploy

## 目录  
- [FastDeploy介绍](#FastDeploy介绍)  
- [PaddleOCR模型部署](#PaddleOCR模型部署)  
- [常见问题](#常见问题)  

## 1. FastDeploy介绍
<div id="FastDeploy介绍"></div>  

**[⚡️FastDeploy](https://github.com/PaddlePaddle/FastDeploy)**是一款**全场景**、**易用灵活**、**极致高效**的AI推理部署工具，支持**云边端**部署.使用FastDeploy可以简单高效的在X86 CPU、NVIDIA GPU、飞腾CPU、ARM CPU、Intel GPU、昆仑、昇腾、算能等10+款硬件上对PaddleOCR模型进行快速部署，并且支持Paddle Inference、Paddle Lite、TensorRT、OpenVINO、ONNXRuntime、SOPHGO等多种推理后端.

<div align="center">

<img src="https://user-images.githubusercontent.com/31974251/219546373-c02f24b7-2222-4ad4-9b43-42b8122b898f.png" >

</div>  

## 2. PaddleOCR模型部署
<div id="PaddleOCR模型部署"></div>  

### 2.1 硬件支持列表

|硬件类型|该硬件是否支持|使用指南|Python|C++|
|:---:|:---:|:---:|:---:|:---:|
|X86 CPU|✅|[链接](./cpu-gpu)|✅|✅|
|NVIDIA GPU|✅|[链接](./cpu-gpu)|✅|✅|
|飞腾CPU|✅|[链接](./cpu-gpu)|✅|✅|
|ARM CPU|✅|[链接](./cpu-gpu)|✅|✅|
|Intel GPU(集成显卡)|✅|[链接](./cpu-gpu)|✅|✅|  
|Intel GPU(独立显卡)|✅|[链接](./cpu-gpu)|✅|✅|  
|昆仑|✅|[链接](./kunlun)|✅|✅|
|昇腾|✅|[链接](./ascend)|✅|✅|
|算能|✅|[链接](./sophgo)|✅|✅|  

### 2.2. 详细使用文档
- X86 CPU
  - [部署模型准备](./cpu-gpu)  
  - [Python部署示例](./cpu-gpu/python/)
  - [C++部署示例](./cpu-gpu/cpp/)
- NVIDIA GPU
  - [部署模型准备](./cpu-gpu)  
  - [Python部署示例](./cpu-gpu/python/)
  - [C++部署示例](./cpu-gpu/cpp/)
- 飞腾CPU
  - [部署模型准备](./cpu-gpu)  
  - [Python部署示例](./cpu-gpu/python/)
  - [C++部署示例](./cpu-gpu/cpp/)
- ARM CPU
  - [部署模型准备](./cpu-gpu)  
  - [Python部署示例](./cpu-gpu/python/)
  - [C++部署示例](./cpu-gpu/cpp/)
- Intel GPU
  - [部署模型准备](./cpu-gpu)  
  - [Python部署示例](./cpu-gpu/python/)
  - [C++部署示例](./cpu-gpu/cpp/)
- 昆仑 XPU
  - [部署模型准备](./kunlun)  
  - [Python部署示例](./kunlun/python/)
  - [C++部署示例](./kunlun/cpp/)
- 昇腾 Ascend
  - [部署模型准备](./ascend)  
  - [Python部署示例](./ascend/python/)
  - [C++部署示例](./ascend/cpp/)  
- 算能 Sophgo
  - [部署模型准备](./sophgo/)  
  - [Python部署示例](./sophgo/python/)
  - [C++部署示例](./sophgo/cpp/)  

### 2.3 更多部署方式

- [Android ARM CPU部署](./android)  
- [服务化Serving部署](./serving)  
- [web部署](./web)


## 3. 常见问题
<div id="常见问题"></div>  

遇到问题可查看常见问题集合，搜索FastDeploy issue，*或给FastDeploy提交[issue](https://github.com/PaddlePaddle/FastDeploy/issues)*:

[常见问题集合](https://github.com/PaddlePaddle/FastDeploy/tree/develop/docs/cn/faq)  
[FastDeploy issues](https://github.com/PaddlePaddle/FastDeploy/issues)  