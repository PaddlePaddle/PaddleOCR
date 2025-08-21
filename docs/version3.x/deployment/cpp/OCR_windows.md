# 通用 OCR 产线 C++ 部署 - Windows


- [1. 环境准备](#1)
    - [1.1 下载 OpenCV 预编译包或者手动编译源码](#12)
    - [1.2 下载或者编译Paddle预测库](#13)

## 1. 准备环境

- Windows 环境：
    - visual studio 2022
    - cmake 3.29

### 1.1 下载 OpenCV 预编译包或者手动编译源码

可以选择直接下载 Opencv 官网提供的预编译包或者手动编译源码，下文分别进行具体说明。

#### 1.1.1 直接下载预编译包（推荐）

在 [OpenCV 官网](https://opencv.org/releases/) 下载适用于 Windows 的 .exe 预编译包，运行后自动解压出 OpenCV 的预编译库和相关文件夹。

以 opencv 4.7.0为例，[opencv-4.7.0-windows.exe下载地址](https://github.com/opencv/opencv/releases/download/4.7.0/opencv-4.7.0-windows.exe)下载解压后，会在当前的文件夹中生成 `opencv/` 的子文件夹，其中 `opencv//build` 为预编译库，在后续编译通用 OCR 产线 demo 时，将作为 OpenCV 库的路径使用。


#### 1.1.1 源码编译

- 首先需要下载 OpenCV 源码，以 opencv 4.7.0为例，[opencv 4.7.0下载地址](https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv-4.7.0.tgz)，下载解压后，会在当前的文件夹中生成 `opencv-4.7.0/` 的子文件夹。

打开 cmake-gui 程序



### 1.2 下载 Paddle Inference C++ 预编译包或者手动编译源码

可以选择直接下载 Paddle Inference官网提供的预编译包或者手动编译源码，下文分别进行具体说明。

#### 1.2.1 直接下载预编译包（推荐）

[Paddle Inference 官网](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/download_lib.html#windows) 上提供了 Windows 预测库，可以在官网查看并选择合适的预编译包（*建议选择 paddle 版本>=3.0.0版本的预测库* ）。

下载解压后，会在当前的文件夹中生成 `paddle_inference/` 的子文件夹。`目录结构为：

```
paddle_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

#### 1.2.2 预测库源码编译

可以选择通过源码自行编译预测库，源码编译可灵活配置各类功能和依赖，以适应不同的硬件和软件环境。详细步骤请参考 [Windows 下源码编译](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/compile/source_compile_under_Windows.html)。

### 2.2 编译预测 demo

在编译通用 OCR C++ 预测 demo 前，请确保您已经编译好 OpenCV 和 Paddle Inference 预测库。

编译步骤如下：

- step1

在 cmake-gui 中指定 `deploy/cpp_infer` 源码路径，并指定编译生成目录为 `deploy/cpp_infer/build`，以下编译步骤说明均以 `D:/PaddleOCR/deploy/cpp_infer` 作为示例源码路径。第一次点击 Configure 报错是正常的，在后续弹出的编译选项中，添加 OpenCV 的安装路径和 Paddle Inference 预测库路径。

<img src="./imgs/cpp_infer_demo_step1.png"/>

- step2 

<img src="./imgs/cpp_infer_demo_step2.png"/>
