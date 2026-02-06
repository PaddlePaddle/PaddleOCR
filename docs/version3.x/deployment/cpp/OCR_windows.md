# 通用 OCR 产线 C++ 部署 - Windows

- [1. 环境准备](#1-环境准备)
    - [1.1 编译 OpenCV 库](#11-编译-opencv-库)
    - [1.2 编译 Paddle Inference](#12-编译-paddle-inference)
- [2. 开始运行](#2-开始运行)
    - [2.1 编译预测 demo](#21-编译预测-demo)
    - [2.2 准备模型](#22-准备模型)
    - [2.3 运行预测 demo](#23-运行预测-demo)
    - [2.4 C++ API 集成](#24-c-api-集成)
- [3. 拓展功能](#3-拓展功能)
    - [3.1 多语种文字识别](#31-多语种文字识别)  
    - [3.2 可视化文本识别结果](#32-可视化文本识别结果)  

## 1. 环境准备

- **本章节编译运行时用到的源代码位于 [PaddleOCR/deploy/cpp_infer](https://github.com/PaddlePaddle/PaddleOCR/tree/main/deploy/cpp_infer) 目录下。**

- Windows 环境：
    - visual studio 2022
    - cmake 3.29

### 1.1 编译 OpenCV 库

可以选择直接下载预编译包或者手动编译源码。

#### 1.1.1 直接下载预编译包（推荐）

在 [OpenCV 官网](https://opencv.org/releases/) 下载适用于 Windows 的 .exe 预编译包，运行后自动解压出 OpenCV 的预编译库和相关文件夹。

以 opencv 4.7.0为例，下载 [opencv-4.7.0-windows.exe](https://github.com/opencv/opencv/releases/download/4.7.0/opencv-4.7.0-windows.exe)，运行后会在当前的文件夹中生成 `opencv/` 的子文件夹，其中 `opencv/build` 为预编译库，在后续编译通用 OCR 产线 预测 demo 时，将作为 OpenCV 安装库的路径使用。

#### 1.1.2 源码编译

首先需要下载 OpenCV 源码，以 opencv 4.7.0 为例，下载 [opencv 4.7.0](https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv-4.7.0.tgz) 源码，解压后会在当前的文件夹中生成 `opencv-4.7.0/` 的文件夹。

- Step 1：构建 Visual Studio 项目

  在 cmake-gui 中指定 `opencv-4.7.0` 源码路径，并指定编译生成目录为 `opencv-4.7.0/build`，默认安装路径为 `opencv-4.7.0/build/install`，此安装路径用于后续编译 demo。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/opencv_step1.png"/>

- Step 2：选择目标平台

  选择目标平台为 x64 ，然后点击 finish。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/opencv_step2.png"/>

- Step 3 ：生成 Visual Studio 项目

  搜索 `BUILD_opencv_world` 并勾选。
  依次点击 Configure  ->  Generate  ->  Open in Project，将进入 Visual Studio 2022 编译界面。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/opencv_step3.png"/>

- Step 4：执行编译

  点击开始生成解决方案，完成编译后，点击 INSTALL，运行后完成安装。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/opencv_step4.png"/>

### 1.2 编译 Paddle Inference

可以选择直接下载预编译包或者手动编译源码。

#### 1.2.1 直接下载预编译包（推荐）

[Paddle Inference 官网](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/download_lib.html#windows) 上提供了 Windows 预测库，可以在官网查看并选择合适的预编译包。

下载后解压，会在当前的文件夹中生成 `paddle_inference/` 的子文件夹。目录结构为：

```
paddle_inference
├── paddle # paddle核心库和头文件
├── third_party # 第三方依赖库和头文件
└── version.txt # 版本和编译信息
```

#### 1.2.2 源码编译预测库

可以选择通过源码自行编译预测库，源码编译可灵活配置各类功能和依赖，以适应不同的硬件和软件环境。详细步骤请参考 [Windows 下源码编译](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/compile/source_compile_under_Windows.html)。

## 2. 开始运行

### 2.1 编译预测 demo

在编译预测 demo 前，请确保您已经按照 1.1 和 1.2 节编译好 OpenCV 库和 Paddle Inference 预测库。

编译步骤如下：

- Step 1：构建 Visual Studio 项目

  在 cmake-gui 中指定 `deploy\cpp_infer` 源码路径，并指定编译生成目录为 `deploy\cpp_infer\build`，以下编译步骤说明均以 `D:\PaddleOCR\deploy\cpp_infer` 作为示例源码路径。第一次点击 Configure 报错是正常的，在后续弹出的编译选项中，添加 OpenCV 的安装路径和 Paddle Inference 预测库路径。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/windows_step1.png"/>

- Step 2：选择目标平台

  选择目标平台为 x64 ，然后点击 Finish。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/windows_step2.png"/>

- Step 3：配置 cmake 编译选项

    - OPENCV_DIR：填写 OpenCV 安装路径。
    - OpenCV_DIR：同 OPENCV_DIR。
    - PADDLE_LIB：Paddle Inference 预测库路径。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/windows_step3.png"/>

- Step 4：生成 Visual Studio 项目

  依次点击 Configure  ->  Generate  ->  Open in Project，将进入 Visual Studio 2022 编译界面。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/windows_step4.png"/>

- Step 5：执行编译

  在开始生成解决方案之前，执行下面步骤：

  1. 将 `Debug` 改为 `Release`。
  2. 下载[dirent.h](https://paddleocr.bj.bcebos.com/deploy/cpp_infer/cpp_files/dirent.h)，并拷贝到  Visual Studio 的 include 文件夹下，如 `C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include`。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/windows_step5.png"/>

  编译完成后，可执行文件位于 `deploy/cpp_infer/build/Release/ppocr.exe` 。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/windows_step6.png"/>

- Step 6：运行预测 demo

  将下面文件拷贝到 `deploy\cpp_infer\build\Release\` 路径下后，参考后续 2.2 和 2.3 小节即可运行预测 demo。

  1. `paddle_inference\paddle\lib\paddle_inference.dll`
  2. `paddle_inference\paddle\lib\common.dll`
  3. `deploy\cpp_infer\build\bin\Release\abseil_dll.dll`
  4. `deploy\cpp_infer\build\third_party\clipper_ver6.4.2\cpp\Release\polyclipping.dll`
  5. `opencv-4.7.0\build\install\x64\vc16\bin\opencv_world470.dll`

### 2.2 准备模型

该步骤参考 [通用 OCR 产线 C++ 部署 - Linux —— 2.2 准备模型](./OCR.md#22-准备模型) 小节。

### 2.3 运行预测 demo

参考 [通用 OCR 产线 C++ 部署 - Linux —— 2.3 运行预测 demo](./OCR.md#23-运行预测-demo) 小节。

### 2.4 C++ API 集成

参考 [通用 OCR 产线 C++ 部署 - Linux —— 2.4 C++ API 集成](./OCR.md#24-c-api-集成) 小节。

## 3. 拓展功能

### 3.1 多语种文字识别

参考 [通用 OCR 产线 C++ 部署 - Linux —— 3.1 多语种文字识别](./OCR.md#31-多语种文字识别) 小节。

### 3.2 可视化文本识别结果

我们使用 4.x 版本的 opencv_contrib 模块中的 FreeType 进行字体渲染，如果想要可视化文本识别结果，需要下载 OpenCV 和 opencv_contrib 的源码并编译包含 FreeType 模块的 OpenCV。下载源码时需确保两者的版本一致。以下以 opencv-4.7.0 和 opencv_contrib-4.7.0 为例进行说明：

[下载 opencv-4.7.0](https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv-4.7.0.tgz)
[下载 opencv_contrib-4.7.0](https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv_contrib-4.7.0.tgz)

- Step 1：编译 freetype 和 harfbuzz

    - [下载pkg-config](https://sourceforge.net/projects/pkgconfiglite/)
    - [下载freetype2](https://download.savannah.gnu.org/releases/freetype/)
    - [下载harfbuzz](https://github.com/harfbuzz/harfbuzz)

  解压 pkg-config 后添加其 bin 目录到系统 PATH 环境变量。
  freetype 编译，需手动更改其安装路径，示例如下：

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/opencv_freetype_step1.png"/>

  再次 Configure， 然后点击 Generate， 完成后，点击 Open Project 按钮，打开 VS ，编译。
  VS里ALL_BUILD, INSTALL. 会在构建文件夹的 install 目录下生成所需的 include 和 lib 文件。

  然后将 freetype 安装路径添加至系统环境变量。

  harfbuzz 编译，需手动更改其安装路径，示例如下：

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/opencv_freetype_step2.png"/>

  设置好上面两项后，再次点击 Configure 按钮，选择 Advanced Options ，填写 freetype 安装路径。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/opencv_freetype_step3.png"/>

  然后将 harfbuzz 安装路径添加至系统环境变量。

- Step 2：修改 opencv_contrib-4.7.0 下的 `modules/freetype/CMakeLists.txt`

  ```bash
  set(the_description "FreeType module. It enables to draw strings with outlines and mono-bitmaps/gray-bitmaps.")
  
  find_package(Freetype REQUIRED)
  
  # find_package(HarfBuzz) is not included in cmake
  set(HARFBUZZ_DIR "$ENV{HARFBUZZ_DIR}" CACHE PATH "HarfBuzz directory")
  find_path(HARFBUZZ_INCLUDE_DIRS
      NAMES hb-ft.h PATH_SUFFIXES harfbuzz
      HINTS ${HARFBUZZ_DIR}/include)
  find_library(HARFBUZZ_LIBRARIES
      NAMES harfbuzz
      HINTS ${HARFBUZZ_DIR}/lib)
  find_package_handle_standard_args(HARFBUZZ
      DEFAULT_MSG HARFBUZZ_LIBRARIES HARFBUZZ_INCLUDE_DIRS)
  
  if(NOT FREETYPE_FOUND)
    message(STATUS "freetype2:   NO")
  else()
    message(STATUS "freetype2:   YES")
  endif()
  
  if(NOT HARFBUZZ_FOUND)
    message(STATUS "harfbuzz:   NO")
  else()
    message(STATUS "harfbuzz:   YES")
  endif()
  
  if(FREETYPE_FOUND AND HARFBUZZ_FOUND)
    ocv_define_module(freetype opencv_core opencv_imgproc PRIVATE_REQUIRED ${FREETYPE_LIBRARIES} ${HARFBUZZ_LIBRARIES} WRAP python)
    ocv_include_directories(${FREETYPE_INCLUDE_DIRS} ${HARFBUZZ_INCLUDE_DIRS})
  else()
    ocv_module_disable(freetype)
  endif()
  ```

- Step 3 编译 OpenCV

  1. 设置 `OPENCV_EXTRA_MODULES_PATH` 项，填入 opencv-contrib-4.7.0 的目录下的 modules 目录。
  2. 勾选 `WITH_FREETYPE` 项，必须先编译 freetype 和 harfbuzz。
  3. 如果需要支持 freetype，则需要在 Opencv 的 Cmake 配置中加入 freetype 的相关路径。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/opencv_freetype_step4.png"/>

  搜索 harfbuzz，加入 harfbuzz，加入 的相关路径。

  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/opencv_freetype_step5.png"/>

  完成后，再次在 Cmake 界面，点击 configure， 确定没报错后，点击 Generate，最后点击 Open Project，打开 Visual studio，将 Debug 切换为 Release，找到 ALL_BUILD 右键 Build， 等待编译完成后， 找到 INSTALL 右键 Build。

  注意：如果完成编译包含 FreeType 的 OpenCV，在编译通用 OCR 产线 demo 时，需要在 2.1节 Step 3 配置编译选项时勾选 `USE_FREETYPE` 开启文字渲染功能，并且在运行 demo 时通过 `--vis_font_dir your_ttf_path` 提供相应 ttf 字体文件路径。

编译并运行预测 demo 可以得到如下可视化文本识别结果：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/ocr_res_with_freetype.png"/>
