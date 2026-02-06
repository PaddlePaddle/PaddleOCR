---
comments: true
---

# OCR 端侧部署 demo 使用指南

- [快速开始](#快速开始)
    - [环境准备](#环境准备)
    - [部署步骤](#部署步骤)
- [代码介绍](#代码介绍)
- [工程详解](#工程详解)
- [进阶使用](#进阶使用)
    - [更新预测库](#更新预测库) 
    - [转换 nb 模型](#转换-nb-模型) 
    - [更新模型、标签文件和预测图片](#更新模型标签文件和预测图片)
        - [更新模型](#更新模型)
        - [更新标签文件](#更新标签文件)
        - [更新预测图片](#更新预测图片)
    - [更新输入/输出预处理](#更新输入输出预处理)

本指南主要介绍 PaddleX 端侧部署——OCR文字识别 demo 在 Android shell 上的运行方法。

本指南适配了以下 OCR 模型：

- PP-OCRv3_mobile（cpu）
- PP-OCRv4_mobile（cpu）
- PP-OCRv5_mobile（cpu）

## 快速开始

### 环境准备

1. 在本地环境安装好 CMAKE 编译工具，并在 [Android NDK 官网](https://developer.android.google.cn/ndk/downloads)下载当前系统的某个版本的 NDK 软件包。例如，在 Mac 上开发，需要在 Android NDK 官网下载 Mac 平台的 NDK 软件包

    **环境要求**

    -  `CMake >= 3.10`（最低版本未经验证，推荐 3.20 及以上）
    -  `Android NDK >= r17c`（最低版本未经验证，推荐 r20b 及以上）

    **本指南所使用的测试环境：**

    -  `cmake == 3.20.0`
    -  `android-ndk == r20b`

2. 准备一部 Android 手机，并开启 USB 调试模式。开启方法: `手机设置 -> 查找开发者选项 -> 打开开发者选项和 USB 调试模式`

3. 电脑上安装 ADB 工具，用于调试。ADB 安装方式如下：

    3.1. Mac 电脑安装 ADB:

    ```shell
    brew cask install android-platform-tools
    ```

    3.2. Linux 安装 ADB

    ```shell
    sudo apt update
    sudo apt install -y wget adb
    ```

    3.3. Windows 安装 ADB

    win 上安装需要去谷歌的安卓平台下载 ADB 软件包进行安装：[链接](https://developer.android.com/studio)

    打开终端，手机连接电脑，在终端中输入

    ```shell
    adb devices
    ```

    如果有 device 输出，则表示安装成功。

    ```shell
    List of devices attached
    744be294    device
    ```

### 物料准备

1. 克隆 `Paddle-Lite-Demo` 仓库的 `feature/paddle-x` 分支到 `PaddleX-Lite-Deploy` 目录。

    ```shell
    git clone -b feature/paddle-x https://github.com/PaddlePaddle/Paddle-Lite-Demo.git PaddleX-Lite-Deploy
    ```

2. 填写 [问卷](https://paddle.wjx.cn/vm/eaaBo0H.aspx#) 下载压缩包，将压缩包放到指定解压目录，切换到指定解压目录后执行解压命令。

    ```shell
    # 1. 切换到指定解压目录
    cd PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo

    # 2. 执行解压命令
    unzip ocr.zip
    ```

### 部署步骤

1. 将工作目录切换到 `PaddleX-Lite-Deploy/libs` 目录，运行 `download.sh` 脚本，下载需要的 Paddle Lite 预测库。此步骤只需执行一次，即可支持每个 demo 使用。

2. 将工作目录切换到 `PaddleX-Lite-Deploy/ocr/assets` 目录，运行 `download.sh` 脚本，下载 [paddle_lite_opt 工具](https://www.paddlepaddle.org.cn/lite/v2.10/user_guides/model_optimize_tool.html) 优化后的 nb 模型文件及预测图片、字典文件等物料。

3. 将工作目录切换到 `PaddleX-Lite-Deploy/ocr/android/shell/cxx/ppocr_demo` 目录，运行 `build.sh` 脚本，完成可执行文件的编译。

4. 将工作目录切换到 `PaddleX-Lite-Deploy/ocr/android/shell/cxx/ppocr_demo`，运行 `run.sh` 脚本，完成在端侧的预测。

**注意事项：**

- 在运行 `build.sh` 脚本前，需要更改 `NDK_ROOT` 指定的路径为实际安装的 NDK 路径。
- 在 Windows 系统上可以使用 Git Bash 执行部署步骤。
- 若在 Windows 系统上编译，需要将 `CMakeLists.txt` 中的 `CMAKE_SYSTEM_NAME` 设置为 `windows`。
- 若在 Mac 系统上编译，需要将 `CMakeLists.txt` 中的 `CMAKE_SYSTEM_NAME` 设置为 `darwin`。
- 在运行 `run.sh` 脚本时需保持 ADB 连接。
- `download.sh` 和 `run.sh` 支持传入参数来指定模型，若不指定则默认使用 `PP-OCRv5_mobile` 模型。目前适配了以下模型：
    - `PP-OCRv3_mobile`
    - `PP-OCRv4_mobile`
    - `PP-OCRv5_mobile`

以下为实际操作时的示例：

```shell
# 1. 下载需要的 Paddle Lite 预测库
cd PaddleX-Lite-Deploy/libs
sh download.sh

# 2. 下载 paddle_lite_opt 工具优化后的 nb 模型文件及预测图片、字典文件等物料
cd ../ocr/assets
sh download.sh PP-OCRv5_mobile

# 3. 完成可执行文件的编译
cd ../android/shell/ppocr_demo
sh build.sh

# 4. 预测
sh run.sh PP-OCRv5_mobile
```

运行结果如下所示：

```text
The detection visualized image saved in ./test_img_result.jpg
0       纯臻营养护发素  0.998541
1       产品信息/参数   0.999094
2       (45元/每公斤，100公斤起订）     0.948841
3       每瓶22元，1000瓶起订)   0.961245
4       【品牌】：代加工方式/OEMODM     0.970401
5       【品名】：纯臻营养护发素        0.977496
6       ODMOEM  0.955396
7       【产品编号】：YM-X-3011 0.977864
8       【净含量】：220ml       0.970538
9       【适用人群】：适合所有肤质      0.995907
10      【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚    0.975813
11      糖、椰油酰胺丙基甜菜碱、泛醌    0.964397
12      （成品包材）    0.97298
13      【主要功能】：可紧致头发磷层，从而达到  0.989097
14      即时持久改善头发光泽的效果，给干燥的头  0.990088
15      发足够的滋养    0.998037
``` 

![预测结果](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipeline_deploy/edge_PP-OCRv5_mobile.jpg)

## 代码介绍

```
.
├── ...
├── ocr 
│    ├── ...
│    ├── android
│    │    ├── ...
│    │    └── shell
│    │        └── ppocr_demo
│    │            ├── src # 存放预测代码
│    │            |   ├── cls_process.cc # 方向分类器的推理全流程，包含预处理、预测和后处理三部分
│    │            |   ├── rec_process.cc # 识别模型 CRNN 的推理全流程，包含预处理、预测和后处理三部分
│    │            |   ├── det_process.cc # 检测模型 CRNN 的推理全流程，包含预处理、预测和后处理三部分
│    │            |   ├── det_post_process.cc # 检测模型 DB 的后处理文件
│    │            |   ├── pipeline.cc # OCR 文字识别 demo 推理全流程代码
│    │            |   └── MakeFile # 预测代码的 MakeFile 文件
│    │            |   
│    │            ├── CMakeLists.txt # CMake 文件，约束可执行文件的编译方法
│    │            ├── README.md
│    │            ├── build.sh # 用于可执行文件的编译
│    │            └── run.sh # 用于预测
│    └── assets # 存放模型、测试图片、标签文件、config 文件
│        ├── images # 存放测试图片
│        ├── labels # 存放字典文件，更多详情可参考下文备注
│        ├── models # 存放 nb 模型
│        ├── config.txt
│        └── download.sh # 下载脚本，用于下载 paddle_lite_opt 工具优化后的模型
└── libs # 存放不同端的预测库和 OpenCV 库。
    ├── ...
    └── download.sh # 下载脚本，用于下载 Paddle Lite 预测库和 OpenCV 库
```

**备注：**

 - `PaddleX-Lite-Deploy/ocr/assets/labels/` 目录下存放了 PP-OCRv3、PP-OCRv4 模型的字典文件 `ppocr_keys_v1.txt` 以及 PP-OCRv5 模型的字典文件 `ppocr_keys_ocrv5.txt`。在实际推理过程中，会根据模型名称自动选择相应的字典文件，因此无需手动干预。
 - 如果使用的 nb 模型是英文数字或其他语言的模型，需要更换为对应语言的字典。PaddleOCR 仓库提供了[部分字典文件](../../../ppocr/utils)。

```shell
# run.sh 脚本中可执行文件的参数含义：
adb shell "cd ${ppocr_demo_path} \
           && chmod +x ./ppocr_demo \
           && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ppocr_demo \
                \"./models/${MODEL_NAME}_det.nb\" \
                \"./models/${MODEL_NAME}_rec.nb\" \
                ./models/${CLS_MODEL_FILE} \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/${LABEL_FILE} \
                ./config.txt"

第一个参数：ppocr_demo 可执行文件
第二个参数：./models/${MODEL_NAME}_det.nb  检测模型的.nb文件
第三个参数：./models/${MODEL_NAME}_rec.nb  识别模型的.nb文件
第四个参数：./models/${CLS_MODEL_FILE} 文本行方向分类模型的.nb文件，默认根据模型名自动选择
第五个参数：./images/test.jpg  测试图片
第六个参数：./test_img_result.jpg  结果保存文件
第七个参数：./labels/${LABEL_FILE}  label 文件，默认根据模型名自动选择
第八个参数：./config.txt  配置文件，模型的超参数配置文件，包含了检测器、分类器的超参数
```

```shell
# config.txt 具体参数 List：
max_side_len  960         # 输入图像长宽大于 960 时，等比例缩放图像，使得图像最长边为 960
det_db_thresh  0.3        # 用于过滤 DB 预测的二值化图像，设置为 0.3 对结果影响不明显
det_db_box_thresh  0.5    # DB 后处理过滤 box 的阈值，如果检测存在漏框情况，可酌情减小
det_db_unclip_ratio  1.6  # 表示文本框的紧致程度，越小则文本框更靠近文本
use_direction_classify  0  # 是否使用方向分类器，0 表示不使用，1 表示使用
```

## 工程详解

OCR 文字识别 demo 由三个模型一起完成 OCR 文字识别功能，对输入图片先通过 `${MODEL_NAME}_det.nb` 模型做检测处理，然后通过 `ch_ppocr_mobile_v2.0_cls_slim_opt.nb` 模型做文字方向分类处理，最后通过 `${MODEL_NAME}_rec.nb` 模型完成文字识别处理。

1. `pipeline.cc` : OCR 文字识别 demo 预测全流程代码
  该文件完成了三个模型串行推理的全流程控制处理，包含整个处理过程的调度处理。

    - `Pipeline::Pipeline(...)` 方法完成调用三个模型类构造函数，完成模型加载和线程数、绑核处理及 predictor 创建处理
    - `Pipeline::Process(...)` 方法用于完成这三个模型串行推理的全流程控制处理
  
2. `cls_process.cc` 方向分类器的预测文件
  该文件完成了方向分类器的预处理、预测和后处理过程

    - `ClsPredictor::ClsPredictor()`  方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
    - `ClsPredictor::Preprocess()` 方法用于模型的预处理
    - `ClsPredictor::Postprocess()` 方法用于模型的后处理

3. `rec_process.cc` 识别模型 CRNN 的预测文件
  该文件完成了识别模型 CRNN 的预处理、预测和后处理过程

    - `RecPredictor::RecPredictor()`  方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
    - `RecPredictor::Preprocess()` 方法用于模型的预处理
    - `RecPredictor::Postprocess()` 方法用于模型的后处理

4. `det_process.cc` 检测模型 DB 的预测文件
  该文件完成了检测模型 DB 的预处理、预测和后处理过程

    - `DetPredictor::DetPredictor()`  方法用于完成模型加载和线程数、绑核处理及 predictor 创建处理
    - `DetPredictor::Preprocess()` 方法用于模型的预处理
    - `DetPredictor::Postprocess()` 方法用于模型的后处理

5. `db_post_process` 检测模型 DB 的后处理函数，包含 clipper 库的调用
  该文件完成了检测模型 DB 的第三方库调用和其他后处理方法实现

    - `std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(...)` 方法从 Bitmap 图中获取检测框
    - `std::vector<std::vector<std::vector<int>>> FilterTagDetRes(...)` 方法根据识别结果获取目标框位置

## 进阶使用

如果快速开始部分无法满足你的需求，可以参考本节对 demo 进行自定义修改。

本节主要包含四部分： 

- 更新预测库；
- 转换 `.nb` 模型；
- 更新模型、标签文件和预测图片；
- 更新输入/输出预处理。

### 更新预测库

本指南所使用的预测库为最新版本（214rc），不推荐自行更新预测库。

若有使用其他版本的需求，可参考如下步骤更新预测库：

* Paddle Lite 项目：https://github.com/PaddlePaddle/Paddle-Lite
  * 参考 [Paddle Lite 源码编译文档](https://www.paddlepaddle.org.cn/lite/develop/source_compile/compile_env.html)，编译 Android 预测库
  * 编译最终产物位于 `build.lite.xxx.xxx.xxx` 下的 `inference_lite_lib.xxx.xxx`
    * 替换 c++ 库
        * 头文件
          将生成的 `build.lite.android.xxx.gcc/inference_lite_lib.android.xxx/cxx/include` 文件夹替换 demo 中的 `PaddleX-Lite-Deploy/libs/android/cxx/include`
        * armeabi-v7a
          将生成的 `build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/cxx/libs/libpaddle_lite_api_shared.so` 库替换 demo 中的 `PaddleX-Lite-Deploy/libs/android/cxx/libs/armeabi-v7a/libpaddle_lite_api_shared.so`
        * arm64-v8a
          将生成的 `build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/cxx/libs/libpaddle_lite_api_shared.so` 库替换 demo 中的 `PaddleX-Lite-Deploy/libs/android/cxx/libs/arm64-v8a/libpaddle_lite_api_shared.so`

### 转换 .nb 模型

若想使用自己训练的模型，可先参考以下流程得到 `.nb` 模型。

#### 终端命令方法（支持Mac/Ubuntu）

1. 进入Paddle-Lite Github仓库的[release界面](https://github.com/PaddlePaddle/Paddle-Lite/releases)，选择所需版本下载对应的转化工具opt（推荐使用最新版本）。

2. 下载 opt 工具后，执行以下命令（此处以 2.14rc 版本的 linux_x86 opt 工具转换 PP-OCRv5_mobile_det 模型为例）：

    ```bash
    ./opt_linux_x86 \
      --model_file=PP-OCRv5_mobile_det/inference.pdmodel \
      --param_file=PP-OCRv5_mobile_det/inference.pdiparams \
      --optimize_out=PP-OCRv5_mobile_det \
      --valid_targets=arm
    ```

有关使用终端命令方法转换 `.nb` 模型的详细介绍，可参考 Paddle-Lite 仓库的[使用可执行文件 opt](https://www.paddlepaddle.org.cn/lite/v2.12/user_guides/opt/opt_bin.html)。

#### python 脚本方法（支持Windows/Mac/Ubuntu）

1. 安装最新版本的 paddlelite wheel 包。

    ```bash
    pip install --pre paddlelite
    ```

2. 使用 python 脚本进行模型转换。以下为转换 PP-OCRv5_mobile_det 模型的示例代码：

    ```python
    from paddlelite.lite import Opt

    # 1. 创建opt实例
    opt = Opt()
    # 2. 指定输入模型地址 
    opt.set_model_file("./PP-OCRv5_mobile_det/inference.pdmodel")
    opt.set_param_file("./PP-OCRv5_mobile_det/inference.pdiparams")
    # 3. 指定转化类型
    opt.set_valid_places("arm")
    # 4. 指定输出模型地址
    opt.set_optimize_out("./PP-OCRv5_mobile_det")
    # 5. 执行模型优化
    opt.run()
    ```

有关使用 python 脚本方法转换 `.nb` 模型的详细介绍，可参考 Paddle-Lite 仓库的[使用 Python 脚本 opt](https://www.paddlepaddle.org.cn/lite/v2.12/api_reference/python_api/opt.html)。

**注意**

- 有关模型优化工具 opt 的详细介绍，可参考 Paddle-Lite 仓库的[模型优化工具 opt](https://www.paddlepaddle.org.cn/lite/v2.12/user_guides/model_optimize_tool.html)
- 目前仅支持将 `.pdmodel` 格式的静态图模型转换为 `.nb` 格式。在 PaddlePaddle 3.0 以上的版本，默认导出为 `.json` 的模型，如果希望导出为 `.pdmodel` 格式的模型，导出时加上 `-o Global.export_with_pir=False` 即可。

### 更新模型、标签文件和预测图片

#### 更新模型

本指南只对 `PP-OCRv3_mobile`、`PP-OCRv4_mobile`、`PP-OCRv5_mobile` 模型进行了验证，其他模型不保证适用性。

如果你对 `PP-OCRv5_mobile` 模型进行了微调，并生成了一个名为 `PP-OCRv5_mobile_ft` 的新模型，可以按照以下步骤将原有模型替换为你的微调模型：

1. 将 `PP-OCRv5_mobile_ft` 的 nb 模型存放到目录 `PaddleX-Lite-Deploy/ocr/assets/models/` 下，最终得到的文件结构如下：

    ```text
    .
    ├── ocr 
    │    ├── ...
    │    └── assets 
    │        ├── models
    │        │   ├── ...
    │        │   ├── PP-OCRv5_mobile_ft_det.nb 
    │        │   └── PP-OCRv5_mobile_ft_rec.nb 
    │        └── ...
    └── ...
    ```

2. 将模型名加入到 `run.sh` 脚本中的 `MODEL_LIST`。

    ```shell
    MODEL_LIST="PP-OCRv3_mobile PP-OCRv4_mobile PP-OCRv5_mobile PP-OCRv5_mobile_ft" # 模型之间以单个空格为间隔
    ```

3. 运行 `run.sh` 脚本时使用模型目录名。

    ```shell
    sh run.sh PP-OCRv5_mobile_ft
    ```

**注意：**

- 如果更新模型中的输入 Tensor、Shape、和 Dtype 发生更新:

    - 更新文字方向分类器模型，则需要更新 `ppocr_demo/src/cls_process.cc` 中 `ClsPredictor::Preprocss` 函数
    - 更新检测模型，则需要更新 `ppocr_demo/src/det_process.cc` 中 `DetPredictor::Preprocss` 函数
    - 更新识别器模型，则需要更新 `ppocr_demo/src/rec_process.cc` 中 `RecPredictor::Preprocss` 函数

- 如果更新模型中的输出 Tensor 和 Dtype 发生更新:

    - 更新文字方向分类器模型，则需要更新 `ppocr_demo/src/cls_process.cc` 中 `ClsPredictor::Postprocss` 函数
    - 更新检测模型，则需要更新 `ppocr_demo/src/det_process.cc` 中 `DetPredictor::Postprocss` 函数
    - 更新识别器模型，则需要更新 `ppocr_demo/src/rec_process.cc` 中 `RecPredictor::Postprocss` 函数

#### 更新标签文件

如果需要更新标签文件，则需要将新的标签文件存放在目录 `PaddleX-Lite-Deploy/ocr/assets/labels/` 下，并参考模型更新方法更新 `PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo/run.sh` 中执行命令；

以更新 `new_labels.txt` 为例：

  ```shell
  # 代码文件 `PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo/run.sh`
  # old
  adb shell "cd ${ppocr_demo_path} \
            && chmod +x ./ppocr_demo \
            && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
            && ./ppocr_demo \
                  \"./models/${MODEL_NAME}_det.nb\" \
                  \"./models/${MODEL_NAME}_rec.nb\" \
                  ./models/${CLS_MODEL_FILE} \
                  ./images/test.jpg \
                  ./test_img_result.jpg \
                  ./labels/${LABEL_FILE} \
                  ./config.txt"
  # update
  adb shell "cd ${ppocr_demo_path} \
            && chmod +x ./ppocr_demo \
            && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
            && ./ppocr_demo \
                  \"./models/${MODEL_NAME}_det.nb\" \
                  \"./models/${MODEL_NAME}_rec.nb\" \
                  ./models/${CLS_MODEL_FILE} \
                  ./images/test.jpg \
                  ./test_img_result.jpg \
                  ./labels/new_labels.txt \
                  ./config.txt"
  ```

#### 更新预测图片

如果需要更新预测图片，将更新的图片存放在 `PaddleX-Lite-Deploy/ocr/assets/images/` 下，更新文件 `PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo/rush.sh` 中执行命令；

以更新 `new_pics.jpg` 为例：

  ```shell
  # 代码文件 `PaddleX-Lite-Deploy/ocr/assets/images/run.sh`
  ## old
  adb shell "cd ${ppocr_demo_path} \
            && chmod +x ./ppocr_demo \
            && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
            && ./ppocr_demo \
                  \"./models/${MODEL_NAME}_det.nb\" \
                  \"./models/${MODEL_NAME}_rec.nb\" \
                  ./models/${CLS_MODEL_FILE} \
                  ./images/test.jpg \
                  ./test_img_result.jpg \
                  ./labels/${LABEL_FILE} \
                  ./config.txt"
  # update
  adb shell "cd ${ppocr_demo_path} \
            && chmod +x ./ppocr_demo \
            && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
            && ./ppocr_demo \
                  \"./models/${MODEL_NAME}_det.nb\" \
                  \"\"./models/${MODEL_NAME}_rec.nb\"\" \
                  ./models/${CLS_MODEL_FILE} \
                  ./images/new_pics.jpg \
                  ./test_img_result.jpg \
                  ./labels/${LABEL_FILE} \
                  ./config.txt"
  ```

### 更新输入/输出预处理

- 更新输入预处理
    - 更新文字方向分类器模型，则需要更新 `ppocr_demo/src/cls_process.cc` 中 `ClsPredictor::Preprocss` 函数
    - 更新检测模型，则需要更新 `ppocr_demo/src/det_process.cc` 中 `DetPredictor::Preprocss` 函数
    - 更新识别器模型，则需要更新 `ppocr_demo/src/rec_process.cc` 中 `RecPredictor::Preprocss` 函数

- 更新输出预处理
    - 更新文字方向分类器模型，则需要更新 `ppocr_demo/src/cls_process.cc` 中 `ClsPredictor::Postprocss` 函数
    - 更新检测模型，则需要更新 `ppocr_demo/src/det_process.cc` 中 `DetPredictor::Postprocss` 函数
    - 更新识别器模型，则需要更新 `ppocr_demo/src/rec_process.cc` 中 `RecPredictor::Postprocss` 函数
