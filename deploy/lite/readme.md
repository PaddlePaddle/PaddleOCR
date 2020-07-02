# PaddleOCR 模型部署

PaddleOCR是集训练、预测、部署于一体的实用OCR工具库。本教程将介绍在安卓移动端部署PaddleOCR超轻量中文检测、识别模型的主要流程。


## 1. 准备环境

### 运行准备
- 电脑（编译Paddle-Lite）
- 安卓手机（armv7或armv8）

### 1.1 准备交叉编译环境
交叉编译环境用于编译[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)和PaddleOCR的C++ demo。
支持多种开发环境，不同开发环境的编译流程请参考对应文档。：
1. [Docker](https://paddle-lite.readthedocs.io/zh/latest/user_guides/source_compile.html#docker)
2. [Linux](https://paddle-lite.readthedocs.io/zh/latest/user_guides/source_compile.html#android)
3. [MAC OS](https://paddle-lite.readthedocs.io/zh/latest/user_guides/source_compile.html#id13)
4. [Windows](https://paddle-lite.readthedocs.io/zh/latest/demo_guides/x86.html#windows)

### 1.2 准备预编译库

预编译库有两种获取方式：
- 1. 直接下载，下载[链接](https://paddle-lite.readthedocs.io/zh/latest/user_guides/release_lib.html#android-toolchain-gcc).
    注意选择`with_extra=ON，with_cv=ON`的下载链接。
- 2. 编译Paddle-Lite得到，Paddle-Lite的编译方式如下：
```
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git checkout release/2.6.0
./lite/tools/build_android.sh  --arch=armv8  --with_cv=ON --with_extra=ON
```

注意：编译Paddle-Lite获得预编译库时，需要打开`--with_cv=ON --with_extra=ON`两个选项，`--arch`表示`arm`版本，这里指定为armv8，
更多编译命令
介绍请参考[链接](https://paddle-lite.readthedocs.io/zh/latest/user_guides/Compile/Android.html#id2)。

直接下载预编译库并解压后，可以得到`inference_lite_lib.android.armv8/`文件夹，通过编译Paddle-Lite得到的预编译库位于
`Paddle-Lite/build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/`文件夹下。
预编译库的文件目录如下：
```
inference_lite_lib.android.armv8/
|-- cxx                           C++ 预测库和头文件
|   |-- include                                C++ 头文件
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib                                    C++预测库
|       |-- libpaddle_api_light_bundled.a             C++静态库
|       `-- libpaddle_light_api_shared.so             C++动态库
|-- java                          Java预测库
|   |-- jar
|   |   `-- PaddlePredictor.jar
|   |-- so
|   |   `-- libpaddle_lite_jni.so
|   `-- src
|-- demo                          C++和Java示例代码
|   |-- cxx                                  C++  预测库demo
|   `-- java                                 Java 预测库demo
```

## 2 开始运行

### 2.1 模型优化

Paddle-Lite 提供了多种策略来自动优化原始的模型，其中包括量化、子图融合、混合调度、Kernel优选等方法，使用Paddle-lite的opt工具可以自动
对inference模型进行优化，优化后的模型更轻量，模型运行速度更快。

下述表格中提供了优化好的超轻量中文模型：

|模型简介|检测模型|识别模型|
|-|-|-|
|超轻量级中文OCR opt优化模型|[下载地址](https://paddleocr.bj.bcebos.com/deploy/lite/ch_det_mv3_db_opt.nb)|[下载地址](https://paddleocr.bj.bcebos.com/deploy/lite/ch_rec_mv3_crnn_opt.nb)|

如果直接使用上述表格中的模型进行部署，可略过下述步骤，直接阅读 [2.2节](###2.2与手机联调)。

如果要部署的模型不在上述表格中，则需要按照如下步骤获得优化后的模型。

模型优化需要Paddle-Lite的opt可执行文件，可以通过编译Paddle-Lite源码获得，编译步骤如下：
```
# 如果准备环境时已经clone了Paddle-Lite，则不用重新clone Paddle-Lite
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
# 启动编译
./lite/tools/build.sh build_optimize_tool
```

编译完成后，opt文件位于`build.opt/lite/api/`下，可通过如下方式查看opt的运行选项和使用方式；
```
cd build.opt/lite/api/
./opt
```

|选项|说明|
|-|-|
|--model_dir|待优化的PaddlePaddle模型（非combined形式）的路径|
|--model_file|待优化的PaddlePaddle模型（combined形式）的网络结构文件路径|
|--param_file|待优化的PaddlePaddle模型（combined形式）的权重文件路径|
|--optimize_out_type|输出模型类型，目前支持两种类型：protobuf和naive_buffer，其中naive_buffer是一种更轻量级的序列化/反序列化实现。若您需要在mobile端执行模型预测，请将此选项设置为naive_buffer。默认为protobuf|
|--optimize_out|优化模型的输出路径|
|--valid_targets|指定模型可执行的backend，默认为arm。目前可支持x86、arm、opencl、npu、xpu，可以同时指定多个backend(以空格分隔)，Model Optimize Tool将会自动选择最佳方式。如果需要支持华为NPU（Kirin 810/990 Soc搭载的达芬奇架构NPU），应当设置为npu, arm|
|--record_tailoring_info|当使用 根据模型裁剪库文件 功能时，则设置该选项为true，以记录优化后模型含有的kernel和OP信息，默认为false|

`--model_dir`适用于待优化的模型是非combined方式，PaddleOCR的inference模型是combined方式，即模型结构和模型参数使用单独一个文件存储。

下面以PaddleOCR的超轻量中文模型为例，介绍使用编译好的opt文件完成inference模型到Paddle-Lite优化模型的转换。

```
# 下载PaddleOCR的超轻量文inference模型，并解压
wget  https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar && tar xf ch_det_mv3_db_infer.tar
wget  https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar && tar xf ch_rec_mv3_crnn_infer.tar

# 转换检测模型
./opt --model_file=./ch_det_mv3_db/model --param_file=./ch_det_mv3_db/params --optimize_out_type=naive_buffer --optimize_out=./ch_det_mv3_db_opt --valid_targets=arm

# 转换识别模型
./opt --model_file=./ch_rec_mv3_crnn/model --param_file=./ch_rec_mv3_crnn/params --optimize_out_type=naive_buffer --optimize_out=./ch_rec_mv3_crnn_opt --valid_targets=arm
```

转换成功后，当前目录下会多出`ch_det_mv3_db_opt.nb`, `ch_rec_mv3_crnn_opt.nb`结尾的文件，即是转换成功的模型文件。

注意：使用paddle-lite部署时，需要使用opt工具优化后的模型。 opt 转换的输入模型是paddle保存的inference模型

### 2.2 与手机联调

首先需要进行一些准备工作。
 1. 准备一台arm8的安卓手机，如果编译的预测库和opt文件是armv7，则需要arm7的手机。
 2. 打开手机的USB调试选项，选择文件传输模式，连接电脑。
 3. 电脑上安装adb工具，用于调试。在电脑终端中输入`adb devices`，如果有类似以下输出，则表示安装成功。
```
    List of devices attached
    744be294    device
```

 4. 准备预测库、模型和预测文件，在预测库`inference_lite_lib.android.armv8/demo/cxx/`下新建一个`ocr/`文件夹，并将转换后的nb模型、
 PaddleOCR repo中`PaddleOCR/deploy/lite/` 下的所有文件放在新建的ocr文件夹下。执行完成后，ocr文件夹下将有如下文件格式：

```
demo/cxx/ocr/
|-- debug/                      新建debug文件夹存放模型文件
|   |--ch_det_mv3_db_opt.nb     优化后的检测模型文件
|   |--ch_rec_mv3_crnn_opt.nb   优化后的识别模型文件
|-- utils/  
|   |-- clipper.cpp             Clipper库的cpp文件
|   |-- clipper.hpp             Clipper库的hpp文件
|   |-- crnn_process.cpp        识别模型CRNN的预处理和后处理cpp文件
|   |-- db_post_process.cpp     检测模型DB的后处理cpp文件
|-- Makefile                    编译文件
|-- ocr_db_crnn.cc              C++预测文件
```

 5. 编译C++预测文件，准备测试图像，准备字典文件
 ```
 cd demo/cxx/ocr/
 # 执行编译
 make
 # 将编译的可执行文件移动到debug文件夹中
 mv ocr_db_crnn ./debug/
 cp ../../../cxx/lib/libpaddle_light_api_shared.so ./debug/
 ```
 准备测试图像，以`PaddleOCR/doc/imgs/12.jpg`为例，将测试的图像复制到`demo/cxx/ocr/debug/`文件夹下。
 准备字典文件，将`PaddleOCR/ppocr/utils/ppocr_keys_v1.txt`复制到`demo/cxx/ocr/debug/`文件夹下。
 上述步骤完成后就可以使用adb将文件push到手机上运行，步骤如下：
 ```
 adb push debug /data/local/tmp/
 adb shell
 cd /data/local/tmp/debug
 export LD_LIBRARY_PATH=/data/local/tmp/debug:$LD_LIBRARY_PATH
 ./ocr_db_crnn ch_det_mv3_db_opt.nb  ch_rec_mv3_crnn_opt.nb ./12.jpg
 ```
 如果对代码做了修改，则需要重新编译并push到手机上。
