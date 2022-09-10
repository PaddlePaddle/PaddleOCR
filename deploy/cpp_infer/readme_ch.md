[English](readme.md) | 简体中文

# 服务器端C++预测

- [1. 准备环境](#1)
    - [1.1 运行准备](#11)
    - [1.2 编译opencv库](#12)
    - [1.3 下载或者编译Paddle预测库](#13)
- [2 开始运行](#2)
    - [2.1 准备模型](#21)
    - [2.2 编译PaddleOCR C++预测demo](#22)
    - [2.3 运行demo](#23)
- [3. FAQ](#3)

本章节介绍PaddleOCR 模型的C++部署方法。C++在性能计算上优于Python，因此，在大多数CPU、GPU部署场景，多采用C++的部署方式，本节将介绍如何在Linux\Windows (CPU\GPU)环境下配置C++环境并完成PaddleOCR模型部署。


<a name="1"></a>

## 1. 准备环境

<a name="11"></a>

### 1.1 运行准备

- Linux环境，推荐使用docker。
- Windows环境。

* 该文档主要介绍基于Linux环境的PaddleOCR C++预测流程，如果需要在Windows下基于预测库进行C++预测，具体编译方法请参考[Windows下编译教程](./docs/windows_vs2019_build.md)

<a name="12"></a>

### 1.2 编译opencv库

* 首先需要从opencv官网上下载在Linux环境下源码编译的包，以opencv3.4.7为例，下载命令如下。

```bash
cd deploy/cpp_infer
wget https://paddleocr.bj.bcebos.com/libs/opencv/opencv-3.4.7.tar.gz
tar -xf opencv-3.4.7.tar.gz
```

最终可以在当前目录下看到`opencv-3.4.7/`的文件夹。

* 编译opencv，设置opencv源码路径(`root_path`)以及安装路径(`install_path`)。进入opencv源码路径下，按照下面的方式进行编译。

```shell
root_path="your_opencv_root_path"
install_path=${root_path}/opencv3
build_dir=${root_path}/build

rm -rf ${build_dir}
mkdir ${build_dir}
cd ${build_dir}

cmake .. \
    -DCMAKE_INSTALL_PREFIX=${install_path} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_IPP=OFF \
    -DBUILD_IPP_IW=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DWITH_ZLIB=ON \
    -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_PNG=ON \
    -DWITH_TIFF=ON \
    -DBUILD_TIFF=ON

make -j
make install
```

也可以直接修改`tools/build_opencv.sh`的内容，然后直接运行下面的命令进行编译。

```shell
sh tools/build_opencv.sh
```

其中`root_path`为下载的opencv源码路径，`install_path`为opencv的安装路径，`make install`完成之后，会在该文件夹下生成opencv头文件和库文件，用于后面的OCR代码编译。

最终在安装路径下的文件结构如下所示。

```
opencv3/
|-- bin
|-- include
|-- lib
|-- lib64
|-- share
```

<a name="13"></a>

### 1.3 下载或者编译Paddle预测库

可以选择直接下载安装或者从源码编译，下文分别进行具体说明。

<a name="131"></a>
#### 1.3.1 直接下载安装

[Paddle预测库官网](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux) 上提供了不同cuda版本的Linux预测库，可以在官网查看并选择合适的预测库版本（*建议选择paddle版本>=2.0.1版本的预测库* ）。

下载之后解压:

```shell
tar -xf paddle_inference.tgz
```

最终会在当前的文件夹中生成`paddle_inference/`的子文件夹。

<a name="132"></a>
#### 1.3.2 预测库源码编译

如果希望获取最新预测库特性，可以从github上克隆最新Paddle代码进行编译，生成最新的预测库。

* 使用git获取代码:

```shell
git clone https://github.com/PaddlePaddle/Paddle.git
git checkout develop
```

* 进入Paddle目录，进行编译:

```shell
rm -rf build
mkdir build
cd build

cmake  .. \
    -DWITH_CONTRIB=OFF \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON  \
    -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_INFERENCE_API_TEST=OFF \
    -DON_INFER=ON \
    -DWITH_PYTHON=ON
make -j
make inference_lib_dist
```

更多编译参数选项介绍可以参考[Paddle预测库编译文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#congyuanmabianyi)。


* 编译完成之后，可以在`build/paddle_inference_install_dir/`文件下看到生成了以下文件及文件夹。

```
build/paddle_inference_install_dir/
|-- CMakeCache.txt
|-- paddle
|-- third_party
|-- version.txt
```

其中`paddle`就是C++预测所需的Paddle库，`version.txt`中包含当前预测库的版本信息。

<a name="2"></a>

## 2. 开始运行

<a name="21"></a>

### 2.1 准备模型

直接下载PaddleOCR提供的推理模型，或者参考[模型预测章节](../../doc/doc_ch/inference_ppocr.md)，将训练好的模型导出为推理模型。模型导出之后，假设放在`inference`目录下，则目录结构如下。

```
inference/
|-- det_db
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- rec_rcnn
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- cls
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- table
|   |--inference.pdiparams
|   |--inference.pdmodel
```

<a name="22"></a>

### 2.2 编译PaddleOCR C++预测demo

编译命令如下，其中Paddle C++预测库、opencv等其他依赖库的地址需要换成自己机器上的实际地址。

```shell
sh tools/build.sh
```

具体的，需要修改`tools/build.sh`中环境路径，相关内容如下：

```shell
OPENCV_DIR=your_opencv_dir
LIB_DIR=your_paddle_inference_dir
CUDA_LIB_DIR=your_cuda_lib_dir
CUDNN_LIB_DIR=/your_cudnn_lib_dir
```

其中，`OPENCV_DIR`为opencv编译安装的地址；`LIB_DIR`为下载(`paddle_inference`文件夹)或者编译生成的Paddle预测库地址(`build/paddle_inference_install_dir`文件夹)；`CUDA_LIB_DIR`为cuda库文件地址，在docker中为`/usr/local/cuda/lib64`；`CUDNN_LIB_DIR`为cudnn库文件地址，在docker中为`/usr/lib/x86_64-linux-gnu/`。**注意：以上路径都写绝对路径，不要写相对路径。**


编译完成之后，会在`build`文件夹下生成一个名为`ppocr`的可执行文件。

<a name="23"></a>

### 2.3 运行demo

本demo支持系统串联调用，也支持单个功能的调用，如，只使用检测或识别功能。

**注意** ppocr默认使用`PP-OCRv3`模型，识别模型使用的输入shape为`3,48,320`, 如需使用旧版本的PP-OCR模型，则需要设置参数`--rec_img_h=32`。


运行方式：  
```shell
./build/ppocr [--param1] [--param2] [...]
```
具体命令如下：

##### 1. 检测+分类+识别：
```shell
./build/ppocr --det_model_dir=inference/det_db \
    --rec_model_dir=inference/rec_rcnn \
    --cls_model_dir=inference/cls \
    --image_dir=../../doc/imgs/12.jpg \
    --use_angle_cls=true \
    --det=true \
    --rec=true \
    --cls=true \
```

##### 2. 检测+识别：
```shell
./build/ppocr --det_model_dir=inference/det_db \
    --rec_model_dir=inference/rec_rcnn \
    --image_dir=../../doc/imgs/12.jpg \
    --use_angle_cls=false \
    --det=true \
    --rec=true \
    --cls=false \
```

##### 3. 检测：
```shell
./build/ppocr --det_model_dir=inference/det_db \
    --image_dir=../../doc/imgs/12.jpg \
    --det=true \
    --rec=false
```

##### 4. 分类+识别：
```shell
./build/ppocr --rec_model_dir=inference/rec_rcnn \
    --cls_model_dir=inference/cls \
    --image_dir=../../doc/imgs_words/ch/word_1.jpg \
    --use_angle_cls=true \
    --det=false \
    --rec=true \
    --cls=true \
```

##### 5. 识别：
```shell
./build/ppocr --rec_model_dir=inference/rec_rcnn \
    --image_dir=../../doc/imgs_words/ch/word_1.jpg \
    --use_angle_cls=false \
    --det=false \
    --rec=true \
    --cls=false \
```

##### 6. 分类：
```shell
./build/ppocr --cls_model_dir=inference/cls \
    --cls_model_dir=inference/cls \
    --image_dir=../../doc/imgs_words/ch/word_1.jpg \
    --use_angle_cls=true \
    --det=false \
    --rec=false \
    --cls=true \
```

##### 7. 表格识别
```shell
./build/ppocr --det_model_dir=inference/det_db \
    --rec_model_dir=inference/rec_rcnn \
    --table_model_dir=inference/table \
    --image_dir=../../ppstructure/docs/table/table.jpg \
    --type=structure \
    --table=true
```

更多支持的可调节参数解释如下：

- 通用参数

|参数名称|类型|默认参数|意义|
| :---: | :---: | :---: | :---: |
|use_gpu|bool|false|是否使用GPU|
|gpu_id|int|0|GPU id，使用GPU时有效|
|gpu_mem|int|4000|申请的GPU内存|
|cpu_math_library_num_threads|int|10|CPU预测时的线程数，在机器核数充足的情况下，该值越大，预测速度越快|
|enable_mkldnn|bool|true|是否使用mkldnn库|
|output|str|./output|可视化结果保存的路径|

- 前向相关

|参数名称|类型|默认参数|意义|
| :---: | :---: | :---: | :---: |
|det|bool|true|前向是否执行文字检测|
|rec|bool|true|前向是否执行文字识别|
|cls|bool|false|前向是否执行文字方向分类|


- 检测模型相关

|参数名称|类型|默认参数|意义|
| :---: | :---: | :---: | :---: |
|det_model_dir|string|-|检测模型inference model地址|
|max_side_len|int|960|输入图像长宽大于960时，等比例缩放图像，使得图像最长边为960|
|det_db_thresh|float|0.3|用于过滤DB预测的二值化图像，设置为0.-0.3对结果影响不明显|
|det_db_box_thresh|float|0.5|DB后处理过滤box的阈值，如果检测存在漏框情况，可酌情减小|
|det_db_unclip_ratio|float|1.6|表示文本框的紧致程度，越小则文本框更靠近文本|
|det_db_score_mode|string|slow|slow:使用多边形框计算bbox score，fast:使用矩形框计算。矩形框计算速度更快，多边形框对弯曲文本区域计算更准确。|
|visualize|bool|true|是否对结果进行可视化，为1时，预测结果会保存在`output`字段指定的文件夹下和输入图像同名的图像上。|

- 方向分类器相关

|参数名称|类型|默认参数|意义|
| :---: | :---: | :---: | :---: |
|use_angle_cls|bool|false|是否使用方向分类器|
|cls_model_dir|string|-|方向分类器inference model地址|
|cls_thresh|float|0.9|方向分类器的得分阈值|
|cls_batch_num|int|1|方向分类器batchsize|

- 文字识别模型相关

|参数名称|类型|默认参数|意义|
| :---: | :---: | :---: | :---: |
|rec_model_dir|string|-|文字识别模型inference model地址|
|rec_char_dict_path|string|../../ppocr/utils/ppocr_keys_v1.txt|字典文件|
|rec_batch_num|int|6|文字识别模型batchsize|
|rec_img_h|int|48|文字识别模型输入图像高度|
|rec_img_w|int|320|文字识别模型输入图像宽度|


- 表格识别模型相关

|参数名称|类型|默认参数|意义|
| :---: | :---: | :---: | :---: |
|table_model_dir|string|-|表格识别模型inference model地址|
|table_char_dict_path|string|../../ppocr/utils/dict/table_structure_dict.txt|字典文件|
|table_max_len|int|488|表格识别模型输入图像长边大小，最终网络输入图像大小为（table_max_len，table_max_len）|
|merge_no_span_structure|bool|true|是否合并<td> 和 </td> 为<td></td>|


* PaddleOCR也支持多语言的预测，更多支持的语言和模型可以参考[识别文档](../../doc/doc_ch/recognition.md)中的多语言字典与模型部分，如果希望进行多语言预测，只需将修改`rec_char_dict_path`（字典文件路径）以及`rec_model_dir`（inference模型路径）字段即可。

最终屏幕上会输出检测结果如下。

- ocr

```bash
predict img: ../../doc/imgs/12.jpg
../../doc/imgs/12.jpg
0       det boxes: [[74,553],[427,542],[428,571],[75,582]] rec text: 打浦路252935号 rec score: 0.947724
1       det boxes: [[23,507],[513,488],[515,529],[24,548]] rec text: 绿洲仕格维花园公寓 rec score: 0.993728
2       det boxes: [[187,456],[399,448],[400,480],[188,488]] rec text: 打浦路15号 rec score: 0.964994
3       det boxes: [[42,413],[483,391],[484,428],[43,450]] rec text: 上海斯格威铂尔大酒店 rec score: 0.980086
The detection visualized image saved in ./output//12.jpg
```

- table

```bash
predict img: ../../ppstructure/docs/table/table.jpg
0       type: table, region: [0,0,371,293], res: <html><body><table><thead><tr><td>Methods</td><td>R</td><td>P</td><td>F</td><td>FPS</td></tr></thead><tbody><tr><td>SegLink [26]</td><td>70.0</td><td>86.0</td><td>77.0</td><td>8.9</td></tr><tr><td>PixelLink [4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td>-</td></tr><tr><td>TextSnake [18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField [37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2 </td></tr><tr><td>MSR[38]</td><td>76.7</td><td>87.4</td><td>81.7</td><td>-</td></tr><tr><td>FTSN [3]</td><td>77.1</td><td>87.6</td><td>82.0</td><td>-</td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><td>-</td></tr><tr><td>CRAFT [2]</td><td>78.2</td><td>88.2</td><td>82.9</td><td>8.6</td></tr><tr><td>MCN [16]</td><td>79</td><td>88</td><td>83</td><td>-</td></tr><tr><td>ATRR[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td>-</td></tr><tr><td>PAN [34]</td><td>83.8</td><td>84.4</td><td>84.1</td><td>30.2</td></tr><tr><td>DB[12]</td><td>79.2</td><td>91.5</td><td>84.9</td><td>32.0</td></tr><tr><td>DRRG [41]</td><td>82.30</td><td>88.05</td><td>85.08</td><td>-</td></tr><tr><td>Ours (SynText)</td><td>80.68</td><td>85.40</td><td>82.97</td><td>12.68</td></tr><tr><td>Ours (MLT-17)</td><td>84.54</td><td>86.62</td><td>85.57</td><td>12.31</td></tr></tbody></table></body></html>
```

<a name="3"></a>
## 3. FAQ

 1.  遇到报错 `unable to access 'https://github.com/LDOUBLEV/AutoLog.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.`， 将 `deploy/cpp_infer/external-cmake/auto-log.cmake` 中的github地址改为 https://gitee.com/Double_V/AutoLog 地址即可。
