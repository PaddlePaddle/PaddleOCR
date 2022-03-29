# Visual Studio 2019 Community CMake 编译指南

PaddleOCR在Windows 平台下基于`Visual Studio 2019 Community` 进行了测试。微软从`Visual Studio 2017`开始即支持直接管理`CMake`跨平台编译项目，但是直到`2019`才提供了稳定和完全的支持，所以如果你想使用CMake管理项目编译构建，我们推荐你使用`Visual Studio 2019`环境下构建。


## 前置条件
* Visual Studio 2019
* CUDA 10.2，cudnn 7+ （仅在使用GPU版本的预测库时需要）
* CMake 3.0+

请确保系统已经安装好上述基本软件，我们使用的是`VS2019`的社区版。

**下面所有示例以工作目录为 `D:\projects`演示**。

### Step1: 下载PaddlePaddle C++ 预测库 paddle_inference

PaddlePaddle C++ 预测库针对不同的`CPU`和`CUDA`版本提供了不同的预编译版本，请根据实际情况下载:  [C++预测库下载列表](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#windows)

解压后`D:\projects\paddle_inference`目录包含内容为：
```
paddle_inference
├── paddle # paddle核心库和头文件
|
├── third_party # 第三方依赖库和头文件
|
└── version.txt # 版本和编译信息
```

### Step2: 安装配置OpenCV

1. 在OpenCV官网下载适用于Windows平台的3.4.6版本， [下载地址](https://sourceforge.net/projects/opencvlibrary/files/3.4.6/opencv-3.4.6-vc14_vc15.exe/download)  
2. 运行下载的可执行文件，将OpenCV解压至指定目录，如`D:\projects\opencv`
3. 配置环境变量，如下流程所示  
    - 我的电脑->属性->高级系统设置->环境变量
    - 在系统变量中找到Path（如没有，自行创建），并双击编辑
    - 新建，将opencv路径填入并保存，如`D:\projects\opencv\build\x64\vc14\bin`

### Step3: 使用Visual Studio 2019直接编译CMake

1. 打开Visual Studio 2019 Community，点击`继续但无需代码`
![step2](https://paddleseg.bj.bcebos.com/inference/vs2019_step1.png)
2. 点击： `文件`->`打开`->`CMake`
![step2.1](https://paddleseg.bj.bcebos.com/inference/vs2019_step2.png)

选择项目代码所在路径，并打开`CMakeList.txt`：

![step2.2](https://paddleseg.bj.bcebos.com/inference/vs2019_step3.png)

3. 点击：`项目`->`CMake设置`

![step3](https://paddleseg.bj.bcebos.com/inference/vs2019_step4.png)

4. 分别设置编译选项指定`CUDA`、`CUDNN_LIB`、`OpenCV`、`Paddle预测库`的路径

三个编译参数的含义说明如下（带`*`表示仅在使用**GPU版本**预测库时指定, 其中CUDA库版本尽量对齐）：

|  参数名   | 含义  |
|  ----  | ----  |
| *CUDA_LIB  | CUDA的库路径 |
| *CUDNN_LIB | CUDNN的库路径 |
| OPENCV_DIR  | OpenCV的安装路径 |
| PADDLE_LIB | Paddle预测库的路径 |

**注意：**
  1. 使用`CPU`版预测库，请把`WITH_GPU`的勾去掉
  2. 如果使用的是`openblas`版本，请把`WITH_MKL`勾去掉

![step4](https://paddleseg.bj.bcebos.com/inference/vs2019_step5.png)

下面给出with GPU的配置示例：
![step5](./vs2019_build_withgpu_config.png)
**注意：**
  CMAKE_BACKWARDS的版本要根据平台安装cmake的版本进行设置。

**设置完成后**, 点击上图中`保存并生成CMake缓存以加载变量`。

5. 点击`生成`->`全部生成`

![step6](https://paddleseg.bj.bcebos.com/inference/vs2019_step6.png)


### Step4: 预测

上述`Visual Studio 2019`编译产出的可执行文件在`out\build\x64-Release\Release`目录下，打开`cmd`，并切换到`D:\projects\PaddleOCR\deploy\cpp_infer\`：

```
cd D:\projects\PaddleOCR\deploy\cpp_infer
```
可执行文件`ppocr.exe`即为样例的预测程序，其主要使用方法如下，更多使用方法可以参考[说明文档](../readme.md)`运行demo`部分。

```shell
#识别中文图片 `D:\projects\PaddleOCR\doc\imgs_words\ch\`  
.\out\build\x64-Release\Release\ppocr.exe rec --rec_model_dir=D:\projects\PaddleOCR\ch_ppocr_mobile_v2.0_rec_infer --image_dir=D:\projects\PaddleOCR\doc\imgs_words\ch\

#识别英文图片 'D:\projects\PaddleOCR\doc\imgs_words\en\'
.\out\build\x64-Release\Release\ppocr.exe rec --rec_model_dir=D:\projects\PaddleOCR\inference\rec_mv3crnn --image_dir=D:\projects\PaddleOCR\doc\imgs_words\en\ --char_list_file=D:\projects\PaddleOCR\ppocr\utils\dict\en_dict.txt
```


第一个参数为配置文件路径，第二个参数为需要预测的图片路径，第三个参数为配置文本识别的字典。


### FQA
* 在Windows下的终端中执行文件exe时，可能会发生乱码的现象，此时需要在终端中输入`CHCP 65001`，将终端的编码方式由GBK编码(默认)改为UTF-8编码，更加具体的解释可以参考这篇博客：[https://blog.csdn.net/qq_35038153/article/details/78430359](https://blog.csdn.net/qq_35038153/article/details/78430359)。

* 编译时，如果报错`错误：C1083 无法打开包括文件:"dirent.h":No such file or directory`，下载可[dirent.h](https://paddleocr.bj.bcebos.com/deploy/cpp_infer/cpp_files/dirent.h)文件，并添加到`utility.cpp`的头文件引用中。

* 编译时，如果报错`Autolog未定义`，新建`autolog.h`文件，内容为：[autolog.h](https://github.com/LDOUBLEV/AutoLog/blob/main/auto_log/autolog.h)，并添加到`main.cpp`的头文件引用中，再次编译。

* 运行时，如果弹窗报错找不到`paddle_inference.dll`或者`openblas.dll`，在`D:\projects\paddle_inference`预测库内找到这两个文件，复制到`D:\projects\PaddleOCR\deploy\cpp_infer\out\build\x64-Release\Release`目录下。不用重新编译，再次运行即可。

* 运行时，弹窗报错提示`应用程序无法正常启动(0xc0000142)`，并且`cmd`窗口内提示`You are using Paddle compiled with TensorRT, but TensorRT dynamic library is not found.`，把tensort目录下的lib里面的所有dll文件复制到release目录下，再次运行即可。
