# Visual Studio 2019 Community CMake 编译指南

PaddleOCR在Windows 平台下基于`Visual Studio 2019 Community` 进行了测试。微软从`Visual Studio 2017`开始即支持直接管理`CMake`跨平台编译项目，但是直到`2019`才提供了稳定和完全的支持，所以如果你想使用CMake管理项目编译构建，我们推荐你使用`Visual Studio 2019`环境下构建。


## 前置条件
* Visual Studio 2019
* CUDA 9.0 / CUDA 10.0，cudnn 7+ （仅在使用GPU版本的预测库时需要）
* CMake 3.0+

请确保系统已经安装好上述基本软件，我们使用的是`VS2019`的社区版。

**下面所有示例以工作目录为 `D:\projects`演示**。

### Step1: 下载PaddlePaddle C++ 预测库 fluid_inference

PaddlePaddle C++ 预测库针对不同的`CPU`和`CUDA`版本提供了不同的预编译版本，请根据实际情况下载:  [C++预测库下载列表](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/05_inference_deployment/inference/windows_cpp_inference.html)

解压后`D:\projects\fluid_inference`目录包含内容为：
```
fluid_inference
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

3. 点击：`项目`->`cpp_inference_demo的CMake设置`

![step3](https://paddleseg.bj.bcebos.com/inference/vs2019_step4.png)

4. 点击`浏览`，分别设置编译选项指定`CUDA`、`CUDNN_LIB`、`OpenCV`、`Paddle预测库`的路径

三个编译参数的含义说明如下（带`*`表示仅在使用**GPU版本**预测库时指定, 其中CUDA库版本尽量对齐，**使用9.0、10.0版本，不使用9.2、10.1等版本CUDA库**）：

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

**设置完成后**, 点击上图中`保存并生成CMake缓存以加载变量`。

5. 点击`生成`->`全部生成`

![step6](https://paddleseg.bj.bcebos.com/inference/vs2019_step6.png)


### Step4: 预测及可视化

上述`Visual Studio 2019`编译产出的可执行文件在`out\build\x64-Release`目录下，打开`cmd`，并切换到该目录：

```
cd D:\projects\PaddleOCR\deploy\cpp_infer\out\build\x64-Release
```
可执行文件`ocr_system.exe`即为样例的预测程序，其主要使用方法如下

```shell
#预测图片 `D:\projects\PaddleOCR\doc\imgs\10.jpg`  
.\ocr_system.exe D:\projects\PaddleOCR\deploy\cpp_infer\tools\config.txt D:\projects\PaddleOCR\doc\imgs\10.jpg
```

第一个参数为配置文件路径，第二个参数为需要预测的图片路径。


### 注意
* 在Windows下的终端中执行文件exe时，可能会发生乱码的现象，此时需要在终端中输入`CHCP 65001`，将终端的编码方式由GBK编码(默认)改为UTF-8编码，更加具体的解释可以参考这篇博客：[https://blog.csdn.net/qq_35038153/article/details/78430359](https://blog.csdn.net/qq_35038153/article/details/78430359)。
