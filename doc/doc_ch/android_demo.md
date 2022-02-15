# Android Demo 快速测试


### 1. 安装最新版本的Android Studio

可以从 https://developer.android.com/studio 下载。本Demo使用是4.0版本Android Studio编写。

### 2. 创建新项目

Demo测试的时候使用的是NDK 20b版本，20版本以上均可以支持编译成功。

如果您是初学者，可以用以下方式安装和测试NDK编译环境。
点击 File -> New ->New Project，  新建  "Native C++" project


1. Start a new Android Studio project
    在项目模版中选择 Native C++ 选择PaddleOCR/deploy/android_demo 路径
    进入项目后会自动编译，第一次编译会花费较长的时间，建议添加代理加速下载。

**代理添加：**

选择 Android Studio -> Preferences -> Appearance & Behavior -> System Settings -> HTTP Proxy -> Manual proxy configuration

![](../demo/proxy.png)

2. 开始编译

点击编译按钮，连接手机，跟着Android Studio的引导完成操作。

在 Android Studio 里看到下图，表示编译完成：

![](../demo/build.png)

**提示:** 此时如果出现下列找不到OpenCV的报错信息，请重新点击编译，编译完成后退出项目，再次进入。

![](../demo/error.png)

### 3. 发送到手机端

完成编译，点击运行，在手机端查看效果。

### 4. 如何自定义demo图片

1. 图片存放路径:android_demo/app/src/main/assets/images

   将自定义图片放置在该路径下

2. 配置文件: android_demo/app/src/main/res/values/strings.xml

   修改 IMAGE_PATH_DEFAULT 为自定义图片名即可


# 获得更多支持
前往[端计算模型生成平台EasyEdge](https://ai.baidu.com/easyedge/app/open_source_demo?referrerUrl=paddlelite)，获得更多开发支持：

- Demo APP：可使用手机扫码安装，方便手机端快速体验文字识别
- SDK：模型被封装为适配不同芯片硬件和操作系统SDK，包括完善的接口，方便进行二次开发
