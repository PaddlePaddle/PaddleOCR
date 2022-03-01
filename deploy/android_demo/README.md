- [Android Demo](#android-demo)
  - [1. 简介](#1-简介)
  - [2. 近期更新](#2-近期更新)
  - [3. 快速使用](#3-快速使用)
    - [3.1 安装最新版本的Android Studio](#31-安装最新版本的android-studio)
    - [3.2 安装 NDK 20 以上版本](#32-安装-ndk-20-以上版本)
    - [3.3 导入项目](#33-导入项目)
  - [4 更多支持](#4-更多支持)

# Android Demo

## 1. 简介
此为PaddleOCR的Android Demo，目前支持文本检测，文本方向分类器和文本识别模型的使用。使用 [PaddleLite v2.10](https://github.com/PaddlePaddle/Paddle-Lite/tree/release/v2.10) 进行开发。


## 2. 近期更新
* 2022.02.27 
    * 预测库更新到PaddleLite v2.10
    * 支持6种运行模式：
      * 检测+分类+识别
      * 检测+识别
      * 分类+识别
      * 检测
      * 识别
      * 分类

## 3. 快速使用

### 3.1 安装最新版本的Android Studio
可以从 https://developer.android.com/studio 下载。本Demo使用是4.0版本Android Studio编写。

### 3.2 安装 NDK 20 以上版本
Demo测试的时候使用的是NDK 20b版本，20版本以上均可以支持编译成功。

如果您是初学者，可以用以下方式安装和测试NDK编译环境。
点击 File -> New ->New Project，  新建  "Native C++" project

### 3.3 导入项目

点击 File->New->Import Project...， 然后跟着Android Studio的引导导入

## 4 更多支持

前往[Paddle-Lite](https://github.com/PaddlePaddle/Paddle-Lite)，获得更多开发支持

