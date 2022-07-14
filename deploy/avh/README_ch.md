<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->
[English](README.md) | 简体中文

通过TVM在 Arm(R) Cortex(R)-M55 CPU 上运行 PaddleOCR文 本能识别模型
===============================================================

此文件夹包含如何使用 TVM 在 Cortex(R)-M55 CPU 上运行 PaddleOCR 模型的示例。

依赖
-------------
本demo运行在TVM提供的docker环境上，在该环境中已经安装好的必须的软件


在非docker环境中，需要手动安装如下依赖项: 

- 软件可通过[安装脚本](https://github.com/apache/tvm/blob/main/docker/install/ubuntu_install_ethosu_driver_stack.sh)一键安装
  - [Fixed Virtual Platform (FVP) based on Arm(R) Corstone(TM)-300 software](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps)
  - [cmake 3.19.5](https://github.com/Kitware/CMake/releases/)
  - [GCC toolchain from Arm(R)](https://developer.arm.com/-/media/Files/downloads/gnu-rm/10-2020q4/gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux.tar.bz2)
  - [Arm(R) Ethos(TM)-U NPU driver stack](https://review.mlplatform.org)
  - [CMSIS](https://github.com/ARM-software/CMSIS_5)
- python 依赖
  ```bash
  pip install -r ./requirements.txt
  ```
- TVM
  - 从源码安装([Install from Source](https://tvm.apache.org/docs/install/from_source.html))
    从源码安装时，需要设置如下字段
      - set(USE_CMSISNN ON)
      - set(USE_MICRO ON)
      - set(USE_LLVM ON)
  - 从TLCPack 安装([TLCPack](https://tlcpack.ai/))

安装完成后需要更新环境变量，以软件安装地址为`/opt/arm`为例：
```bash
export PATH=/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4:/opt/arm/cmake/bin:$PATH
```

运行demo
----------------------------
使用如下命令可以一键运行demo

```bash
./run_demo.sh
```

如果 Ethos(TM)-U 平台或 CMSIS 没有安装在 `/opt/arm/ethosu` 中，可通过参数进行设置，例如：

```bash
./run_demo.sh --cmsis_path /home/tvm-user/cmsis \
--ethosu_platform_path /home/tvm-user/ethosu/core_platform
```

`./run_demo.sh`脚本会执行如下步骤:
- 下载 PaddleOCR 文字识别模型
- 使用tvm将PaddleOCR 文字识别模型编译为 Cortex(R)-M55 CPU 和 CMSIS-NN 后端的可执行文件
- 创建一个包含输入图像数据的头文件`inputs.c`
- 创建一个包含输出tensor大小的头文件`outputs.c`
- 编译可执行程序
- 运行程序
- 输出图片上的文字和置信度

使用自己的图片
--------------------
替换 `run_demo.sh ` 中140行处的图片地址即可

使用自己的模型
--------------------
替换 `run_demo.sh ` 中130行处的模型地址即可

模型描述
-----------------

在这个demo中，我们使用的模型是基于[PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/PP-OCRv3_introduction.md)的英文识别模型。由于Arm(R) Cortex(R)-M55 CPU不支持rnn算子，我们在PP-OCRv3原始文本识别模型的基础上进行适配，最终模型大小为2.7M。

PP-OCRv3是[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)发布的PP-OCR系列模型的第三个版本，该系列模型具有以下特点：
   - 超轻量级OCR系统：检测（3.6M）+方向分类器（1.4M）+识别（12M）=17.0M。
   - 支持80多种多语言识别模型，包括英文、中文、法文、德文、阿拉伯文、韩文、日文等。 
   - 支持竖排文本识别，长文本识别。
