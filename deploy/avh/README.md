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

English | [简体中文](README_ch.md)

Running PaddleOCR text recognition model on bare metal Arm(R) Cortex(R)-M55 CPU using Arm Virtual Hardware
======================================================================

This folder contains an example of how to run a PaddleOCR model on bare metal [Cortex(R)-M55 CPU](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m55) using [Arm Virtual Hardware](https://www.arm.com/products/development-tools/simulation/virtual-hardware).


Running environment and prerequisites
-------------
Case 1: If the demo is run in Arm Virtual Hardware Amazon Machine Image(AMI) instance hosted by [AWS](https://aws.amazon.com/marketplace/pp/prodview-urbpq7yo5va7g?sr=0-1&ref_=beagle&applicationId=AWSMPContessa)/[AWS China](https://awsmarketplace.amazonaws.cn/marketplace/pp/prodview-2y7nefntbmybu), the following software will be installed through [configure_avh.sh](./configure_avh.sh) script. It will install automatically when you run the application through [run_demo.sh](./run_demo.sh) script.
You can refer to this [guide](https://arm-software.github.io/AVH/main/examples/html/MicroSpeech.html#amilaunch) to launch an Arm Virtual Hardware AMI instance.

Case 2: If the demo is run in the [ci_cpu Docker container](https://github.com/apache/tvm/blob/main/docker/Dockerfile.ci_cpu) provided with [TVM](https://github.com/apache/tvm), then the following software will already be installed.

Case 3: If the demo is not run in the ci_cpu Docker container, then you will need the following:
- Software required to build and run the demo (These can all be installed by running
  tvm/docker/install/ubuntu_install_ethosu_driver_stack.sh.)
  - [Fixed Virtual Platform (FVP) based on Arm(R) Corstone(TM)-300 software](https://developer.arm.com/tools-and-software/open-source-software/arm-platforms-software/arm-ecosystem-fvps)
  - [cmake 3.19.5](https://github.com/Kitware/CMake/releases/)
  - [GCC toolchain from Arm(R)](https://developer.arm.com/-/media/Files/downloads/gnu-rm/10-2020q4/gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux.tar.bz2)
  - [Arm(R) Ethos(TM)-U NPU driver stack](https://review.mlplatform.org)
  - [CMSIS](https://github.com/ARM-software/CMSIS_5)
- The python libraries listed in the requirements.txt of this directory
  - These can be installed by running the following from the current directory:
    ```bash
    pip install -r ./requirements.txt
    ```

In case2 and case3:

You will need to update your PATH environment variable to include the path to cmake 3.19.5 and the FVP.
For example if you've installed these in ```/opt/arm``` , then you would do the following:
```bash
export PATH=/opt/arm/FVP_Corstone_SSE-300/models/Linux64_GCC-6.4:/opt/arm/cmake/bin:$PATH
```

You will also need TVM which can either be:
  - Installed from TLCPack(see [TLCPack](https://tlcpack.ai/))
  - Built from source (see [Install from Source](https://tvm.apache.org/docs/install/from_source.html))
    - When building from source, the following need to be set in config.cmake:
      - set(USE_CMSISNN ON)
      - set(USE_MICRO ON)
      - set(USE_LLVM ON)


Running the demo application
----------------------------
Type the following command to run the bare metal text recognition application ([src/demo_bare_metal.c](./src/demo_bare_metal.c)):

```bash
./run_demo.sh
```

If you are not able to use Arm Virtual Hardware Amazon Machine Image(AMI) instance hosted by AWS/AWS China, specify argument --enable_FVP to 1 to make the application run on local Fixed Virtual Platforms (FVPs) executables.

```bash
./run_demo.sh --enable_FVP 1
```

If the Ethos(TM)-U platform and/or CMSIS have not been installed in /opt/arm/ethosu then
the locations for these can be specified as arguments to run_demo.sh, for example:

```bash
./run_demo.sh --cmsis_path /home/tvm-user/cmsis \
--ethosu_platform_path /home/tvm-user/ethosu/core_platform
```

With [run_demo.sh](./run_demo.sh) to run the demo application, it will:
- Set up running environment by installing the required prerequisites automatically if running in Arm Virtual Hardware Amazon AMI instance(not specify --enable_FVP to 1)
- Download a PaddleOCR text recognition model
- Use tvmc to compile the text recognition model for Cortex(R)-M55 CPU and CMSIS-NN
- Create a C header file inputs.c containing the image data as a C array
- Create a C header file outputs.c containing a C array where the output of inference will be stored
- Build the demo application
- Run the demo application on a Arm Virtual Hardware based on Arm(R) Corstone(TM)-300 software
- The application will report the text on the image and the corresponding score.

Using your own image
--------------------
The create_image.py script takes a single argument on the command line which is the path of the
image to be converted into an array of bytes for consumption by the model.

The demo can be modified to use an image of your choice by changing the following line in run_demo.sh

```bash
python3 ./convert_image.py path/to/image
```

Model description
-----------------
The example is built on [PP-OCRv3](https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/doc/doc_ch/PP-OCRv3_introduction.md) English recognition model released by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). Since Arm(R) Cortex(R)-M55 CPU does not support rnn operator, we delete the unsupported operator based on the PP-OCRv3 text recognition model to obtain the current 2.7M English recognition model.

PP-OCRv3 is the third version of the PP-OCR series model. This series of models has the following features:
  - PP-OCRv3: ultra-lightweight OCR system: detection (3.6M) + direction classifier (1.4M) + recognition (12M) = 17.0M
  - Support more than 80 kinds of multi-language recognition models, including English, Chinese, French, German, Arabic, Korean, Japanese and so on. For details
  - Support vertical text recognition, and long text recognition
