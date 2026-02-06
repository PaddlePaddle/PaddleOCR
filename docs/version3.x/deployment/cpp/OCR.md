# 通用 OCR 产线 C++ 部署 - Linux

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
- [4. FAQ](#4-faq)

本章节介绍通用 OCR 产线 C++ 部署方法。通用 OCR 产线由以下5个模块组成：

1. 文档图像方向分类模块（可选）
2. 文本图像矫正模块 (可选)
3. 文本行方向分类模块（可选）
4. 文本检测模块
5. 文本识别模块

下面将介绍如何在 Linux (CPU/GPU) 环境下配置 C++ 环境并完成通用 OCR 产线部署。

- 备注
    - Windows 环境具体编译方法请参考 [Windows 编译教程](./OCR_windows.md)，在编译完成后，后续运行 demo 的指令与 Linux 一致。

## 1. 环境准备

- **本章节编译运行时用到的源代码位于 [PaddleOCR/deploy/cpp_infer](https://github.com/PaddlePaddle/PaddleOCR/tree/main/deploy/cpp_infer) 目录下。**

- Linux 环境。
    - gcc   8.2（编译使用 Paddle Inference GPU 版本时，gcc>=11.2）
    - cmake 3.18

### 1.1 编译 OpenCV 库

目前仅支持 OpenCV 4.x 版本。下面以 OpenCV 4.7.0 为例。

1. 执行如下命令下载 OpenCV 源码：

```bash
cd deploy/cpp_infer
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv-4.7.0.tgz
tar -xf opencv-4.7.0.tgz
```

2. 配置并编译 OpenCV 库：

- a. 在 `tools/build_opencv.sh` 脚本中，将 `root_path` 设置为 opencv-4.7.0 源码的绝对路径。
- b. 设置 `install_path`，如默认的 `${root_path}/opencv4`。`install_path` 在后续编译预测 demo 时，将作为 OpenCV 库的路径使用。
- c. 配置完成后，运行以下命令进行 OpenCV 的编译：

    ```bash
    sh tools/build_opencv.sh
    ```

### 1.2 编译 Paddle Inference

可以选择直接下载预编译包或者手动编译源码。

#### 1.2.1 直接下载预编译包（推荐）

[Paddle Inference 官网](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/download_lib.html) 上提供了 Linux 预测库，可以在官网查看并选择合适的预编译包。

下载之后解压:

```shell
tar -xvf paddle_inference.tgz
```

最终会在当前的文件夹中生成 `paddle_inference/` 的子文件夹。

#### 1.2.2 源码编译预测库

可以选择通过源码自行编译预测库，源码编译可灵活配置各类功能和依赖，以适应不同的硬件和软件环境。详细步骤请参考 [Linux 下源码编译](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/compile/source_compile_under_Linux.html)。

## 2. 开始运行

### 2.1 编译预测 demo

在编译预测demo前，请确保您已经按照 1.1 和 1.2 节编译好 OpenCV 库和 Paddle Inference 预测库。

修改 tools/build.sh 中的配置后，执行以下命令进行编译：

```shell
sh tools/build.sh
```

相关配置参数详细介绍如下：

<table>
<tr>
<th>参数</th>
<th>说明</th>
<th>默认值</th>
</tr>
<tr>
<td><code>OPENCV_DIR</code></td>
<td>OpenCV编译安装的路径（如上述编译 OpenCV 时的 <code>install_path</code> ，必填。</td>
<td></td>
</tr>
<tr>
<td><code>LIB_DIR</code></td>
<td>下载的 <code>Paddle Inference</code> 的预编译包或手动编译生成的Paddle Inference库路径（如 <code>build/paddle_inference_install_dir</code> 文件夹），必填。</td>
<td></td>
</tr>
<tr>
<td><code>CUDA_LIB_DIR</code></td>
<td>CUDA库文件路径，通常为<code>/usr/local/cuda/lib64</code>。当Paddle Inference库为GPU版本且设置 <code>-DWITH_GPU=ON</code> 时需要设置该参数。</td>
<td></td>
</tr>
<tr>
<td><code>CUDNN_LIB_DIR</code></td>
<td>cuDNN 库文件路径，通常为 <code>/usr/lib/x86_64-linux-gnu/</code> 。当 Paddle Inference 库为 GPU 版本且设置 <code>-DWITH_GPU=ON</code> 时需要设置该参数。</td>
<td></td>
</tr>
<tr>
<td><code>WITH_GPU</code></td>
<td>当设置为 ON 时可以进行 GPU 版本 demo 的编译，要求 Paddle Inference 库为 GPU 版本。</td>
<td>OFF</td>
</tr>
</table>

**注意：以上路径需要填绝对路径。**

### 2.2 准备模型

可以直接下载 PaddleOCR 提供的推理模型：

<details>
<summary><b>文档图像方向分类模块（可选）：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a></td>
<td>99.06</td>
<td>7</td>
<td>基于 PP-LCNet_x1_0 的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>文本图像矫正模块（可选）：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>CER </th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">推理模型</a></td>
<td>0.179</td>
<td>30.3</td>
<td>高精度文本图像矫正模型</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>文本行方向分类模块（可选）：</b></summary>
<table>
<thead>
<tr>
<th>模型</th>
<th>模型下载链接</th>
<th>Top-1 Acc（%）</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<td>PP-LCNet_x1_0_textline_ori (默认)</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar">推理模型</a></td>
<td>99.42</td>
<td>6.5</td>
<td>基于 PP-LCNet_x1_0 的文本行分类模型，含有两个类别，即0度，180度</td>
</tr>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">推理模型</a></td>
<td>98.85</td>
<td>0.96</td>
<td>基于 PP-LCNet_x0_25 的文本行分类模型，含有两个类别，即0度，180度</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>文本检测模块：</b></summary>
<table>
<thead>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>检测Hmean（%）</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det (默认)</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">推理模型</a></td>
<td>83.8</td>
<td>84.3</td>
<td>PP-OCRv5 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">推理模型</a></td>
<td>79.0</td>
<td>4.7</td>
<td>PP-OCRv5 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">推理模型</a></td>
<td>69.2</td>
<td>109</td>
<td>PP-OCRv4 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">推理模型</a></td>
<td>63.8</td>
<td>4.7</td>
<td>PP-OCRv4 的移动端文本检测模型，效率更高，适合在端侧设备部署</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>文本识别模块：</b></summary>
<table>
<tr>
<th>模型</th><th>模型下载链接</th>
<th>识别 Avg Accuracy(%)</th>
<th>模型存储大小（MB）</th>
<th>介绍</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec (默认)</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar">推理模型</a></td>
<td>86.38</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。在保持识别效果的同时，兼顾推理速度和模型鲁棒性，为各种场景下的文档理解提供高效、精准的技术支撑。</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar">推理模型</a></td>
<td>81.29</td>
<td>16</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_doc_infer.tar">推理模型</a></td>
<td>86.58</td>
<td>182</td>
<td>PP-OCRv4_server_rec_doc 是在 PP-OCRv4_server_rec 的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+，除文档相关的文字识别能力提升外，也同时提升了通用文字的识别能力</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a></td>
<td>78.74</td>
<td>10.5</td>
<td>PP-OCRv4 的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">推理模型</a></td>
<td>85.19</td>
<td>173</td>
<td>PP-OCRv4 的服务器端模型，推理精度高，可以部署在多种不同的服务器上</td>
</tr>
</tbody>
</table>
</details>

也可以参考各模块的模型导出章节，如[文本检测模块-模型导出](../../module_usage/text_detection.md#44-模型导出)，将训练好的模型导出为推理模型。

模型的目录结构一般如下所示：

```
PP-OCRv5_mobile_det
|--inference.pdiparams
|--inference.json
|--inference.yml
```

### 2.3 运行预测 demo

在本地使用通用 OCR 产线 C++前，请先成功编译预测 demo。编译后，可通过命令行体验或调用 api 进行二次开发并重新编译生成应用程序。

**请注意，如果在执行过程中遇到程序失去响应、程序异常退出、内存资源耗尽、推理速度极慢等问题，请尝试参考文档调整配置，例如关闭不需要使用的功能或使用更轻量的模型。**

本 demo 支持系统串联调用，也支持单个模块的调用，运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png)到本地：

运行方式：

```shell
./build/ppocr <pipeline_or_module> [--param1] [--param2] [...]
```

常用参数如下：

<li>输入输出相关</li>

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>本地待预测图片，必填。仅支持<code>jpg</code>，<code>png</code>, <code>jpeg</code>,<code>bmp</code>格式的图像。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>指定推理结果文件保存的路径，该路径下会保存推理结果的 json 文件和预测结果图片。</td>
<td><code>str</code></td>
<td><code>./output</code></td>
</tr>
</tbody>
</table>

<details><summary><b>点击展开查看更多参数的详细说明</b></summary>

<li>通用参数</li>

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。支持指定具体卡号：
<ul>
<li><b>CPU</b>：如 <code>cpu</code> 表示使用 CPU 进行推理；</li>
<li><b>GPU</b>：如 <code>gpu:0</code> 表示使用第 1 块 GPU 进行推理；</li>
</ul>如果不设置，将默认使用产线初始化的该参数值，初始化时，如果编译时添加<code>-DWITH_GPU=ON</code>，则会优先使用本地的 GPU 0号设备，否则，将使用 CPU 设备。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>计算精度，如 <code>fp32</code>、<code>fp16</code>。</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速推理。如果 MKL-DNN 不可用或模型不支持通过 MKL-DNN 加速，即使设置了此标志，也不会使用加速。
</td>
<td><code>bool</code></td>
<td><code>true</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN 缓存容量。
</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>PaddleInference CPU 加速库线程数量</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>PaddleX产线配置文件路径。</td>
<td><code>str</code></td>
<td></td>
</tr>
</tbody>
</table>

<li>模块开关</li>

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>是否加载并使用文档方向分类模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>true</code>。</td>
<td><code>bool</code></td>
<td><code>true</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载并使用文本图像矫正模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>true</code>。</td>
<td><code>bool</code></td>
<td><code>true</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>是否加载并使用文本行方向模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>true</code>。</td>
<td><code>bool</code></td>
<td><code>true</code></td>
</tr>
</tbody>
</table>

<li>检测模型相关</li>

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>text_detection_model_name</code></td>
<td>文本检测模型的名称。如果不设置，将会使用产线默认模型。当传入文本检测模型路径的模型名称与产线默认文本识别模型名称配置不一致时，需指定传入模型的名称。</td>
<td><code>str</code></td>
<td><code>PP-OCRv5_server_det</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>文本检测模型的目录路径，必填。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>文本检测的图像边长限制。
大于 <code>0</code> 的任意整数。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>64</code>。
</td>
<td><code>int</code></td>
<td><code>64</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>文本检测的边长度限制类型。支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code>。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>min</code>。
</td>
<td><code>str</code></td>
<td><code>min</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>文本检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
大于<code>0</code>的任意浮点数。如果不设置，将使用产线初始化的该参数值。
</td>
<td><code>float</code></td>
<td><code>0.3</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>文本检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
大于 <code>0</code> 的任意浮点数。如果不设置，将使用产线初始化的该参数值（默认为 <code>0.6</code>）。
</td>
<td><code>float</code></td>
<td><code>0.6</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。大于 <code>0</code> 的任意浮点数。如果不设置，将使用产线初始化的该参数值。
</td>
<td><code>float</code></td>
<td><code>1.5</code></td>
</tr>
<tr>
<td><code>text_det_input_shape</code></td>
<td>文本检测的输入形状，您可以设置3个值代表C，H，W。</td>
<td><code>str</code></td>
<td>""</td>
</tr>
</tbody>
</table>

<li>方向分类器相关</li>

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>文档方向分类模型的名称。如果不设置，将会使用产线默认模型。当传入文档方向分类模型与产线默认模型不一致时，需指定传入模型的名称。</td>
<td><code>str</code></td>
<td><code>PP-LCNet_x1_0_doc_ori</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。当设置<code>use_doc_orientation_classify = false</code>时，可不添加。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>文本行方向分类模型的名称。如果不设置，将会使用产线默认模型。当传入文本行方向分类模型与产线默认模型不一致时，需指定传入模型的名称。</td>
<td><code>str</code></td>
<td><code>PP-LCNet_x1_0_textline_ori</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>文本行方向分类模型的目录路径。当设置<code>use_textline_orientation = false</code>时，可不添加。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>文本行方向模型的batch size。如果不设置，将会使用产线默认模型。</td>
<td><code>int</code></td>
<td><code>6</code></td>
</tr>
</tbody>
</table>

<li>文字识别模型相关</li>

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>文本识别模型的名称。如果不设置，将会使用产线默认模型。当传入文本识别模型路径的模型名称与产线默认文本识别模型名称配置不一致时，需指定传入模型的名称。</td>
<td><code>str</code></td>
<td><code>PP-OCRv5_server_rec</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>文本识别模型的目录路径，必填。</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>文本识别模型的batch size。如果不设置，将会使用产线默认值。</td>
<td><code>int</code></td>
<td><code>6</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。大于<code>0</code>的任意浮点数。</td>
<td><code>float</code></td>
<td><code>0.0</code></td>
</tr>
<tr>
<td><code>text_rec_input_shape</code></td>
<td>文本识别的输入形状，您可以设置3个值代表C，H，W。</td>
<td><code>str</code></td>
<td>""</td>
</tr>
</tbody>
</table>

</details>

#### 2.3.1 调用示例-系统串联调用

本节提供了系统串联调用的调用示例，请参考 2.1 节准备好模型，假设模型的目录结构如下所示：

```
models
|--PP-LCNet_x1_0_doc_ori_infer
|--UVDoc_infer
|--PP-LCNet_x1_0_textline_ori_infer
|--PP-OCRv5_server_det_infer
|--PP-OCRv5_server_rec_infer
```

=== "全模块串联"

    ```bash
    ./build/ppocr ocr --input ./general_ocr_002.png --save_path ./output/  \
    --doc_orientation_classify_model_dir models/PP-LCNet_x1_0_doc_ori_infer \
    --doc_unwarping_model_dir models/UVDoc_infer \
    --textline_orientation_model_dir models/PP-LCNet_x1_0_textline_ori_infer \
    --text_detection_model_dir models/PP-OCRv5_server_det_infer \
    --text_recognition_model_dir models/PP-OCRv5_server_rec_infer \
    --device cpu
    ```

    输出示例（若指定了 `save_path`，则会在该路径下生成标准的 json 预测结果文件和预测结果图片）：

    ```bash
    {
       "input_path": "./general_ocr_002.png",
       "doc_preprocessor_res": {
           "model_settings": {"use_doc_unwarping": true, "use_doc_orientation_classify": true},
           "angle": 0
        },
       ...,
       "dt_polys": [[[132, 6], [355, 6], [355, 64], [132, 64]],
        [[424, 9], [689, 9], [689, 59], [424, 59]],
         ...,
        [[664, 8], [867, 4], [868, 55], [665, 60]],
        [[31, 99], [173, 99], [173, 126], [31, 126]]],
         ...,
       "rec_texts": ["登机牌", "BOARDING", "GPASS", ..., ],
       ...,
    }        
    ```

=== "文本检测+文本行方向分类+文本识别"

    ```bash
    ./build/ppocr ocr --input ./general_ocr_002.png --save_path ./output/  \
    --textline_orientation_model_dir models/PP-LCNet_x1_0_textline_ori_infer \
    --text_detection_model_dir models/PP-OCRv5_server_det_infer \
    --text_recognition_model_dir models/PP-OCRv5_server_rec_infer \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --device cpu
    ```

    输出示例（若指定了 `save_path`，则会在该路径下生成标准的 json 预测结果文件和预测结果图片）：

    ```bash
    {
       "input_path": "./general_ocr_002.png",
       ...,
        "dt_polys": [[[0, 1], [334, 1], [334, 34], [0, 34]],
        [[151, 21], [357, 16], [358, 72], [152, 76]],
         ...,
        [[675, 97], [740, 97], [740, 121], [675, 121]],
        [[751, 97], [836, 94], [837, 115], [752, 119]],
         ...,
       "rec_texts": ["净小8866-", "登机牌", "BOARDING", "GPASS", ..., ],
       ...,
    }    
    ```

=== "文本检测+文本识别"

    ```bash
    ./build/ppocr ocr --input ./general_ocr_002.png --save_path ./output/  \
    --text_detection_model_dir models/PP-OCRv5_server_det_infer \
    --text_recognition_model_dir models/PP-OCRv5_server_rec_infer \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --device cpu
    ```

    输出示例（若指定了 `save_path`，则会在该路径下生成标准的 json 预测结果文件和预测结果图片）：

    ```bash
    {
       "input_path": "./general_ocr_002.png",
       ...,
       "dt_polys": [[[0, 1], [334, 1], [334, 34], [0, 34]],
        [[151, 21], [357, 16], [358, 72], [152, 76]],
         ...,
        [[61, 109], [194, 106], [194, 132], [61, 135]],
        [[80, 138], [219, 136], [219, 162], [80, 164]],
         ...,
       "rec_texts": ["www.997788.com中国收藏热线","登机牌", "BOARDING", "GPASS", ..., ],
       ...,
    }    
    ```

以上示例代码会生成如下文本检测结果图：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/ocr_res.png"/>

若需要查看文本识别结果图，请参考后文 **可视化文本识别结果** 小节。

#### 2.3.2 调用示例-单模块调用

=== "文档图像方向分类"   

    ```bash
    ./build/ppocr doc_img_orientation_classification --input ./general_ocr_002.png --save_path ./output/  \
    --doc_orientation_classify_model_dir models/PP-LCNet_x1_0_doc_ori_infer \
    --device cpu 
    ```

    输出示例（若指定了 `save_path`，则会在该路径下生成标准的 json 预测结果文件和预测结果图片）：

    ```bash
    {
    "res": {
        "input_path": {./general_ocr_002.png},
        "class_ids": {0},
        "scores": {0.926328},
        "label_names": {0},
    }
    ```

=== "文档图像矫正"

    ```bash
    ./build/ppocr text_image_unwarping --input ./general_ocr_002.png --save_path ./output/  \
    --doc_unwarping_model_dir models/UVDoc_infer \
    --device cpu 
    ```

    输出示例（若指定了 `save_path`，则会在该路径下生成标准的 json 预测结果文件和预测结果图片）：

    ```bash
    {
    "res": {
        "input_path": {./general_ocr_002.png},
        "doctr_img": {...}
    }    
    ```   

=== "文本行方向分类"

    ```bash
    ./build/ppocr textline_orientation_classification --input ./general_ocr_002.png --save_path ./output/  \
    --textline_orientation_model_dir models/PP-LCNet_x1_0_textline_ori_infer \
    --device cpu 
    ```

    输出示例（若指定了 `save_path`，则会在该路径下生成标准的 json 预测结果文件和预测结果图片）：

    ```bash
    {
    "res": {
        "input_path": {./general_ocr_002.png},
        "class_ids": {0},
        "scores": {0.719926},
        "label_names": {0_degree},
    }
    ```      

=== "文本检测"

    ```bash
    ./build/ppocr text_detection --input ./general_ocr_002.png --save_path ./output/  \
    --text_detection_model_dir models/PP-OCRv5_server_det_infer \
    --device cpu 
    ```

    输出示例（若指定了 `save_path`，则会在该路径下生成标准的 json 预测结果文件和预测结果图片）：

    ```bash
    {
    "res": {
        "input_path": {./general_ocr_002.png    },
        "dt_polys": [
            [[98, 456], [834, 441], [834, 466], [98, 480]],
            [[344, 347], [662, 343], [662, 366], [344, 371]],
            [[66, 341], [165, 337], [167, 363], [67, 367]],
            ...,
            [[0, 1], [331, 0], [332, 32], [0, 34]],
        ]},
        "dt_scores": [
            0.812284, 0.8082, 0.848293, ..., 
        ]
      }
    }
    ```
    
=== "文本识别"

    ```bash
    ./build/ppocr text_recognition --input ./general_ocr_rec_001.png --save_path ./output/  \
    --text_recognition_model_dir models/PP-OCRv5_server_rec_infer \ 
    --device cpu 
    ```

    输出示例（若指定了 `save_path`，则会在该路径下生成标准的 json 预测结果文件和预测结果图片）：

    ```bash
    {
    "res": {
        "input_path": {./general_ocr_rec_001.png },
        "rec_text": {绿洲仕格维花园公寓 }
        "rec_score": {0.982409 }
    }
    ```

### 2.4 C++ API 集成

命令行方式是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：

由于通用 OCR 产线配置参数较多，故采用结构体传参进行实例化，结构体命名规则为 `pipeline_class_name  + Params`，如通用 OCR 产线对应的类名为 `PaddleOCR` ，结构体为 `PaddleOCRParams`。

```c++
#include "src/api/pipelines/ocr.h"


int main(){
    PaddleOCRParams params;
    params.doc_orientation_classify_model_dir = "models/PP-LCNet_x1_0_doc_ori_infer"; // 文档方向分类模型路径。
    params.doc_unwarping_model_dir = "models/UVDoc_infer"; // 文本图像矫正模型路径。
    params.textline_orientation_model_dir = "models/PP-LCNet_x1_0_textline_ori_infer"; // 文本行方向分类模型路径。
    params.text_detection_model_dir = "models/PP-OCRv5_server_det_infer"; // 文本检测模型路径
    params.text_recognition_model_dir = "models/PP-OCRv5_server_rec_infer"; // 文本识别模型路径

    // params.device = "gpu"; // 推理时使用GPU。请确保编译时添加 -DWITH_GPU=ON 选项，否则使用CPU。
    // params.use_doc_orientation_classify = false;  // 不使用文档方向分类模型。
    // params.use_doc_unwarping = false; // 不使用文本图像矫正模型。
    // params.use_textline_orientation = false; // 不使用文本行方向分类模型。
    // params.text_detection_model_name = "PP-OCRv5_server_det"; // 使用 PP-OCRv5_server_det 模型进行检测。
    // params.text_recognition_model_name = "PP-OCRv5_server_rec"; // 使用 PP-OCRv5_server_rec 模型进行识别。
    // params.vis_font_dir = "your_vis_font_dir"; // 当编译时添加 -DUSE_FREETYPE=ON 选项，必须提供相应 ttf 字体文件路径。

    auto infer = PaddleOCR(params);
    auto outputs = infer.Predict("./general_ocr_002.png");
    for (auto& output : outputs) {
      output->Print();
      output->SaveToImg("./output/");
      output->SaveToJson("./output/");
    }
}
```

## 3. 拓展功能

### 3.1 多语种文字识别

PP-OCRv5 还提供了覆盖 39 种语言的多语种文字识别能力，包括韩文、西班牙文、法文、葡萄牙文、德文、意大利文、俄罗斯文、泰文、希腊文等，具体支持语种如下：

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>链接</th>
      <th>支持语种</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PP-OCRv5_server_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar">推理模型</a></td>
      <td>简体中文、繁体中文、英文、日文</td>
    </tr>
    <tr>
      <td>PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar">推理模型</a></td>
      <td>简体中文、繁体中文、英文、日文</td>
    </tr>
    <tr>
      <td>korean_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv5_mobile_rec_infer.tar">推理模型</a></td>
      <td>韩文、英文</td>
    </tr>
    <tr>
      <td>latin_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv5_mobile_rec_infer.tar">推理模型</a></td>
      <td>英文、法文、德文、南非荷兰文、意大利文、西班牙文、波斯尼亚文、葡萄牙文、捷克文、威尔士文、丹麦文、爱沙尼亚文、爱尔兰文、克罗地亚文、乌兹别克文、匈牙利文、塞尔维亚文（latin）、印度尼西亚文、欧西坦文、冰岛文、立陶宛文、毛利文、马来文、荷兰文、挪威文、波兰文、斯洛伐克文、斯洛文尼亚文、阿尔巴尼亚文、瑞典文、西瓦希里文、塔加洛文、土耳其文、拉丁文</td>
    </tr>
    <tr>
      <td>eslav_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/eslav_PP-OCRv5_mobile_rec_infer.tar">推理模型</a></td>
      <td>俄罗斯文、白俄罗斯文、乌克兰文、英文</td>
    </tr>
    <tr>
      <td>th_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/th_PP-OCRv5_mobile_rec_infer.tar">推理模型</a></td>
      <td>泰文、英文</td>
    </tr>
    <tr>
      <td>el_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/el_PP-OCRv5_mobile_rec_infer.tar">推理模型</a></td>
      <td>希腊文、英文</td>
    </tr>
    <tr>
      <td>en_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar">推理模型</a></td>
      <td>英文</td>
    </tr>
  </tbody>
</table>

在使用产线或模块时传入对应的识别模型即可，如使用文本识别模块进行法文识别：

```
./build/ppocr text_recognition \
--input ./french.png \
--text_recognition_model_name latin_PP-OCRv5_mobile_rec \
--text_recognition_model_dir latin_PP-OCRv5_mobile_rec_infer \
--save_path ./output/
```

更多详细说明可参考 [PP-OCRv5多语种文字识别介绍](../../algorithm/PP-OCRv5/PP-OCRv5_multi_languages.md)。

### 3.2 可视化文本识别结果

我们使用 4.x 版本的 opencv_contrib 模块中的 FreeType 进行字体渲染，如果想要可视化文本识别结果，需要下载 OpenCV 和 opencv_contrib 的源码并编译包含 FreeType 模块的 OpenCV。下载源码时需确保两者的版本一致。以下以 opencv-4.7.0 和 opencv_contrib-4.7.0 为例进行说明：

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv-4.7.0.tgz
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv_contrib-4.7.0.tgz
tar -xf opencv-4.7.0.tgz
tar -xf opencv_contrib-4.7.0.tgz
```

安装FreeType依赖库

```bash
sudo apt-get update
sudo apt-get install libfreetype6-dev libharfbuzz-dev
```

编译包含 FreeType 模块的 OpenCV 的步骤如下：

- a. 在 `tools/build_opencv.sh` 脚本中增加如下三个选项：
    - -DOPENCV_EXTRA_MODULES_PATH=your_opencv_contrib-4.7.0/modules/
    - -DBUILD_opencv_freetype=ON
    - -DWITH_FREETYPE=ON
- b. 在 `tools/build_opencv.sh` 脚本中，将 `root_path` 设置为 opencv-4.7.0 源码的绝对路径。
- c. 在 `tools/build_opencv.sh` 脚本中，设置 `install_path`，如默认的 `${root_path}/opencv4`。`install_path` 在后续编译预测 demo 时，将作为 OpenCV 库的路径使用。
- d. 配置完成后，运行以下命令进行 OpenCV 的编译：

    ```bash
    sh tools/build_opencv.sh
    ```

- e. 在 `tools/build.sh` 设置 `-DUSE_FREETYPE=ON` 开启文字渲染功能，设置 `--vis_font_dir your_ttf_path` 提供相应 ttf 字体文件路径。运行以下命令进行预测 demo 的编译：

    ```bash
    sh tools/build.sh
    ```

编译并运行预测 demo 可以得到如下可视化文本识别结果：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/ocr_res_with_freetype.png"/>

## 4. FAQ

1. 如果遇到 `Model name mismatch, please input the correct model dir. model dir is xxx, but model name is xxx` 的报错，说明指定的模型名称和传入模型不匹配。比如文本识别模型指定名称是 `PP-OCRv5_server_rec `，但传入模型是 `PP-OCRv5_mobile_rec`。
解决：需要调整模型名称或传入的模型。例如上述例子，可以使用 `--text_recognition_model_name PP-OCRv5_mobile_rec` 指定和传入模型匹配的模型名称。

2. 在 Windows 的控制台中输出出现乱码，原因可能是 Windows 控制台的字符编码是 GBK，请设置为 UTF-8 编码。
