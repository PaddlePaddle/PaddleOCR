# 通用OCR产线C++部署

- [1. 环境准备](#1)
    - [1.1 编译opencv库](#12)
    - [1.2 下载或者编译Paddle预测库](#13)
- [2. 运行](#2)
    - [2.1 准备模型](#21)
    - [2.2 编译可执行文件](#22)
    - [2.3 运行](#23)
- [3. FAQ](#3)

本章节介绍 通用OCR产线的C++部署方法。通用OCR 产线由以下5个模块组成：

1. 文档图像方向分类模块（可选）
2. 文本图像矫正模块 (可选)
3. 文本行方向分类模块（可选）
4. 文本检测模块
5. 文本识别模块

下面将介绍如何在 Linux (CPU/GPU) 环境下配置 C++ 环境并完成通用 OCR 产线部署。

## 1. 准备环境

- Linux 环境。
    - gcc   8.2（当使用Paddle Inference GPU版本时需要更高版本时，gcc>=11.2）
    - cmake 3.18

- Windows 环境：具体编译方法请参考 [Windows 编译教程](./docs/windows_vs2022_build.md)。

### 1.1 编译 OpenCV 库

首先需要编译 OpenCV 库，编译流程如下：

修改 `tools/build_opencv.sh`，运行下面的命令完成 OpenCV 的编译。

```shell
sh tools/build_opencv.sh
```

### 1.2 下载Paddle Inference C++ 预编译包或者手动编译源码

可以选择直接下载Paddle Inference官网提供的预编译包或者手动编译源码，下文分别进行具体说明。

#### 1.2.1 直接下载预编译包（推荐）
[Paddle Inference官网](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/download_lib.html) 上提供了Linux预测库，可以在官网查看并选择合适的预编译包（*建议选择paddle版本>=3.0.0版本的预测库* ）。

下载之后解压:

```shell
tar -xf paddle_inference.tgz
```
最终会在当前的文件夹中生成`paddle_inference/`的子文件夹。
#### 1.2.2 预测库源码编译
[Linux下源码编译](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/compile/source_compile_under_Linux.html)

## 2. 开始运行

### 2.1 准备模型

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
<td>基于PP-LCNet_x1_0的文档图像分类模型，含有四个类别，即0度，90度，180度，270度</td>
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
<td>PP-LCNet_x1_0_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar">推理模型</a></td>
<td>99.42</td>
<td>6.5</td>
<td>基于PP-LCNet_x1_0的文本行分类模型，含有两个类别，即0度，180度</td>
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
<td>PP-OCRv5_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">推理模型</a></td>
<td>83.8</td>
<td>84.3</td>
<td>PP-OCRv5 的服务端文本检测模型，精度更高，适合在性能较好的服务器上部署</td>
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
<td>PP-OCRv5_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">推理模型</a></td>
<td>86.38</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。在保持识别效果的同时，兼顾推理速度和模型鲁棒性，为各种场景下的文档理解提供高效、精准的技术支撑。</td>
</tr>
</tbody>
</table>
</details>

也可以参考[模型预测章节]()，将训练好的模型导出为推理模型。

模型的目录结构一般如下所示：

```
PP-OCRv5_mobile_det
|--inference.pdiparams
|--inference.json
|--inference.yml
```

### 2.2 编译PaddleOCR C++预测demo
在编译PaddleOCR C++预测demo前，请确保您已经编译好OpenCV库和Paddle Inference预测库。

```shell
sh tools/build.sh
```

具体的，需要修改tools/build.sh中环境路径及相关选项，相关内容如下：

```shell
OPENCV_DIR=your_opencv_dir
LIB_DIR=your_paddle_inference_dir
CUDA_LIB_DIR=your_cuda_lib_dir
CUDNN_LIB_DIR=/your_cudnn_lib_dir

cmake .. \
    -DPADDLE_LIB=${LIB_DIR} \
    -DWITH_MKL=ON \
    -DWITH_GPU=OFF \
    -DWITH_STATIC_LIB=OFF \
    -DWITH_TENSORRT=OFF \
    -DOPENCV_DIR=${OPENCV_DIR} \
    -DCUDNN_LIB=${CUDNN_LIB_DIR} \
    -DCUDA_LIB=${CUDA_LIB_DIR} \
    -DUSE_FREETYPE=OFF
```

<table>
<tr>
<th>参数</th>
<th>说明</th>
<th>默认值</th>
</tr>
<tr>
<td><code>OPENCV_DIR</code></td>
<td>OpenCV编译安装的路径，必填。</td>
<td></td>
</tr>
<tr>
<td><code>LIB_DIR</code></td>
<td>下载的 <code>paddle_inference</code> 文件夹或编译生成的Paddle Inference库路径（如 <code>build/paddle_inference_install_dir</code> 文件夹），必填。</td>
<td></td>
</tr>
<tr>
<td><code>CUDA_LIB_DIR</code></td>
<td> CUDA库文件路径，通常为<code>/usr/local/cuda/lib64</code>。当Paddle Inference库为GPU版本且设置 <code>-DWITH_GPU=ON</code> 时需要设置该参数。</td>
<td></td>
</tr>
<tr>
<td><code>CUDNN_LIB_DIR</code></td>
<td>cuDNN 库文件路径，通常为 <code>/usr/lib/x86_64-linux-gnu/</code> 。当 Paddle Inference 库为 GPU 版本且设置 <code>-DWITH_GPU=ON</code> 时需要设置该参数。</td>
<td></td>
</tr>
<tr>
<td><code>WITH_GPU</code></td>
<td>当设置为 ON 且 Paddle Inference 库为 GPU 版本时，可以使用 GPU 推理。</td>
<td>OFF</td>
</tr>
</table>

**注意：以上路径都写绝对路径，不要写相对路径。**

### 2.3 运行
在本地使用PaddleOCR C++前，请确保您已经成功编译PaddleOCR C++预测demo。编译完成后，可以在本地使用命令行体验或者根据您的实际需求调用PaddleOCR C++ API进行二次开发，并重新编译生成您自己的应用程序。

**请注意，如果在执行过程中遇到程序失去响应、程序异常退出、内存资源耗尽、推理速度极慢等问题，请尝试参考文档调整配置，例如关闭不需要使用的功能或使用更轻量的模型。**

#### 2.3.1 命令行方式
本demo支持系统串联调用，也支持单个模块的调用。

运行方式：

```shell
./build/ppocr   [--param1] [--param2] [...]
```

具体命令如下：

##### 系统串联调用

=== "全模块串联"

    ```bash
    ./build/ppocr paddleocr --input your_input --save_path your_save_path/  
    --doc_orientation_classify_model_dir your_doc_orientation_classify_model_dir
    --doc_unwarping_model_dir your_doc_unwarping_model_dir
    --textline_orientation_model_dir your_textline_orientation_model_dir
    --text_detection_model_dir your_text_detection_model_dir
    --text_recognition_model_dir your_text_recognition_model_dir
    ```

    输出示例：

    ```bash

    ```


=== "文本检测+文本行方向分类+文本识别"

    ```bash
    ./build/ppocr paddleocr --input your_input --save_path your_save_path/  
    --doc_orientation_classify_model_dir your_doc_orientation_classify_model_dir
    --doc_unwarping_model_dir your_doc_unwarping_model_dir
    --textline_orientation_model_dir your_textline_orientation_model_dir
    --text_detection_model_dir your_text_detection_model_dir
    --text_recognition_model_dir your_text_recognition_model_dir
    --use_doc_orientation_classify False
    --use_doc_unwarping False
    ```

    输出示例：

    ```bash
    ```

=== "文本检测+文本识别"

    ```bash
    ./build/ppocr paddleocr --input your_input --save_path your_save_path/  
    --doc_orientation_classify_model_dir your_doc_orientation_classify_model_dir
    --doc_unwarping_model_dir your_doc_unwarping_model_dir
    --textline_orientation_model_dir your_textline_orientation_model_dir
    --text_detection_model_dir your_text_detection_model_dir
    --text_recognition_model_dir your_text_recognition_model_dir
    --use_doc_orientation_classify False
    --use_doc_unwarping False
    --use_textline_orientation False
    ```

    输出示例：

    ```bash

    ```
##### 单模块调用

=== "文档图像方向分类"

    ```bash
    ./build/ppocr doc_img_orientation_classification --input your_input --save_path your_save_path/  
    --doc_orientation_classify_model_dir your_doc_orientation_classify_model_dir
    ```

    输出示例：

    ```bash

    ```

=== "文档图像矫正"

    ```bash
    ./build/ppocr text_image_unwarping --input your_input --save_path your_save_path/  
    --doc_unwarping_model_dir your_doc_unwarping_model_dir
    ```

    输出示例：

    ```bash

    ```    
=== "文本行方向分类"

    ```bash
    ./build/ppocr textline_orientation_classification --input your_input --save_path your_save_path/  
    --textline_orientation_model_dir your_textline_orientation_model_dir
    ```

    输出示例：

    ```bash

    ```      

=== "文本检测"

    ```bash
    ./build/ppocr text_detection --input your_input --save_path your_save_path/  
    --text_detection_model_dir your_text_detection_model_dir
    ```

    输出示例：

    ```bash

    ```
    
=== "文本识别"

    ```bash
    ./build/ppocr text_recognition --input your_input --save_path your_save_path/  
    --text_recognition_model_dir your_text_recognition_model_dir
    ```

    输出示例：

    ```bash

    ```
#### 2.3.2 C++ API方式集成
命令行方式是为了快速体验查看效果，一般来说，在项目中，往往需要通过代码集成，您可以通过几行代码即可完成产线的快速推理，推理代码如下：
```c++
#include "src/API/pipelines/ocr.h"

int main(){
    PaddleOCRParams params;
    params.doc_orientation_classify_model_dir = your_doc_orientation_classify_model_dir; // 文档方向分类模型路径。
    params.doc_unwarping_model_dir = your_doc_unwarping_model_dir; //文本图像矫正模型路径。
    params.textline_orientation_model_dir = your_textline_orientation_model_dir; //文本行方向分类模型路径。
    params.text_detection_model_dir = your_text_detection_model_dir; //文本检测模型路径
    params.text_recognition_model_dir = your_text_recognition_model_dir; //文本识别模型路径
    params.vis_font_dir  = your_vis_font_dir; //当编译时添加-DUSE_FREETYPE=ON选项，必须提供相应tff字体文件路径。

    //params.device = "gpu"; //推理时使用GPU。请确保编译时添加-DWITH_GPU=ON选项，否则使用CPU。
    //params.thread_num = 1;  // 多线程推理，根据硬件性能选择配置。
    //params.use_doc_orientation_classify = false;  // 不使用文档方向分类模型。
    //params.use_doc_unwarping = false; // 不使用文本图像矫正模型。
    //params.use_textline_orientation = false; // 不使用文本行方向分类模型。
    //params.params.text_recognition_model_name = "PP-OCRv5_server_rec" //使用PP-OCRv5_server_rec模型进行识别。

    auto infer = PaddleOCR(params);
    auto outputs  = infer.Predict("./input.jpg");

    for (auto& output : outputs) {
      output->Print();
      output->SaveToImg("./output/");
      output->SaveToJson("./output/");
    }
}
```

<details><summary><b>更多支持的可调节参数设置，点击展开以查看调节参数的详细说明</b></summary>

- 通用参数

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
<td>计算精度，如<code>fp32</code>、<code>fp16</code>。</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>是否启用 MKL-DNN 加速推理。如果 MKL-DNN 不可用或模型不支持通过 MKL-DNN 加速，即使设置了此标志，也不会使用加速。
</td>
<td><code>bool</code></td>
<td><code>True</code></td>
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
<td><code>thread_num</code></td>
<td>在 CPU 上进行推理时使用的线程数。实例化相应数量的推理实例并发执行，根据硬件资源合理设置。如果不设置，默认值为1。</td>
<td><code>int</code></td>
<td><code>1</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>PaddleX产线配置文件路径。</td>
<td><code>str</code></td>
<td></td>
</tr>
</tbody>
</table>

- 模块开关

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
<td>是否加载并使用文档方向分类模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>是否加载并使用文本图像矫正模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>是否加载并使用文本行方向模块。如果不设置，将使用产线初始化的该参数值，默认初始化为<code>True</code>。</td>
<td><code>bool</code></td>
<td></td>
</tr>
</tbody>
</table>

- 检测模型相关

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
<td></td>
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
<td></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>文本检测的边长度限制类型。支持 <code>min</code> 和 <code>max</code>，<code>min</code> 表示保证图像最短边不小于 <code>det_limit_side_len</code>，<code>max</code> 表示保证图像最长边不大于 <code>limit_side_len</code>。如果不设置，将使用产线初始化的该参数值，默认初始化为 <code>min</code>。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>文本检测像素阈值，输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点。
大于<code>0</code>的任意浮点数。如果不设置，将使用产线初始化的该参数值（默认为 <code>0.3</code>）。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>文本检测框阈值，检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域。
大于 <code>0</code> 的任意浮点数。如果不设置，将使用产线初始化的该参数值（默认为 <code>0.6</code>）。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>文本检测扩张系数，使用该方法对文字区域进行扩张，该值越大，扩张的面积越大。大于 <code>0</code> 的任意浮点数。如果不设置，将使用产线初始化的该参数值（默认为 <code>2.0</code>）。
</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_input_shape</code></td>
<td>文本检测的输入形状，您可以设置3个值代表C，H，W。</td>
<td><code>std::vector</code></td>
<td></td>
</tr>
</tbody>
</table>

- 方向分类器相关

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
<td><code></code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>文档方向分类模型的目录路径。当设置<code>use_doc_orientation_classify = false</code>时，可不添加。</td>
<td><code>str</code></td>
<td><code></code></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>文本行方向分类模型的名称。如果不设置，将会使用产线默认模型。当传入文本行方向分类模型与产线默认模型不一致时，需指定传入模型的名称。</td>
<td><code>str</code></td>
<td><code></code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>文本行方向分类模型的目录路径。当设置<code>use_textline_orientation = false</code>时，可不添加。</td>
<td><code>str</code></td>
<td><code></code></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>文本行方向模型的batch size。如果不设置，将会使用产线默认模型。</td>
<td><code>int</code></td>
<td><code></code></td>
</tr>
</tbody>
</table>

- 文字识别模型相关

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
<td><code></code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>文本识别模型的目录路径，必填。</td>
<td><code>str</code></td>
<td><code></code></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>文本识别模型的batch size。如果不设置，将会使用产线默认值。</td>
<td><code>int</code></td>
<td><code></code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>文本识别阈值，得分大于该阈值的文本结果会被保留。大于<code>0</code>的任意浮点数。</td>
<td><code>float</code></td>
<td><code></code></td>
</tr>
<tr>
<td><code>text_rec_input_shape</code></td>
<td>文本识别的输入形状，您可以设置3个值代表C，H，W。</td>
<td><code>std::vector</code></td>
<td><code></code></td>
</tr>
</tbody>
</table>

- 输入输出相关

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
<td>待预测数据，必填。仅支持<code>jpg</code>，<code>png</code>, <code>jpeg</code>,<code>bmp</code>格式的图像，暂不支持 PDF 文件。
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>指定推理结果文件保存的路径。如果不设置，推理结果将保存至当前运行路径下的<code>output</code>文件夹。</td>
<td><code>str</code></td>
<td></td>
</tr>
</tbody>
</table>

注意：命令行方式使用上述可调节参数，加前缀<code>--</code>，如：<code>--input your_image.jpg --save_path ./your_output/</code>。

</details>



## 3. 额外功能

### 3.1 可视化文本识别结果

我们需要 FreeType 去完成字体的渲染，所以需要自己编译包含 FreeType 的 OpenCV。
FreeType属于opencv_contrib模块，需要下载opencv和opencv_contrib源码，注意版本一致。以下以opencv4.7.0为例，源码下载命令如下。

```bash
wget https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip
wget https://github.com/opencv/opencv/archive/4.7.0.zip
unzip opencv.4.7.0.zip
unzip opencv_contrib.4.7.0.zip
```

安装FreeType依赖库

```bash
sudo apt-get update
sudo apt-get install libfreetype6-dev libharfbuzz-dev
```
编译OpenCV包含FreeType模块的命令如下：
相比于不编译FreeType方式，只需要增加如下三个参数：

- -DOPENCV_EXTRA_MODULES_PATH=your_opencv_contrib-4.7.0/modules/ \
- -DBUILD_opencv_freetype=ON \
- -DWITH_FREETYPE=ON

完整命令如下：
```shell
root_path="your_opencv_root_path"
install_path=${root_path}/opencv4
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
    -DBUILD_TIFF=ON \
    -DOPENCV_EXTRA_MODULES_PATH=your_opencv_contrib-4.7.0/modules/ \
    -DBUILD_opencv_freetype=ON \
    -DWITH_FREETYPE=ON

make -j
make install
```

也可以直接修改`tools/build_opencv.sh`的内容，然后直接运行下面的命令进行编译。

```shell
sh tools/build_opencv.sh
```
其中`root_path`为下载的opencv源码路径，`install_path`为opencv的安装路径，`make install`完成之后，会在该文件夹下生成opencv头文件和库文件，用于后面的OCR代码编译。

## 4. FAQ

1. TODO 
