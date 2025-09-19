# C++ Local Deployment for General OCR Pipeline - Linux

- [1. Environment Preparation](#1-environment-preparation)
    - [1.1 Compile OpenCV Library](#11-compile-opencv-library)
    - [1.2 Compile Paddle Inference](#12-compile-paddle-inference)
- [2. Getting Started](#2-getting-started)
    - [2.1 Compile Prediction Demo](#21-compile-prediction-demo)
    - [2.2 Prepare Models](#22-prepare-models)
    - [2.3 Run Prediction Demo](#23-run-prediction-demo)
    - [2.4 C++ API Integration](#24-c-api-integration)
- [3. Extended Features](#3-extended-features)
    - [3.1 Multilingual Text Recognition](#31-multilingual-text-recognition)
    - [3.2 Visualize Text Recognition Results](#32-visualize-text-recognition-results)
- [4. FAQ](#4-faq)

This section introduces the method for deploying a general OCR pipeline in C++. The general OCR pipeline consists of the following five modules:

1. Document Image Orientation Classification Module (Optional)
2. Text Image Unwarping Module (Optional)
3. Text Line Orientation Classification Module (Optional)
4. Text Detection Module
5. Text Recognition Module

Below, we will explain how to configure the C++ environment and complete the deployment of the general OCR pipeline in a Linux (CPU/GPU) environment.

- Note:
    - For specific compilation methods in a Windows environment, please refer to the [Windows Compilation Tutorial](./OCR_windows.en.md). After compilation, the subsequent commands for running the demo are the same as those in Linux.

## 1. Environment Preparation

- **The source code used for compilation and execution in this chapter can be found in the [PaddleOCR/deploy/cpp_infer](https://github.com/PaddlePaddle/PaddleOCR/tree/main/deploy/cpp_infer) directory.**

- Linux environment.
    - gcc 8.2 (when compiling with the Paddle Inference GPU version, gcc>=11.2)
    - cmake 3.18

### 1.1 Compile OpenCV Library

Currently, only OpenCV 4.x versions are supported. Below, we use OpenCV 4.7.0 as an example.

1. Execute the following commands to download the OpenCV source code:

```bash
cd deploy/cpp_infer
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv-4.7.0.tgz
tar -xf opencv-4.7.0.tgz
```

2. Configure and compile the OpenCV library:

- a. In the `tools/build_opencv.sh` script, set `root_path` to the absolute path of the opencv-4.7.0 source code.
- b. Set `install_path`, such as the default `${root_path}/opencv4`. `install_path` will be used as the path to the OpenCV library when compiling the prediction demo later.
- c. After configuration, run the following command to compile OpenCV:

    ```bash
    sh tools/build_opencv.sh
    ```

### 1.2 Compile Paddle Inference

You can choose to directly download a pre-compiled package or manually compile the source code.

#### 1.2.1 Directly Download Pre-compiled Package (Recommended)

The [Paddle Inference official website](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/download_lib.html) provides Linux prediction libraries. You can view and select the appropriate pre-compiled package on the website.

After downloading, extract it:

```shell
tar -xvf paddle_inference.tgz
```

This will generate a subfolder `paddle_inference/` in the current directory.

#### 1.2.2 Compile Prediction Library from Source Code

You can choose to compile the prediction library from source code. Compiling from source allows flexible configuration of various features and dependencies to adapt to different hardware and software environments. For detailed steps, please refer to [Source Code Compilation under Linux](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/compile/source_compile_under_Linux.html).

## 2. Getting Started

### 2.1 Compile Prediction Demo

Before compiling the prediction demo, ensure that you have compiled the OpenCV library and the Paddle Inference prediction library according to sections 1.1 and 1.2.

After modifying the configurations in `tools/build.sh`, execute the following command to compile:

```shell
sh tools/build.sh
```

Detailed descriptions of the relevant configuration parameters are as follows:

<table>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Default Value</th>
</tr>
<tr>
<td><code>OPENCV_DIR</code></td>
<td>The path where OpenCV is compiled and installed (such as the <code>install_path</code> mentioned when compiling OpenCV above, required).</td>
<td></td>
</tr>
<tr>
<td><code>LIB_DIR</code></td>
<td>The path to the downloaded <code>Paddle Inference</code> pre-compiled package or the manually compiled Paddle Inference library path (such as the <code>build/paddle_inference_install_dir</code> folder), required.</td>
<td></td>
</tr>
<tr>
<td><code>CUDA_LIB_DIR</code></td>
<td>The path to the CUDA library files, usually <code>/usr/local/cuda/lib64</code>. This parameter needs to be set when the Paddle Inference library is the GPU version and <code>-DWITH_GPU=ON</code> is set.</td>
<td></td>
</tr>
<tr>
<td><code>CUDNN_LIB_DIR</code></td>
<td>The path to the cuDNN library files, usually <code>/usr/lib/x86_64-linux-gnu/</code>. This parameter needs to be set when the Paddle Inference library is the GPU version and <code>-DWITH_GPU=ON</code> is set.</td>
<td></td>
</tr>
<tr>
<td><code>WITH_GPU</code></td>
<td>When set to ON, you can compile the GPU version demo, which requires the Paddle Inference library to be the GPU version.</td>
<td>OFF</td>
</tr>
</table>

**Note: The above paths need to be absolute paths.**

### 2.2 Prepare Models

You can directly download the inference models provided by PaddleOCR:

<details>
<summary><b>Document Image Orientation Classification Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a></td>
<td>99.06</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Image Unwrapping Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>CER</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a></td>
<td>0.179</td>
<td>30.3</td>
<td>A high-precision text image unwarping model.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Line Orientation Classification Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th>
<th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<td>PP-LCNet_x1_0_textline_ori (Default)</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar">Inference Model</a></td>
<td>99.42</td>
<td>6.5</td>
<td>A text line classification model based on PP-LCNet_x1_0, with two categories: 0 degrees and 180 degrees.</td>
</tr>
<tr>
<td>PP-LCNet_x0_25_textline_ori</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x0_25_textline_ori_infer.tar">Inference Model</a></td>
<td>98.85</td>
<td>0.96</td>
<td>A text line classification model based on PP-LCNet_x0_25, with two categories: 0 degrees and 180 degrees.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Detection Module:</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Detection Hmean (%)</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det (Default)</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">Inference Model</a></td>
<td>83.8</td>
<td>84.3</td>
<td>A server-side text detection model for PP-OCRv5, with higher precision, suitable for deployment on servers with better performance.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">Inference Model</a></td>
<td>79.0</td>
<td>4.7</td>
<td>A mobile-side text detection model for PP-OCRv5, with higher efficiency, suitable for deployment on edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">Inference Model</a></td>
<td>69.2</td>
<td>109</td>
<td>A server-side text detection model for PP-OCRv4, with higher precision, suitable for deployment on servers with better performance.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">Inference Model</a></td>
<td>63.8</td>
<td>4.7</td>
<td>A mobile-side text detection model for PP-OCRv4, with higher efficiency, suitable for deployment on edge devices.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Recognition Module:</b></summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Recognition Avg Accuracy (%)</th>
<th>Model Storage Size (MB)</th>
<th>Introduction</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec (Default)</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_server_rec_infer.tar">Inference Model</a></td>
<td>86.38</td>
<td>81</td>
<td rowspan="2">PP-OCRv5_rec is a new-generation text recognition model. It aims to efficiently and accurately support the recognition of four major languages: simplified Chinese, traditional Chinese, English, and Japanese, as well as complex text scenarios such as handwriting, vertical text, pinyin, and rare characters, with a single model. While maintaining recognition effectiveness, it balances inference speed and model robustness, providing efficient and accurate technical support for document understanding in various scenarios.</td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv5_mobile_rec_infer.tar">Inference Model</a></td>
<td>81.29</td>
<td>16</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a></td>
<td>86.58</td>
<td>182</td>
<td>PP-OCRv4_server_rec_doc is trained on a mixed dataset of more Chinese document data and PP-OCR training data, based on PP-OCRv4_server_rec. It enhances the recognition ability of some traditional Chinese characters, Japanese characters, and special characters, supporting over 15,000 characters. In addition to improving the text recognition ability related to documents, it also enhances the general text recognition ability.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a></td>
<td>78.74</td>
<td>10.5</td>
<td>A lightweight recognition model for PP-OCRv4, with high inference efficiency, suitable for deployment on various hardware devices including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a></td>
<td>85.19</td>
<td>173</td>
<td>A server-side model for PP-OCRv4, with high inference precision, suitable for deployment on various servers.</td>
</tr>
</tbody>
</table>
</details>

You can also refer to the model export sections of each module, such as [Text Detection Module - Model Export](../../module_usage/text_detection.en.md#44-model-export), to export the trained models as inference models.

The directory structure of the models is generally as follows:

```
PP-OCRv5_mobile_det
|–inference.pdiparams (Model weights file)
|–inference.json (Model structure file, in JSON format)
|–inference.yml (Model configuration file, in YAML format)
```

### 2.3 Run the Prediction Demo

Before using the General OCR Pipeline C++ locally, please first successfully compile the prediction demo. After compilation, you can experience it via the command line or call the API for secondary development and then recompile to generate the application.

**Please note that if you encounter issues such as the program becoming unresponsive, abnormal program exits, memory resource exhaustion, or extremely slow inference speeds during execution, try adjusting the configuration by referring to the documentation, for example, by disabling unused features or using a lighter model.**

This demo supports both system pipeline calls and individual module calls. Before running the following code, please download the [sample image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png) locally:

Running Method:

```shell
./build/ppocr <pipeline_or_module> [--param1] [--param2] [...]
```

Common parameters are as follows:

<li>Input and Output Related</li>

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>The local image to be predicted, required. Only supports images in <code>jpg</code>, <code>png</code>, <code>jpeg</code>, and <code>bmp</code> formats.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>Specifies the path where the inference result files will be saved. Both the JSON file and the predicted result image will be saved under this path.</td>
<td><code>str</code></td>
<td><code>./output</code></td>
</tr>
</tbody>
</table>

<details><summary><b>Click to expand for detailed descriptions of more parameters</b></summary>

<li>General Parameters</li>

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Supports specifying a specific card number:
<ul>
<li><b>CPU</b>: For example, <code>cpu</code> indicates using the CPU for inference;</li>
<li><b>GPU</b>: For example, <code>gpu:0</code> indicates using the first GPU for inference;</li>
</ul>If not set, it will use the default value initialized by the pipeline. During initialization, if <code>-DWITH_GPU=ON</code> is added during compilation, it will prioritize using the local GPU device 0; otherwise, it will use the CPU device.
</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>The computation precision, such as <code>fp32</code>, <code>fp16</code>.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN for accelerated inference. If MKL-DNN is not available or the model does not support acceleration via MKL-DNN, acceleration will not be used even if this flag is set.</td>
<td><code>bool</code></td>
<td><code>true</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN cache capacity.
</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>The number of threads for the PaddleInference CPU acceleration library.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>The path to the PaddleX pipeline configuration file.</td>
<td><code>str</code></td>
<td></td>
</tr>
</tbody>
</table>

<li>Module Switches</li>

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load and use the document orientation classification module. If not set, it will use the default value initialized by the pipeline, which is <code>true</code> by default.</td>
<td><code>bool</code></td>
<td><code>true</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load and use the text image correction module. If not set, it will use the default value initialized by the pipeline, which is <code>true</code> by default.</td>
<td><code>bool</code></td>
<td><code>true</code></td>
</tr>
<tr>
<td><code>use_textline_orientation</code></td>
<td>Whether to load and use the text line orientation module. If not set, it will use the default value initialized by the pipeline, which is <code>true</code> by default.</td>
<td><code>bool</code></td>
<td><code>true</code></td>
</tr>
</tbody>
</table>

<li>Detection Model Related</li>

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>text_detection_model_name</code></td>
<td>The name of the text detection model. If not set, it will use the default model of the pipeline. When the model name passed in via the text detection model path is inconsistent with the configured name of the pipeline's default text recognition model, you need to specify the name of the passed-in model.</td>
<td><code>str</code></td>
<td><code>PP-OCRv5_server_det</code></td>
</tr>
<tr>
<td><code>text_detection_model_dir</code></td>
<td>The directory path of the text detection model, required.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_det_limit_side_len</code></td>
<td>The image side length limit for text detection.
Any integer greater than <code>0</code>. If not set, it will use the default value initialized by the pipeline, which is <code>64</code> by default.
</td>
<td><code>int</code></td>
<td><code>64</code></td>
</tr>
<tr>
<td><code>text_det_limit_type</code></td>
<td>The side length limit type for text detection. Supports <code>min</code> and <code>max</code>, where <code>min</code> means ensuring the shortest side of the image is not less than <code>det_limit_side_len</code>, and <code>max</code> means ensuring the longest side of the image is not greater than <code>limit_side_len</code>. If not set, it will use the default value initialized by the pipeline, which is <code>min</code> by default.
</td>
<td><code>str</code></td>
<td><code>min</code></td>
</tr>
<tr>
<td><code>text_det_thresh</code></td>
<td>The pixel threshold for text detection. Only pixels with scores greater than this threshold in the output probability map will be considered as text pixels.
Any floating-point number greater than <code>0</code>. If not set, it will use the default value initialized by the pipeline.
</td>
<td><code>float</code></td>
<td><code>0.3</code></td>
</tr>
<tr>
<td><code>text_det_box_thresh</code></td>
<td>The bounding box threshold for text detection. When the average score of all pixels within the detected bounding box is greater than this threshold, the result will be considered as a text region.
Any floating-point number greater than <code>0</code>. If not set, it will use the default value initialized by the pipeline (default is <code>0.6</code>).
</td>
<td><code>float</code></td>
<td><code>0.6</code></td>
</tr>
<tr>
<td><code>text_det_unclip_ratio</code></td>
<td>The expansion coefficient for text detection. This method is used to expand the text region. The larger the value, the greater the expanded area. Any floating-point number greater than <code>0</code>. If not set, it will use the default value initialized by the pipeline.
</td>
<td><code>float</code></td>
<td><code>1.5</code></td>
</tr>
<tr>
<td><code>text_det_input_shape</code></td>
<td>The input shape for text detection. You can set 3 values representing C, H, W.</td>
<td><code>str</code></td>
<td>""</td>
</tr>
</tbody>
</table>

<li>Orientation Classifier Related</li>

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>The name of the document orientation classification model. If not set, it will use the default model of the pipeline. When the name of the passed-in document orientation classification model is inconsistent with the configured name of the pipeline's default model, you need to specify the name of the passed-in model.</td>
<td><code>str</code></td>
<td><code>PP-LCNet_x1_0_doc_ori</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>The directory path of the document orientation classification model. It can be omitted when setting <code>use_doc_orientation_classify = false</code>.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_model_name</code></td>
<td>The name of the text line orientation classification model. If not set, it will use the default model of the pipeline. When the name of the passed-in text line orientation classification model is inconsistent with the configured name of the pipeline's default model, you need to specify the name of the passed-in model.</td>
<td><code>str</code></td>
<td><code>PP-LCNet_x1_0_textline_ori</code></td>
</tr>
<tr>
<td><code>textline_orientation_model_dir</code></td>
<td>The directory path of the text line orientation classification model. It can be omitted when setting <code>use_textline_orientation = false</code>.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>textline_orientation_batch_size</code></td>
<td>The batch size for the text line orientation model. If not set, it will use the default model of the pipeline.</td>
<td><code>int</code></td>
<td><code>6</code></td>
</tr>
</tbody>
</table>

<li>Text Recognition Model Related</li>

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>text_recognition_model_name</code></td>
<td>The name of the text recognition model. If not set, it will use the default model of the pipeline. When the name of the passed-in text recognition model path is inconsistent with the configured name of the pipeline's default text recognition model, you need to specify the name of the passed-in model.</td>
<td><code>str</code></td>
<td><code>PP-OCRv5_server_rec</code></td>
</tr>
<tr>
<td><code>text_recognition_model_dir</code></td>
<td>The directory path of the text recognition model, required.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>text_recognition_batch_size</code></td>
<td>The batch size for the text recognition model. If not set, it will use the default value of the pipeline.</td>
<td><code>int</code></td>
<td><code>6</code></td>
</tr>
<tr>
<td><code>text_rec_score_thresh</code></td>
<td>The text recognition threshold. Text results with scores greater than this threshold will be retained. Any floating-point number greater than <code>0</code>.</td>
<td><code>float</code></td>
<td><code>0.0</code></td>
</tr>
<tr>
<td><code>text_rec_input_shape</code></td>
<td>The input shape for text recognition. You can set 3 values representing C, H, W.</td>
<td><code>str</code></td>
<td>""</td>
</tr>
</tbody>
</table>

</details>

#### 2.3.1 Example of System Pipeline Call

This section provides an example of a system pipeline call. Please refer to Section 2.1 to prepare the models. Assume the model directory structure is as follows:

```
models
|--PP-LCNet_x1_0_doc_ori_infer
|--UVDoc_infer
|--PP-LCNet_x1_0_textline_ori_infer
|--PP-OCRv5_server_det_infer
|--PP-OCRv5_server_rec_infer
```

=== "Full Pipeline Serialization"

    ```bash
    ./build/ppocr ocr --input ./general_ocr_002.png --save_path ./output/  \
    --doc_orientation_classify_model_dir models/PP-LCNet_x1_0_doc_ori_infer \
    --doc_unwarping_model_dir models/UVDoc_infer \
    --textline_orientation_model_dir models/PP-LCNet_x1_0_textline_ori_infer \
    --text_detection_model_dir models/PP-OCRv5_server_det_infer \
    --text_recognition_model_dir models/PP-OCRv5_server_rec_infer \
    --device cpu
    ```

    Example Output (If `save_path` is specified, a standard JSON prediction result file and prediction result image will be generated under this path):

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

=== "Text Detection + Textline Orientation Classification + Text Recognition"

    ```bash
    ./build/ppocr ocr --input ./general_ocr_002.png --save_path ./output/  \
    --doc_orientation_classify_model_dir models/PP-LCNet_x1_0_doc_ori_infer \
    --doc_unwarping_model_dir models/UVDoc_infer \
    --textline_orientation_model_dir models/PP-LCNet_x1_0_textline_ori_infer \
    --text_detection_model_dir models/PP-OCRv5_server_det_infer \
    --text_recognition_model_dir models/PP-OCRv5_server_rec_infer \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --device cpu
    ```

    Example Output (If `save_path` is specified, a standard JSON prediction result file and prediction result image will be generated under this path):

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

=== "Text Detection + Text Recognition"

    ```bash
    ./build/ppocr ocr --input ./general_ocr_002.png --save_path ./output/  \
    --doc_orientation_classify_model_dir models/PP-LCNet_x1_0_doc_ori_infer \
    --doc_unwarping_model_dir models/UVDoc_infer \
    --textline_orientation_model_dir models/PP-LCNet_x1_0_textline_ori_infer \
    --text_detection_model_dir models/PP-OCRv5_server_det_infer \
    --text_recognition_model_dir models/PP-OCRv5_server_rec_infer \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --device cpu
    ```

    Example Output (If `save_path` is specified, a standard JSON prediction result file and prediction result image will be generated under this path):

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

The above sample code will generate the following text detection result image:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/ocr_res.png"/>

If you want to view the text recognition result image, please refer to the **Visualizing Text Recognition Results** section later.

#### 2.3.2 Example of Single Module Call

=== "Document Image Orientation Classification"   

    ```bash
    ./build/ppocr doc_img_orientation_classification --input ./general_ocr_002.png --save_path ./output/  \
    --doc_orientation_classify_model_dir models/PP-LCNet_x1_0_doc_ori_infer \
    --device cpu 
    ```

    Example Output (if `save_path` is specified, standard JSON prediction result files and prediction result images will be generated at this path):

    ```bash
    {
    "res": {
        "input_path": {./general_ocr_002.png},
        "class_ids": {0},
        "scores": {0.926328},
        "label_names": {0},
    }
    ```

=== "Document Image Unwarping"

    ```bash
    ./build/ppocr text_image_unwarping --input ./general_ocr_002.png --save_path ./output/  \
    --doc_unwarping_model_dir models/UVDoc_infer \
    --device cpu 
    ```

    Example Output (if `save_path` is specified, standard JSON prediction result files and prediction result images will be generated at this path):

    ```bash
    {
    "res": {
        "input_path": {./general_ocr_002.png},
        "doctr_img": {...}
    }    
    ```   

=== "Text Line Orientation Classification"

    ```bash
    ./build/ppocr textline_orientation_classification --input ./general_ocr_002.png --save_path ./output/  \
    --textline_orientation_model_dir models/PP-LCNet_x1_0_textline_ori_infer \
    --device cpu 
    ```

    Example Output (if `save_path` is specified, standard JSON prediction result files and prediction result images will be generated at this path):

    ```bash
    {
    "res": {
        "input_path": {./general_ocr_002.png},
        "class_ids": {0},
        "scores": {0.719926},
        "label_names": {0_degree},
    }
    ```      

=== "Text Detection"

    ```bash
    ./build/ppocr text_detection --input ./general_ocr_002.png --save_path ./output/  \
    --text_detection_model_dir models/PP-OCRv5_server_det_infer \
    --device cpu 
    ```

    Example Output (if `save_path` is specified, standard JSON prediction result files and prediction result images will be generated at this path):

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
    
=== "Text Recognition"

    ```bash
    ./build/ppocr text_recognition --input ./general_ocr_rec_001.png --save_path ./output/  \
    --text_recognition_model_dir models/PP-OCRv5_server_rec_infer \ 
    --device cpu 
    ```

    Example Output (if `save_path` is specified, standard JSON prediction result files and prediction result images will be generated at this path):

    ```bash
    {
    "res": {
        "input_path": {./general_ocr_rec_001.png },
        "rec_text": {绿洲仕格维花园公寓 }
        "rec_score": {0.982409 }
    }
    ```

### 2.4 C++ API Integration

The command-line interface is for quickly experiencing and viewing the results. Generally, in projects, integration through code is often required. You can achieve rapid inference in production lines with just a few lines of code. The inference code is as follows:

Since the general OCR production line has many configuration parameters, a struct is used for parameter passing during instantiation. The naming rule for the struct is `pipeline_class_name + Params`. For example, the corresponding class name for the general OCR production line is `PaddleOCR`, and the struct is `PaddleOCRParams`.

```c++
#include "src/api/pipelines/ocr.h"


int main(){
    PaddleOCRParams params;
    params.doc_orientation_classify_model_dir = "models/PP-LCNet_x1_0_doc_ori_infer"; // Path to the document orientation classification model.
    params.doc_unwarping_model_dir = "models/UVDoc_infer"; // Path to the text image unwarping model.
    params.textline_orientation_model_dir = "models/PP-LCNet_x1_0_textline_ori_infer"; // Path to the text line orientation classification model.
    params.text_detection_model_dir = "models/PP-OCRv5_server_det_infer"; // Path to the text detection model.
    params.text_recognition_model_dir = "models/PP-OCRv5_server_rec_infer"; // Path to the text recognition model.

    // params.device = "gpu"; // Use GPU for inference. Ensure that the -DWITH_GPU=ON option is added during compilation; otherwise, CPU will be used.
    // params.use_doc_orientation_classify = false; // Do not use the document orientation classification model.
    // params.use_doc_unwarping = false; // Do not use the text image unwarping model.
    // params.use_textline_orientation = false; // Do not use the text line orientation classification model.
    // params.text_recognition_model_name = "PP-OCRv5_server_rec" // Use the PP-OCRv5_server_rec model for recognition.
    // params.vis_font_dir = "your_vis_font_dir"; // When the -DUSE_FREETYPE=ON option is added during compilation, the corresponding ttf font file path must be provided.

    auto infer = PaddleOCR(params);
    auto outputs = infer.Predict("./general_ocr_002.png");
    for (auto& output : outputs) {
      output->Print();
      output->SaveToImg("./output/");
      output->SaveToJson("./output/");
    }
}
```

## 3. Extended Features

### 3.1 Multilingual Text Recognition

PP-OCRv5 also provides multilingual text recognition capabilities covering 39 languages, including Korean, Spanish, French, Portuguese, German, Italian, Russian, Thai, Greek, etc. The specific supported languages are as follows:

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Link</th>
      <th>Supported Languages</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PP-OCRv5_server_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar">Inference Model</a></td>
      <td>Simplified Chinese, Traditional Chinese, English, Japanese</td>
    </tr>
    <tr>
      <td>PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar">Inference Model</a></td>
      <td>Simplified Chinese, Traditional Chinese, English, Japanese</td>
    </tr>
    <tr>
      <td>korean_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a></td>
      <td>Korean, English</td>
    </tr>
    <tr>
      <td>latin_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a></td>
      <td>English, French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin</td>
    </tr>
    <tr>
      <td>eslav_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/eslav_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a></td>
      <td>Russian, Belarusian, Ukrainian, English</td>
    </tr>
    <tr>
      <td>th_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/th_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a></td>
      <td>Thai, English</td>
    </tr>
    <tr>
      <td>el_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/el_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a></td>
      <td>Greek, English</td>
    </tr>
    <tr>
      <td>en_PP-OCRv5_mobile_rec</td>
      <td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a></td>
      <td>English</td>
    </tr>
  </tbody>
</table>

Simply pass the corresponding recognition model when using the pipeline or module. For example, to recognize French text using the text recognition module:

```
./build/ppocr text_recognition \
--input ./french.png \
--text_recognition_model_name latin_PP-OCRv5_mobile_rec \
--text_recognition_model_dir latin_PP-OCRv5_mobile_rec_infer \
--save_path ./output/
```

For more detailed information, refer to the [Introduction to PP-OCRv5 Multilingual Text Recognition](../../algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md).

### 3.2 Visualize Text Recognition Results

We use the FreeType module from the opencv_contrib 4.x version for font rendering. If you want to visualize text recognition results, you need to download the source code of both OpenCV and opencv_contrib and compile OpenCV with the FreeType module included. Make sure that the versions of the two are consistent when downloading the source code. The following instructions use opencv-4.7.0 and opencv_contrib-4.7.0 as examples:

```bash
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv-4.7.0.tgz
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/cpp/libs/opencv_contrib-4.7.0.tgz
tar -xf opencv-4.7.0.tgz
tar -xf opencv_contrib-4.7.0.tgz
```

Install FreeType dependencies:

```bash
sudo apt-get update
sudo apt-get install libfreetype6-dev libharfbuzz-dev
```

Follow these steps to compile OpenCV with FreeType support:

- a. Add the following three options to the `tools/build_opencv.sh` script:
    - `-DOPENCV_EXTRA_MODULES_PATH=your_opencv_contrib-4.7.0/modules/`
    - `-DBUILD_opencv_freetype=ON`
    - `-DWITH_FREETYPE=ON`
- b. In `tools/build_opencv.sh`, set `root_path` to the absolute path of your opencv-4.7.0 source code.
- c. In `tools/build_opencv.sh`, set `install_path` (default: `${root_path}/opencv4`). This path will be used as the OpenCV library path when compiling the prediction demo later.
- d. After configuration, run the following command to compile OpenCV:

    ```bash
    sh tools/build_opencv.sh
    ```

- e. In `tools/build.sh`, add `-DUSE_FREETYPE=ON` to enable text rendering and specify the ttf font file path with `--vis_font_dir your_ttf_path`. Then compile the prediction demo with:

    ```bash
    sh tools/build.sh
    ```

After compiling and running the prediction demo, you should see visualized text recognition results like this:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/deployment/cpp/ocr_res_with_freetype.png"/>

## 4. FAQ

1. If you encounter the error `Model name mismatch, please input the correct model dir. model dir is xxx, but model name is xxx`, it means the specified model name doesn't match the provided model. For example, if the text recognition model expects `PP-OCRv5_server_rec` but you provided `PP-OCRv5_mobile_rec`.
Solution: Adjust either the model name or the provided model. In the example above, you can specify `--text_recognition_model_name PP-OCRv5_mobile_rec` to match the provided model.

2. If you see garbled text in the Windows console, it may be due to the console's default character encoding (GBK). Change it to UTF-8 encoding.
