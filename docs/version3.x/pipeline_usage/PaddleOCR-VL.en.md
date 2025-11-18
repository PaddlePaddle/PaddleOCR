---
comments: true
---

# PaddleOCR-VL Usage Tutorial

PaddleOCR-VL is an advanced and efficient document parsing model designed specifically for element recognition in documents. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful Vision-Language Model (VLM) composed of a NaViT-style dynamic resolution visual encoder and the ERNIE-4.5-0.3B language model, enabling precise element recognition. The model supports 109 languages and excels in recognizing complex elements (such as text, tables, formulas, and charts) while maintaining extremely low resource consumption. Comprehensive evaluations on widely used public benchmarks and internal benchmarks demonstrate that PaddleOCR-VL achieves SOTA performance in both page-level document parsing and element-level recognition. It significantly outperforms existing Pipeline-based solutions, document parsing multimodal schemes, and advanced general-purpose multimodal large models, while offering faster inference speeds. These advantages make it highly suitable for deployment in real-world scenarios.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl/metrics/allmetric.png"/>

## PaddleOCR-VL Inference Device Support

Currently, PaddleOCR-VL offers three inference methods, each with varying levels of support for inference devices. Please verify that your inference device meets the requirements in the table below before proceeding with PaddleOCR-VL inference deployment:

<table border="1">
<thead>
  <tr>
    <th>Inference Method</th>
    <th>x64 CPU Support</th>
    <th>GPU Compute Capability Support</th>
    <th>CUDA Version Support</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>PaddlePaddle</td>
    <td>✅</td>
    <td>≥ 7</td>
    <td>≥ 11.8</td>
  </tr>
  <tr>
    <td>vLLM</td>
    <td>🚧</td>
    <td>≥ 8 (RTX 3060, RTX 5070, A10, A100, ...) <br />  
    7 ≤ GPU Compute Capability < 8 (T4, V100, ...) is supported but may encounter request timeouts, OOM errors, or other abnormalities. Not recommended.
    </td>
    <td>≥ 12.6</td>
  </tr>
  <tr>
    <td>SGLang</td>
     <td>🚧</td>
    <td>8 ≤ GPU Compute Capability < 12</td>
    <td>≥ 12.6</td>
  </tr>
</tbody>
</table>

> Currently, PaddleOCR-VL does not support ARM architecture CPUs. Additional hardware support will be added based on actual demand in the future. Stay tuned!  
> vLLM and SGLang cannot run natively on Windows or macOS. Please use our provided Docker image instead.

Since different hardware configurations require different dependencies, if your hardware meets the requirements in the table above, please refer to the following table for the corresponding environment configuration tutorial:

<table border="1">
  <thead>
    <tr>
      <th>Hardware Type</th>
      <th>Hardware Model</th>
      <th>Environment Configuration Tutorial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">NVIDIA GPU</td>
      <td>RTX 30, 40 Series</td>
      <td>This usage tutorial</td>
    </tr>
    <tr>
      <td>RTX 50 Series</td>
      <td><a href="./PaddleOCR-VL-RTX50.en.md">PaddleOCR-VL RTX 50 Environment Configuration Tutorial</a></td>
    </tr>
    <tr>
      <td>x64 CPU</td>
      <td>-</td>
      <td>This usage tutorial</td>
    </tr>
    <tr>
      <td>XPU</td>
      <td>🚧</td>
      <td>🚧</td>
    </tr>
    <tr>
      <td>DCU</td>
      <td>🚧</td>
      <td>🚧</td>
    </tr>
  </tbody>
</table>

> For example, if you are using an RTX 50 Series GPU that meets the device requirements for PaddlePaddle and vLLM inference methods, please refer to the [PaddleOCR-VL RTX 50 Environment Configuration Tutorial](./PaddleOCR-VL-RTX50.en.md) to complete environment configuration before using PaddleOCR-VL.

## 1. Environment Preparation

This section explains how to set up the runtime environment for PaddleOCR-VL. Choose one of the following two methods:

- Method 1: Use the official Docker image.

- Method 2: Manually install PaddlePaddle and PaddleOCR.

### 1.1 Method 1: Using Docker Image

We recommend using the official Docker image (requires Docker version >= 19.03, GPU-equipped machine with NVIDIA drivers supporting CUDA 12.6 or later):

```shell
docker run \
    -it \
    --gpus all \
    --network host \
    --user root \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest \
    /bin/bash
# Invoke PaddleOCR CLI or Python API within the container
```

The image size is approximately 8 GB. If you need to use PaddleOCR-VL in an offline environment, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest` in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-offline` (offline image size is approximately 11 GB). You will need to pull the image on an internet-connected machine, import it into the offline machine, and then start the container using this image on the offline machine. For example:

```shell
# Execute on an internet-connected machine
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-offline
# Save the image to a file
docker save ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl:latest-offline -o paddleocr-vl-latest-offline.tar

# Transfer the image file to the offline machine

# Execute on the offline machine
docker load -i paddleocr-vl-latest-offline.tar
# After that, you can use `docker run` to start the container on the offline machine
```

### 1.2 Method 2: Manually Install PaddlePaddle and PaddleOCR

If you cannot use Docker, you can manually install PaddlePaddle and PaddleOCR. The required Python version is 3.8–3.12.

**We strongly recommend installing PaddleOCR-VL in a virtual environment to avoid dependency conflicts.** For example, use the Python venv standard library to create a virtual environment:

```shell
# Create a virtual environment
python -m venv .venv_paddleocr
# Activate the environment
source .venv_paddleocr/bin/activate
```

Run the following commands to complete the installation:

```shell
# The following command installs the PaddlePaddle version for CUDA 12.6. For other CUDA versions and the CPU version, please refer to https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
# For Linux systems, run:
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
# For Windows systems, run:
python -m pip install https://xly-devops.cdn.bcebos.com/safetensors-nightly/safetensors-0.6.2.dev0-cp38-abi3-win_amd64.whl
```

> **Please ensure that you install PaddlePaddle framework version 3.2.1 or above, along with the special version of safetensors.** For macOS users, please use Docker to set up the environment.

## 2. Quick Start

PaddleOCR-VL supports two usage methods: CLI command line and Python API. The CLI command line method is simpler and suitable for quickly verifying functionality, while the Python API method is more flexible and suitable for integration into existing projects.

> The methods introduced in this section are primarily for rapid validation. Their inference speed, memory usage, and stability may not meet the requirements of a production environment. **If deployment to a production environment is needed, we strongly recommend using a dedicated inference acceleration framework**. For specific methods, please refer to the next section.

### 2.1 Command Line Usage

Run a single command to quickly test the PaddleOCR-VL ：

```shell
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png

# Use --use_doc_orientation_classify to enable document orientation classification
paddleocr doc_parser -i ./paddleocr_vl_demo.png --use_doc_orientation_classify True

# Use --use_doc_unwarping to enable document unwarping module
paddleocr doc_parser -i ./paddleocr_vl_demo.png --use_doc_unwarping True

# Use --use_layout_detection to enable layout detection
paddleocr doc_parser -i ./paddleocr_vl_demo.png --use_layout_detection False
```

<details><summary><b>Command line supports more parameters. Click to expand for detailed parameter descriptions</b></summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, required.
For example, the local path of an image file or PDF file: <code>/root/data/img.jpg</code>;<b>Such as a URL link</b>, for example, the network URL of an image file or PDF file:<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>;<b>Such as a local directory</b>, which should contain the images to be predicted, for example, the local path: <code>/root/data/</code>(Currently, prediction for directories containing PDF files is not supported. PDF files need to be specified with a specific file path).</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>Specify the path where the inference result file will be saved. If not set, the inference results will not be saved locally.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>Name of the layout area detection and ranking model. If not set, the default model of the production line will be used.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>Directory path of the layout area detection and ranking model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Score threshold for the layout model. Any value between  <code>0-1</code>. If not set, the default value is used, which is  <code>0.5</code>.
</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use post-processing NMS for layout detection. If not set, the initialized default value will be used.</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Expansion coefficient for the detection boxes of the layout area detection model.Any floating-point number greater than <code>0</code>. If not set, the initialized default value will be used.</td>
<td><code>float</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merging mode for the detection boxes output by the model in layout detection.
<ul>
<li><b>large</b> when set to large, it means that among the detection boxes output by the model, for overlapping and contained boxes, only the outermost largest box is retained, and the overlapping inner boxes are deleted;</li>
<li><b>small</b>, when set to small, it means that among the detection boxes output by the model, for overlapping and contained boxes, only the innermost contained small box is retained, and the overlapping outer boxes are deleted;</li>
<li><b>union</b>,no filtering is performed on the boxes, and both inner and outer boxes are retained;</li></ul>
If not set, the initialized parameter value will be used.
</td>
<td><code>str</code></td>
<tr>
<td><code>vl_rec_model_name</code></td>
<td>Name of the multimodal recognition model. If not set, the default model will be used.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>vl_rec_model_dir</code></td>
<td>Directory path of the multimodal recognition model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>vl_rec_backend</code></td>
<td>Inference backend used by the multimodal recognition model.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>vl_rec_server_url</code></td>
<td>If the multimodal recognition model uses an inference service, this parameter is used to specify the server URL.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>vl_rec_max_concurrency</code></td>
<td>If the multimodal recognition model uses an inference service, this parameter is used to specify the maximum number of concurrent requests.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If not set, the initialized default value will be used.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>Directory path of the document orientation classification model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>Name of the text image rectification model. If not set, the initialized default value will be used.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>Directory path of the text image rectification model. If not set, the official model will be downloaded.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load and use the document orientation classification module. If not set, the initialized default value will be used, which is initialized to<code>False</code>.</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load and use the text image rectification module. If not set, the initialized default value will be used, which is initialized to <code>False.</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to load and use the layout area detection and ranking module. If not set, the initialized default value will be used, which is initialized to <code>True</code>.</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to use the chart parsing function. If not set, the initialized default value will be used, which is initialized to <code>False</code>.</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>format_block_content</code></td>
<td>Controls whether to format the <code>block_content</code> content within as Markdown. If not set, the initialized default value will be used, which defaults to initialization as<code>False</code>.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>use_queues</code></td>
<td>Used to control whether to enable internal queues. When set to <code>True</code>, data loading (such as rendering PDF pages as images), layout detection model processing, and VLM inference will be executed asynchronously in separate threads, with data passed through queues, thereby improving efficiency. This approach is particularly efficient for PDF documents with a large number of pages or directories containing a large number of images or PDF files.</td>
<td><code>bool</code></td>
<td></td>
</tr>
<tr>
<td><code>prompt_label</code></td>
<td>The prompt type setting for the VL model, which takes effect if and only if <code>use_layout_detection=False</code>.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>repetition_penalty</code></td>
<td>The repetition penalty parameter used in VL model sampling.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>temperature</code></td>
<td>The temperature parameter used in VL model sampling.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>top_p</code></td>
<td>The top-p parameter used in VL model sampling.</td>
<td><code>float</code></td>
<td></td>
</tr>
<tr>
<td><code>min_pixels</code></td>
<td>The minimum number of pixels allowed when the VL model preprocesses images.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>max_pixels</code></td>
<td>The maximum number of pixels allowed when the VL model preprocesses images.</td>
<td><code>int</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Supports specifying specific card numbers:<ul>
<li><b>CPU</b>: For example,<code>cpu</code> indicates using the CPU for inference;</li>
<li><b>GPU</b>: For example,<code>gpu:0</code> indicates using the first GPU for inference;</li>
<li><b>NPU</b>: For example,<code>npu:0</code> indicates using the first NPU for inference;</li>
<li><b>XPU</b>: For example,<code>xpu:0</code> indicates using the first XPU for inference;</li>
<li><b>MLU</b>: For example,<code>mlu:0</code> indicates using the first MLU for inference;</li>
<li><b>DCU</b>: For example,<code>dcu:0</code> indicates using the first DCU for inference;</li>
</ul>If not set, the initialized default value will be used. During initialization, the local GPU device 0 will be used preferentially. If it is not available, the CPU device will be used.</td>
<td><code>str</code></td>
<td></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable high-performance inference.</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to enable the TensorRT subgraph engine of Paddle Inference. If the model does not support acceleration via TensorRT, acceleration will not be used even if this flag is set.<br/>For PaddlePaddle version with CUDA 11.8, the compatible TensorRT version is 8.x (x&amp;gt;=6). It is recommended to install TensorRT 8.6.1.6.<br/>
</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computational precision, such as fp32, fp16.</td>
<td><code>str</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN accelerated inference. If MKL-DNN is not available or the model does not support acceleration via MKL-DNN, acceleration will not be used even if this flag is set.</td>
<td><code>bool</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>MKL-DNN cache capacity.</td>
<td><code>int</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>The number of threads used for inference on the CPU.</td>
<td><code>int</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>The file path for PaddleX production line configuration.</td>
<td><code>str</code></td>
<td></td>
</tr>
</tbody>
</table>
</details>
<br/>

The inference result will be printed in the terminal. The default output of the PP-StructureV3 pipeline is as follows:

<details><summary> 👉Click to expand</summary>
<pre>
 <code>
{'res': {'input_path': 'paddleocr_vl_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_chart_recognition': False, 'format_block_content': False}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 6, 'label': 'doc_title', 'score': 0.9636914134025574, 'coordinate': [np.float32(131.31366), np.float32(36.450516), np.float32(1384.522), np.float32(127.984665)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9281806349754333, 'coordinate': [np.float32(585.39465), np.float32(158.438), np.float32(930.2184), np.float32(182.57469)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840355515480042, 'coordinate': [np.float32(9.023666), np.float32(200.86115), np.float32(361.41583), np.float32(343.8828)]}, {'cls_id': 14, 'label': 'image', 'score': 0.9871416091918945, 'coordinate': [np.float32(775.50574), np.float32(200.66502), np.float32(1503.3807), np.float32(684.9304)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9801855087280273, 'coordinate': [np.float32(9.532196), np.float32(344.90594), np.float32(361.4413), np.float32(440.8244)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9708921313285828, 'coordinate': [np.float32(28.040405), np.float32(455.87976), np.float32(341.7215), np.float32(520.7117)]}, {'cls_id': 24, 'label': 'vision_footnote', 'score': 0.9002962708473206, 'coordinate': [np.float32(809.0692), np.float32(703.70044), np.float32(1488.3016), np.float32(750.5238)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9825374484062195, 'coordinate': [np.float32(8.896561), np.float32(536.54895), np.float32(361.05237), np.float32(655.8058)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9822263717651367, 'coordinate': [np.float32(8.971573), np.float32(657.4949), np.float32(362.01715), np.float32(774.625)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9767460823059082, 'coordinate': [np.float32(9.407074), np.float32(776.5216), np.float32(361.31067), np.float32(846.82874)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9868153929710388, 'coordinate': [np.float32(8.669495), np.float32(848.2543), np.float32(361.64703), np.float32(1062.8568)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9826608300209045, 'coordinate': [np.float32(8.8025055), np.float32(1063.8615), np.float32(361.46588), np.float32(1182.8524)]}, {'cls_id': 22, 'label': 'text', 'score': 0.982555627822876, 'coordinate': [np.float32(8.820602), np.float32(1184.4663), np.float32(361.66394), np.float32(1302.4507)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9584776759147644, 'coordinate': [np.float32(9.170288), np.float32(1304.2161), np.float32(361.48898), np.float32(1351.7483)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9782056212425232, 'coordinate': [np.float32(389.1618), np.float32(200.38202), np.float32(742.7591), np.float32(295.65146)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9844875931739807, 'coordinate': [np.float32(388.73303), np.float32(297.18463), np.float32(744.00024), np.float32(441.3034)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9680547714233398, 'coordinate': [np.float32(409.39468), np.float32(455.89386), np.float32(721.7174), np.float32(520.9387)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9741666913032532, 'coordinate': [np.float32(389.71606), np.float32(536.8138), np.float32(742.7112), np.float32(608.00165)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840384721755981, 'coordinate': [np.float32(389.30988), np.float32(609.39636), np.float32(743.09247), np.float32(750.3231)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9845995306968689, 'coordinate': [np.float32(389.13272), np.float32(751.7772), np.float32(743.058), np.float32(894.8815)]}, {'cls_id': 22, 'label': 'text', 'score': 0.984852135181427, 'coordinate': [np.float32(388.83267), np.float32(896.0371), np.float32(743.58215), np.float32(1038.7345)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9804865717887878, 'coordinate': [np.float32(389.08478), np.float32(1039.9119), np.float32(742.7585), np.float32(1134.4897)]}, {'cls_id': 22, 'label': 'text', 'score': 0.986461341381073, 'coordinate': [np.float32(388.52643), np.float32(1135.8137), np.float32(743.451), np.float32(1352.0085)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9869391918182373, 'coordinate': [np.float32(769.8341), np.float32(775.66235), np.float32(1124.9813), np.float32(1063.207)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9822869896888733, 'coordinate': [np.float32(770.30383), np.float32(1063.938), np.float32(1124.8295), np.float32(1184.2192)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9689218997955322, 'coordinate': [np.float32(791.3042), np.float32(1199.3169), np.float32(1104.4521), np.float32(1264.6985)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9713128209114075, 'coordinate': [np.float32(770.4253), np.float32(1279.6072), np.float32(1124.6917), np.float32(1351.8672)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9236552119255066, 'coordinate': [np.float32(1153.9058), np.float32(775.5814), np.float32(1334.0654), np.float32(798.1581)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9857938885688782, 'coordinate': [np.float32(1151.5197), np.float32(799.28015), np.float32(1506.3619), np.float32(991.1156)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9820687174797058, 'coordinate': [np.float32(1151.5686), np.float32(991.91095), np.float32(1506.6023), np.float32(1110.8875)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9866049885749817, 'coordinate': [np.float32(1151.6919), np.float32(1112.1301), np.float32(1507.1611), np.float32(1351.9504)]}]}}}
</code></pre></details>

For explanation of the result parameters, refer to [2.2 Python Script Integration](#222-python-script-integration).

<b>Note: </b> The default model for the production line is relatively large, which may result in slower inference speed. It is recommended to use [inference acceleration frameworks to enhance VLM inference performance](#31-starting-the-vlm-inference-service) for faster inference.

### 2.2 Python Script Integration

The command line method is for quick testing and visualization. In actual projects, you usually need to integrate the model via code. You can perform pipeline inference with just a few lines of code as shown below:

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL()
# pipeline = PaddleOCRVL(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# pipeline = PaddleOCRVL(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# pipeline = PaddleOCRVL(use_layout_detection=False) # Use use_layout_detection to enable/disable layout detection module
output = pipeline.predict("./paddleocr_vl_demo.png")
for res in output:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the current image's structured result in JSON format
    res.save_to_markdown(save_path="output") ## Save the current image's result in Markdown format
```

For PDF files, each page will be processed individually and generate a separate Markdown file. If you want to convert the entire PDF to a single Markdown file, use the following method:

```python
from pathlib import Path
from paddleocr import PaddleOCRVL

input_file = "./your_pdf_file.pdf"
output_path = Path("./output")

pipeline = PaddleOCRVL()
output = pipeline.predict(input=input_file)

markdown_list = []
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)
```

**Note:**

- In the example code, the parameters `use_doc_orientation_classify` and  `use_doc_unwarping` are all set to `False` by default. These indicate that document orientation classification and document image unwarping are disabled. You can manually set them to `True` if needed.

The above Python script performs the following steps:

<details><summary>(1) Instantiate the production line object. Specific parameter descriptions are as follows:</summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>Name of the layout area detection and ranking model. If set to <code>None</code>, the default model of the production line will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>Directory path of the layout area detection and ranking model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Score threshold for the layout model.
<ul>
<li><b>float</b>: Any floating-point number between <code>0-1</code>;</li>
<li><b>dict</b>: <code>{0:0.1}</code> The key is the class ID, and the value is the threshold for that class;</li>
<li><b>None</b>: If set to <code>None</code>, the parameter value initialized by the production line will be used.</li>
</ul>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use post-processing NMS for layout detection. If set to <code>None</code>, the parameter value initialized by the production line will be used.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>
Expansion coefficient for the detection box of the layout area detection model.
<ul>
<li><b>float</b>: Any floating-point number greater than <code>0</code></li>
<li><b>Tuple[float,float]</b>: The respective expansion coefficients in the horizontal and vertical directions;</li>
<li><b>dict</b>: where the key of the dict is of <b>int</b> type, representing <code>cls_id</code>, and the value is of</code>tuple <code>type, such as</code>{0: (1.1, 2.0)}, indicating that the center of the detection box for class 0 output by the model remains unchanged, with the width expanded by 1.1 times and the height expanded by 2.0 times;</li>
<li><b>None</b>: If set to <code>None</code>, the parameter value initialized by the production line will be used.</li>
</ul>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code><ul>
<td>Merging mode for the detection boxes output by the model in layout detection.
<ul>
<li><b>large</b> when set to large, it means that among the detection boxes output by the model, for overlapping and contained boxes, only the outermost largest box is retained, and the overlapping inner boxes are deleted;</li>
<li><b>small</b>, when set to small, it means that among the detection boxes output by the model, for overlapping and contained boxes, only the innermost contained small box is retained, and the overlapping outer boxes are deleted;</li>
<li><b>union</b>,no filtering is performed on the boxes, and both inner and outer boxes are retained;</li></ul>
If not set, the initialized parameter value will be used.
</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_model_name</code></td>
<td>Name of the multimodal recognition model. If not set, the default model will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_model_dir</code></td>
<td>Directory path of the multimodal recognition model. If not set, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_backend</code></td>
<td>Inference backend used by the multimodal recognition model.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_server_url</code></td>
<td>If the multimodal recognition model uses an inference service, this parameter is used to specify the server URL.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>vl_rec_max_concurrency</code></td>
<td>If the multimodal recognition model uses an inference service, this parameter is used to specify the maximum number of concurrent requests.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>Name of the document orientation classification model. If not set, the initialized default value will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>Directory path of the document orientation classification model. If not set, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>Name of the text image rectification model. If not set, the initialized default value will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>Directory path of the text image rectification model. If not set, the official model will be downloaded.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load and use the document orientation classification module. If not set, the initialized default value will be used, which is initialized to<code>False</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load and use the text image rectification module. If not set, the initialized default value will be used, which is initialized to <code>False.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to load and use the layout area detection and ranking module. If not set, the initialized default value will be used, which is initialized to <code>True</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to use the chart parsing function. If not set, the initialized default value will be used, which is initialized to <code>False</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>format_block_content</code></td>
<td>Controls whether to format the <code>block_content</code> content within as Markdown. If not set, the initialized default value will be used, which defaults to initialization as<code>False</code>.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
<td></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. Supports specifying specific card numbers:<ul>
<li><b>CPU</b>: For example,<code>cpu</code> indicates using the CPU for inference;</li>
<li><b>GPU</b>: For example,<code>gpu:0</code> indicates using the first GPU for inference;</li>
<li><b>NPU</b>: For example,<code>npu:0</code> indicates using the first NPU for inference;</li>
<li><b>XPU</b>: For example,<code>xpu:0</code> indicates using the first XPU for inference;</li>
<li><b>MLU</b>: For example,<code>mlu:0</code> indicates using the first MLU for inference;</li>
<li><b>DCU</b>: For example,<code>dcu:0</code> indicates using the first DCU for inference;</li>
</ul>If not set, the initialized default value will be used. During initialization, the local GPU device 0 will be used preferentially. If it is not available, the CPU device will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to enable the TensorRT subgraph engine of Paddle Inference. If the model does not support acceleration via TensorRT, acceleration will not be used even if this flag is set.<br/>For PaddlePaddle version with CUDA 11.8, the compatible TensorRT version is 8.x (x&amp;gt;=6). It is recommended to install TensorRT 8.6.1.6.<br/>
</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computational precision, such as fp32, fp16.</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN accelerated inference. If MKL-DNN is not available or the model does not support acceleration via MKL-DNN, acceleration will not be used even if this flag is set.</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>MKL-DNN cache capacity.</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>The number of threads used for inference on the CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
<tr>
<td><code>paddlex_config</code></td>
<td>The file path for PaddleX production line configuration.</td>
<td><code>str</code></td>
<td><code>None</code></td>
<td></td>
</tr>
</tbody>
</table>
</details>

<details><summary>(2) Call the <code>predict()</code>method of the PaddleOCR-VL production line object for inference prediction. This method will return a list of results. Additionally, the production line also provides the <code>predict_iter()</code>Method. The two are completely consistent in terms of parameter acceptance and result return. The difference lies in that <code>predict_iter()</code>returns a <code>generator</code>, which can process and obtain prediction results step by step. It is suitable for scenarios involving large datasets or where memory conservation is desired. You can choose either of these two methods based on actual needs. Below are the parameters of the <code>predict()</code>method and their descriptions:</summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types. Required.<ul>
<li><b>Python Var</b>: such as <code>numpy.ndarray</code> representing image data</li>
<li><b>str</b>: such as the local path of an image file or PDF file: <code>/root/data/img.jpg</code>;<b>such as a URL link</b>, such as the network URL of an image file or PDF file:<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>;<b>such as a local directory</b>, which should contain the images to be predicted, such as the local path: <code>/root/data/</code>(Currently, prediction for directories containing PDF files is not supported. PDF files need to be specified with a specific file path)</li>
<li><b>list</b>: List elements should be of the aforementioned data types, such as <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"].</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module during inference. Setting it to <code>None</code> means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the text image rectification module during inference. Setting it to <code>None</code> means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to use the layout region detection and sorting module during inference. Setting it to <code>None</code> means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_chart_recognition</code></td>
<td>Whether to use the chart parsing module during inference. Setting it to <code>None</code> means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code>None</code> means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code>None</code> means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code>None</code> means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td><code>float|Tuple[float,float]|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code>None</code> means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_queues</code></td>
<td>Used to control whether to enable internal queues. When set to <code>True</code>, data loading (such as rendering PDF pages as images), layout detection model processing, and VLM inference will be executed asynchronously in separate threads, with data passed through queues, thereby improving efficiency. This approach is particularly efficient for PDF documents with many pages or directories containing a large number of images or PDF files.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>prompt_label</code></td>
<td>The prompt type setting for the VL model, which takes effect only when <code>use_layout_detection=False</code>. The fillable parameters are <code>ocr</code>、<code>formula</code>、<code>table</code> and <code>chart</code>.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>format_block_content</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code>None</code> means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>repetition_penalty</code></td>
<td>The repetition penalty parameter used for VL model sampling.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>temperature</code></td>
<td>Temperature parameter used for VL model sampling.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>top_p</code></td>
<td>Top-p parameter used for VL model sampling.</td>
<td><code>float|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>min_pixels</code></td>
<td>The minimum number of pixels allowed when the VL model preprocesses images.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>max_pixels</code></td>
<td>The maximum number of pixels allowed when the VL model preprocesses images.</td>
<td><code>int|None</code></td>
<td><code>None</code></td>
</tr>
</table>
</details>
<details><summary>(3) Process the prediction results: The prediction result for each sample is a corresponding Result object, supporting operations such as printing, saving as an image, and saving as a <code>json</code> file:</summary>
<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"> <code>print()</code></td>
<td rowspan="3">Print results to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation.</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code>  data, making it more readable. Only valid when <code>format_json</code> is <code>True</code>.</td>
<td><code>4</code></td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non- <code>ASCII</code> characters are escaped as <code>Unicode</code>. When set to <code>True</code>, all non- <code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only valid when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"> <code>save_to_json()</code></td>
<td rowspan="3">Save the results as a json format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file type naming.</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code>data, making it more readable. Only valid when <code>format_json</code>is <code>True</code>.</td>
<td><code>4</code></td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether non- <code>ASCII</code> characters are escaped as <code>Unicode</code>. When set to <code>True</code>, all non- <code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Only valid when <code>format_json</code> is <code>True</code>.</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the visualized images of each intermediate module in png format</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting directory or file paths.</td>
<td><code>None</code></td>
</tr>
<tr>
<td rowspan="3"> <code>save_to_markdown()</code></td>
<td rowspan="3">Save each page in an image or PDF file as a markdown format file separately</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file type naming</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>pretty</code></td>
<td><code>bool</code></td>
<td>Whether to beautify the <code>markdown</code> output results, centering charts, etc., to make the <code>markdown</code> rendering more aesthetically pleasing.</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>show_formula_number</code></td>
<td><code>bool</code></td>
<td>Control whether to retain formula numbers in <code>markdown</code>. When set to <code>True</code>, all formula numbers are retained; <code>False</code> retains only the formulas</td>
<td><code>False</code></td>
</tr>
<tr>
<tr>
<td><code>save_to_html()</code></td>
<td>Save the tables in the file as html format files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting directory or file paths.</td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>save_to_xlsx()</code></td>
<td>Save the tables in the file as xlsx format files</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The file path for saving, supporting directory or file paths.</td>
<td><code>None</code></td>
</tr>
</tr>
</table>


- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` The input path of the image or PDF to be predicted.

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`.

    - `model_settings`: `(Dict[str, bool])` Model parameters required for configuring PaddleOCR-VL.
        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-pipeline.
        - `use_layout_detection`: `(bool)` Controls whether to enable the layout detection module.
        - `use_chart_recognition`: `(bool)` Controls whether to enable the chart recognition function.
        - `format_block_content`: `(bool)` Controls whether to save the formatted markdown content in `JSON`.

    - `doc_preprocessor_res`: `(Dict[str, Union[List[float], str]])` A dictionary of document preprocessing results, which exists only when `use_doc_preprocessor=True`.
        - `input_path`: `(str)` The image path accepted by the document preprocessing sub-pipeline. When the input is a `numpy.ndarray`, it is saved as `None`; here, it is `None`.
        - `page_index`: `None`. Since the input here is a `numpy.ndarray`, the value is `None`.
        - `model_settings`: `(Dict[str, bool])` Model configuration parameters for the document preprocessing sub-pipeline.
          - `use_doc_orientation_classify`: `(bool)` Controls whether to enable the document image orientation classification sub-module.
          - `use_doc_unwarping`: `(bool)` Controls whether to enable the text image distortion correction sub-module.
        - `angle`: `(int)` The prediction result of the document image orientation classification sub-module. When enabled, it returns the actual angle value.

    - `parsing_res_list`: `(List[Dict])` A list of parsing results, where each element is a dictionary. The list order is the reading order after parsing.
        - `block_bbox`: `(np.ndarray)` The bounding box of the layout area.
        - `block_label`: `(str)` The label of the layout area, such as `text`, `table`, etc.
        - `block_content`: `(str)` The content within the layout area.
        - `block_id`: `(int)` The index of the layout area, used to display the layout sorting results.
        - `block_order` `(int)` The order of the layout area, used to display the layout reading order. For non-sorted parts, the default value is `None`.
- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since json files do not support saving numpy arrays, the `numpy.array` types within will be converted to list form.
    - `input_path`: `(str)` The input path of the image or PDF to be predicted.

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`.

    - `model_settings`: `(Dict[str, bool])` Model parameters required for configuring PaddleOCR-VL.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-pipeline.
        - `use_layout_detection`: `(bool)` Controls whether to enable the layout detection module.
        - `use_chart_recognition`: `(bool)` Controls whether to enable the chart recognition function.
        - `format_block_content`: `(bool)` Controls whether to save the formatted markdown content in `JSON`.

    - `doc_preprocessor_res`: `(Dict[str, Union[List[float], str]])` A dictionary of document preprocessing results, which exists only when `use_doc_preprocessor=True`.
        - `input_path`: `(str)` The image path accepted by the document preprocessing sub-pipeline. When the input is a `numpy.ndarray`, it is saved as `None`; here, it is `None`.
        - `page_index`: `None`. Since the input here is a `numpy.ndarray`, the value is `None`.
        - `model_settings`: `(Dict[str, bool])` Model configuration parameters for the document preprocessing sub-pipeline.
          - `use_doc_orientation_classify`: `(bool)` Controls whether to enable the document image orientation classification sub-module.
          - `use_doc_unwarping`: `(bool)` Controls whether to enable the text image distortion correction sub-module.
        - `angle`: `(int)` The prediction result of the document image orientation classification sub-module. When enabled, it returns the actual angle value.

    - `parsing_res_list`: `(List[Dict])` A list of parsing results, where each element is a dictionary. The list order represents the reading order after parsing.
        - `block_bbox`: `(np.ndarray)` The bounding box of the layout region.
        - `block_label`: `(str)` The label of the layout region, such as `text`, `table`, etc.
        - `block_content`: `(str)` The content within the layout region.
        - `block_id`: `(int)` The index of the layout region, used to display the layout sorting results.
        - `block_order` `(int)` The order of the layout region, used to display the layout reading order. For non-sorted parts, the default value is `None`.


- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, visualized images for layout region detection, global OCR, layout reading order, etc., will be saved. If a file is specified, it will be saved directly to that file. (Production lines typically contain many result images, so it is not recommended to directly specify a specific file path, as multiple images will be overwritten, retaining only the last one.)
- Calling the `save_to_markdown()` method will save the converted Markdown file to the specified `save_path`. The saved file path will be `save_path/{your_img_basename}.md`. If the input is a PDF file, it is recommended to directly specify a directory; otherwise, multiple markdown files will be overwritten.

Additionally, it also supports obtaining visualized images and prediction results with results through attributes, as follows:<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>json</code></td>
<td>Obtain the prediction <code>json</code>result in the format</td>
</tr>
<tr>
<td rowspan="2"> <code>img</code></td>
<td rowspan="2">obtain in the format of <code>dict</code>visualized image</td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3"> <code>markdown</code></td>
<td rowspan="3">obtain in the format of <code>dict</code>markdown result</td>
</tr>
<tr>
</tr>
<tr>
</tr>
</tbody>
</table>

- The prediction result obtained through the `json` attribute is data of dict type, with relevant content consistent with that saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is data of dict type. The keys are `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, with corresponding values being `Image.Image` objects: used to display visualized images of layout region detection, OCR, OCR text paragraphs, formulas, tables, and seal results, respectively. If optional modules are not used, the dict only contains `layout_det_res`.
- The prediction result returned by the `markdown` attribute is data of dict type. The keys are `markdown_texts`, `markdown_images`, and `page_continuation_flags`, with corresponding values being markdown text, images displayed in Markdown (`Image.Image` objects), and a bool tuple used to identify whether the first element on the current page is the start of a paragraph and whether the last element is the end of a paragraph, respectively.</details>

## 3. Enhancing VLM Inference Performance Using Inference Acceleration Frameworks

The inference performance under default configurations is not fully optimized and may not meet actual production requirements. This step primarily introduces how to use the vLLM and SGLang inference acceleration frameworks to enhance the inference performance of PaddleOCR-VL.

### 3.1 Launching the VLM Inference Service

There are two methods to launch the VLM inference service; choose either one:

- Method 1: Launch the service using the official Docker image.

- Method 2: Launch the service by manually installing dependencies via the PaddleOCR CLI.

#### 3.1.1 Method 1: Using Docker Image

PaddleOCR provides a Docker image (approximately 13 GB in size) for quickly launching the vLLM inference service. Use the following command to launch the service (requires Docker version >= 19.03, a machine equipped with a GPU, and NVIDIA drivers supporting CUDA 12.6 or higher):

```shell
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest \
    paddleocr genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

More parameters can be passed when launching the vLLM inference service; refer to the next subsection for supported parameters.

If you wish to launch the service in an environment without internet access, replace `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest` in the above command with the offline version image `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-offline`. The offline image is approximately 15 GB in size.

#### 3.1.2 Method 2: Installation and Usage via PaddleOCR CLI

Since inference acceleration frameworks may have dependency conflicts with the PaddlePaddle framework, it is recommended to install them in a virtual environment. Taking vLLM as an example:

```shell
# If there is an active virtual environment currently, deactivate it first using `deactivate`
# Create a virtual environment
python -m venv .venv_vlm
# Activate the environment
source .venv_vlm/bin/activate
# Install PaddleOCR
python -m pip install "paddleocr[doc-parser]"
# Install dependencies for the inference acceleration service
paddleocr install_genai_server_deps vllm
```

Usage of the `paddleocr install_genai_server_deps` command:

```shell
paddleocr install_genai_server_deps <inference acceleration framework name>
```

Currently supported framework names are `vllm` and `sglang`, corresponding to vLLM and SGLang, respectively.

The vLLM and SGLang installed via `paddleocr install_genai_server_deps` are both **CUDA 12.6** versions; ensure that your local NVIDIA drivers are consistent with or higher than this version.

> The `paddleocr install_genai_server_deps` command may require CUDA compilation tools such as nvcc during execution. If these tools are not available in your environment (e.g., when using the `paddleocr-vl` image), you can obtain a precompiled version of FlashAttention from [this repository](https://github.com/mjun0812/flash-attention-prebuild-wheels). Install the precompiled package before executing subsequent commands. For example, if you are in the `paddleocr-vl` image, execute `python -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.3.14/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-linux_x86_64.whl`.

After installation, you can launch the service using the `paddleocr genai_server` command:

```shell
paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8118
```

The parameters supported by this command are as follows:

| Parameter          | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `--model_name`     | Model name                                                    |
| `--model_dir`      | Model directory                                               |
| `--host`           | Server hostname                                               |
| `--port`           | Server port number                                             |
| `--backend`        | Backend name, i.e., the name of the inference acceleration framework used; options are `vllm` or `sglang` |
| `--backend_config` | Can specify a YAML file containing backend configurations      |

### 3.2 Client Usage Methods

After launching the VLM inference service, the client can call the service through PaddleOCR.

#### 3.2.1 CLI Invocation

Specify the backend type (`vllm-server` or `sglang-server`) using `--vl_rec_backend` and the service address using `--vl_rec_server_url`, for example:

```shell
paddleocr doc_parser --input paddleocr_vl_demo.png --vl_rec_backend vllm-server --vl_rec_server_url http://127.0.0.1:8118/v1
```

#### 3.2.2 Python API Invocation

Pass the `vl_rec_backend` and `vl_rec_server_url` parameters when creating a `PaddleOCRVL` object:

```python
pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8118/v1")
```

### 3.3 Performance Tuning

The default configurations are optimized for single NVIDIA A100 GPUs with exclusive client access and may not be suitable for other environments. If users encounter performance issues in actual use, the following optimization methods can be attempted.

#### 3.3.1 Server-Side Parameter Adjustment

Different inference acceleration frameworks support different parameters. Refer to their official documentation for available parameters and adjustment timing:

- [vLLM Official Parameter Tuning Guide](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- [SGLang Hyperparameter Tuning Documentation](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html)

The PaddleOCR VLM inference service supports parameter tuning through configuration files. The following example shows how to adjust the `gpu-memory-utilization` and `max-num-seqs` parameters for the vLLM server:

1. Create a YAML file `vllm_config.yaml` with the following content:

   ```yaml
   gpu-memory-utilization: 0.3
   max-num-seqs: 128
   ```

2. Specify the configuration file path when starting the service, for example, using the `paddleocr genai_server` command:

   ```shell
   paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --backend_config vllm_config.yaml
   ```

If using a shell that supports process substitution (like Bash), you can also pass configuration items directly without creating a configuration file:

```bash
paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --backend_config <(echo -e 'gpu-memory-utilization: 0.3\nmax-num-seqs: 128')
```

#### 3.3.2 Client-Side Parameter Adjustment

PaddleOCR groups sub-images from single or multiple input images and sends concurrent requests to the server, so the number of concurrent requests significantly impacts performance.

- For CLI and Python API, adjust the maximum number of concurrent requests using the `vl_rec_max_concurrency` parameter;
- For service deployment, modify the `VLRecognition.genai_config.max_concurrency` field in the configuration file.

When there is a 1:1 client-to-VLM inference service ratio and sufficient server resources, increasing concurrency can improve performance. If the server needs to support multiple clients or has limited computing resources, reduce concurrency to avoid resource overload and service abnormalities.

#### 3.3.3 Common Hardware Performance Tuning Recommendations

The following configurations are for scenarios with a 1:1 client-to-VLM inference service ratio.

**NVIDIA RTX 3060**

- **Server-Side**
  - vLLM: `gpu-memory-utilization=0.8`

## 4. Service Deployment

This step mainly introduces how to deploy PaddleOCR-VL as a service and invoke it. There are two methods; choose either one:

- Method 1: Deploy using Docker Compose (recommended).

- Method 2: Manually install dependencies for deployment.

Note that the PaddleOCR-VL service described in this section differs from the VLM inference service in the previous section: the latter is responsible for only one part of the complete process (i.e., VLM inference) and is called as an underlying service by the former.

### 4.1 Method 1: Deploy Using Docker Compose (Recommended)

You can obtain the Compose file and the environment variables configuration file from [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/compose.yaml) and [here](https://github.com/PaddlePaddle/PaddleOCR/blob/main/deploy/paddleocr_vl_docker/.env), respectively, and download them to your local machine. Then, in the directory where the files were just downloaded, execute the following command to start the server, which will listen on port **8080** by default:

```shell
# Must be executed in the directory containing the compose.yaml and .env files
docker compose up
```

After startup, you will see output similar to the following:

```text
paddleocr-vl-api             | INFO:     Started server process [1]
paddleocr-vl-api             | INFO:     Waiting for application startup.
paddleocr-vl-api             | INFO:     Application startup complete.
paddleocr-vl-api             | INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

This method accelerates VLM inference using the vLLM framework and is more suitable for production environment deployment. It requires the machine to be equipped with a GPU and NVIDIA drivers supporting CUDA 12.6 or higher.

Additionally, after starting the server using this method, no internet connection is required except for pulling the image. For offline environment deployment, you can first pull the images involved in the Compose file on an online machine, export and transfer them to the offline machine for import, and then start the service in the offline environment.

If you need to adjust production-related configurations (such as model path, batch size, deployment device, etc.), refer to Section 4.4.

### 4.2 Method 2: Manually Install Dependencies for Deployment

Execute the following command to install the service deployment plugin via the PaddleX CLI:

```shell
paddlex --install serving
```

Then, start the server using the PaddleX CLI:

```shell
paddlex --serve --pipeline PaddleOCR-VL
```

After startup, you will see output similar to the following, with the server listening on port **8080** by default:

```text
INFO:     Started server process [63108]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

The command-line options related to serving are as follows:

<table>
<thead>
<tr>
<th>Name</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>--pipeline</code></td>
<td>PaddleX pipeline registration name or pipeline configuration file path.</td>
</tr>
<tr>
<td><code>--device</code></td>
<td>Deployment device for the pipeline. By default, a GPU will be used if available; otherwise, a CPU will be used."</td>
</tr>
<tr>
<td><code>--host</code></td>
<td>Hostname or IP address to which the server is bound. Defaults to <code>0.0.0.0</code>.</td>
</tr>
<tr>
<td><code>--port</code></td>
<td>Port number on which the server listens. Defaults to <code>8080</code>.</td>
</tr>
<tr>
<td><code>--use_hpip</code></td>
<td>If specified, uses high-performance inference. Refer to the High-Performance Inference documentation for more information.</td>
</tr>
<tr>
<td><code>--hpi_config</code></td>
<td>High-performance inference configuration. Refer to the High-Performance Inference documentation for more information.</td>
</tr>
</tbody>
</table>

If you need to adjust pipeline configurations (such as model path, batch size, deployment device, etc.), you can specify the `--pipeline` parameter as a custom configuration file path. For the correspondence between PaddleOCR pipelines and PaddleX pipeline registration names, as well as how to obtain and modify PaddleX pipeline configuration files, please refer to [PaddleOCR and PaddleX](../paddleocr_and_paddlex.en.md). Furthermore, section 4.1.3 will introduce how to adjust the pipeline configuration based on common requirements.

### 4.3 Client-Side Invocation

Below are the API reference and examples of multi-language service invocation:

<details><summary>API Reference</summary>
<p>Main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is<code>200</code>, and the properties of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Fixed as <code>0</code>.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>Operation result.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the properties of the response body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>logId</code></td>
<td><code>string</code></td>
<td>The UUID of the request.</td>
</tr>
<tr>
<td><code>errorCode</code></td>
<td><code>integer</code></td>
<td>Error code. Same as the response status code.</td>
</tr>
<tr>
<td><code>errorMsg</code></td>
<td><code>string</code></td>
<td>Error description.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Perform layout parsing.</p>
<p><code>POST /layout-parsing</code></p>
<ul>
<li>The properties of the request body are as follows:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
<th>Required</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>file</code></td>
<td><code>string</code></td>
<td>The URL of an image file or PDF file accessible to the server, or the Base64-encoded result of the content of the aforementioned file types. By default, for PDF files with more than 10 pages, only the first 10 pages will be processed.<br/>To remove the page limit, add the following configuration to the production line configuration file:<pre> <code>Serving:
  extra:
    max_num_input_imgs: null</code></pre>
</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code>|<code>null</code></td>
<td>File type.<code>0</code> represents a PDF file,<code>1</code> represents an image file. If this property is not present in the request body, the file type will be inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_orientation_classify</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Please refer to the description of the <code>use_layout_detection</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>useChartRecognition</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Please refer to the description of the <code>use_chart_recognition</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code>|<code>object</code>|<code>null</code></td>
<td>Please refer to the description of the <code>layout_threshold</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Please refer to the description of the <code>layout_nms</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code>|<code>array</code>|<code>object</code>|<code>null</code></td>
<td>Please refer to the description of the <code>layout_unclip_ratio</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code>|<code>object</code>|<code>null</code></td>
<td>Please refer to the description of the <code>layout_merge_bboxes_mode</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>promptLabel</code></td>
<td><code>string</code>|<code>null</code></td>
<td>Please refer to the description of the <code>prompt_label</code> parameter in the  <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>formatBlockContent</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Please refer to the description of the <code>format_block_content</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>repetitionPenalty</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Please refer to the description of the <code>repetition_penalty</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>temperature</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Please refer to the description of the <code>temperature</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>topP</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Please refer to the description of the <code>top_p</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>minPixels</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Please refer to the description of the <code>min_pixels</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>maxPixels</code></td>
<td><code>number</code>|<code>null</code></td>
<td>Please refer to the description of the <code>max_pixels</code> parameter in the <code>predict</code> method of the PaddleOCR-VL object.</td>
<td>No</td>
</tr>
<tr>
<td><code>prettifyMarkdown</code></td>
<td><code>boolean</code></td>
<td>Whether to output beautified Markdown text. The default is <code>true</code>.</td>
<td>No</td>
</tr>
<tr>
<td><code>showFormulaNumber</code></td>
<td><code>boolean</code></td>
<td>Whether to include formula numbers in the output Markdown text. The default is <code>false</code>.</td>
<td>No</td>
</tr>
<tr>
<td><code>visualize</code></td>
<td><code>boolean</code>|<code>null</code></td>
<td>Whether to return visualization result images and intermediate images during the processing.<ul style="margin: 0 0 0 1em; padding-left: 0em;">
<li>Pass <code>true</code>: Return images.</li>
<li>Pass <code>false</code>: Do not return images.</li>
<li>If this parameter is not provided in the request body or <code>null</code> is passed: Follow the setting in the configuration file <code>Serving.visualize</code>.</li>
</ul>
<br/>For example, add the following field in the configuration file:<br/>
<pre><code>Serving:
  visualize: False</code></pre>Images will not be returned by default, and the default behavior can be overridden by the <code>visualize</code> parameter in the request body. If this parameter is not set in either the request body or the configuration file (or <code>null</code> is passed in the request body and the configuration file is not set), images will be returned by default.</td>
<td>No</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is processed successfully, the <code>result</code> in the response body has the following attributes:</li>
</ul>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>layoutParsingResults</code></td>
<td><code>array</code></td>
<td>Layout parsing results. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each actual page processed in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Input data information.</td>
</tr>
</tbody>
</table>
<p>Each element in<code>layoutParsingResults</code> is an <code>object</code> with the following attributes:</p>
<table>
<thead>
<tr>
<th>Meaning</th>
<th>Name</th>
<th>Type</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>A simplified version of the <code>res</code> field in the JSON representation of the results generated by the <code>predict</code> method of the object, with the <code>input_path</code> and <code>page_index</code> fields removed.</td>
</tr>
<tr>
<td><code>markdown</code></td>
<td><code>object</code></td>
<td>Markdown results.</td>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code>|<code>null</code></td>
<td>Refer to the <code>img</code> property description of the prediction results. The image is in JPEG format and encoded using Base64.</td>
</tr>
<tr>
<td><code>inputImage</code></td>
<td><code>string</code>|<code>null</code></td>
<td>Input image. The image is in JPEG format and encoded using Base64.</td>
</tr>
</tbody>
</table>
<p><code>markdown</code>is an <code>object</code>with the following properties:</p>
<table>
<thead>
<tr>
<th>Name</th>
<th>Type</th>
<th>Meaning</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>text</code></td>
<td><code>string</code></td>
<td>Markdown text.</td>
</tr>
<tr>
<td><code>images</code></td>
<td><code>object</code></td>
<td>Key-value pairs of relative paths to Markdown images and Base64-encoded images.</td>
</tr>
<tr>
<td><code>isStart</code></td>
<td><code>boolean</code></td>
<td>Whether the first element on the current page is the start of a paragraph.</td>
</tr>
<tr>
<td><code>isEnd</code></td>
<td><code>boolean</code></td>
<td>Whether the last element on the current page is the end of a paragraph.</td>
</tr>
</tbody>
</table></details>
<details><summary>Multi-Language Service Invocation Examples</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">
import base64
import requests
import pathlib

API_URL = "http://localhost:8080/layout-parsing" # Service URL

image_path = "./demo.jpg"

# Encode the local image in Base64
with open(image_path, "rb") as file:
    image_bytes = file.read()
    image_data = base64.b64encode(image_bytes).decode("ascii")

payload = {
    "file": image_data, # Base64-encoded file content or file URL
    "fileType": 1, # File type, 1 indicates an image file
}

# Call the API
response = requests.post(API_URL, json=payload)

# Process the returned data from the interface
assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["layoutParsingResults"]):
    print(res["prunedResult"])
    md_dir = pathlib.Path(f"markdown_{i}")
    md_dir.mkdir(exist_ok=True)
    (md_dir / "doc.md").write_text(res["markdown"]["text"])
    for img_path, img in res["markdown"]["images"].items():
        img_path = md_dir / img_path
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(base64.b64decode(img))
    print(f"Markdown document saved at {md_dir / 'doc.md'}")
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        pathlib.Path(img_path).parent.mkdir(exist_ok=True)
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>

<details><summary>C++</summary>

<pre><code class="language-cpp">#include &lt;iostream&gt;
#include &lt;filesystem&gt;
#include &lt;fstream&gt;
#include &lt;vector&gt;
#include &lt;string&gt;
#include "cpp-httplib/httplib.h" // https://github.com/Huiyicc/cpp-httplib
#include "nlohmann/json.hpp" // https://github.com/nlohmann/json
#include "base64.hpp" // https://github.com/tobiaslocker/base64

namespace fs = std::filesystem;

int main() {
    httplib::Client client("localhost", 8080);

    const std::string filePath = "./demo.jpg";

    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return 1;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Error reading file." << std::endl;
        return 1;
    }

    std::string bufferStr(buffer.data(), static_cast<size_t>(size));
    std::string encodedFile = base64::to_base64(bufferStr);

    nlohmann::json jsonObj;
    jsonObj["file"] = encodedFile;
    jsonObj["fileType"] = 1;

    auto response = client.Post("/layout-parsing", jsonObj.dump(), "application/json");

    if (response && response->status == 200) {
        nlohmann::json jsonResponse = nlohmann::json::parse(response->body);
        auto result = jsonResponse["result"];

        if (!result.is_object() || !result.contains("layoutParsingResults")) {
            std::cerr << "Unexpected response format." << std::endl;
            return 1;
        }

        const auto& results = result["layoutParsingResults"];
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& res = results[i];

            if (res.contains("prunedResult")) {
                std::cout << "Layout result [" << i << "]: " << res["prunedResult"].dump() << std::endl;
            }

            if (res.contains("outputImages") && res["outputImages"].is_object()) {
                for (auto& [imgName, imgBase64] : res["outputImages"].items()) {
                    std::string outputPath = imgName + "_" + std::to_string(i) + ".jpg";
                    fs::path pathObj(outputPath);
                    fs::path parentDir = pathObj.parent_path();
                    if (!parentDir.empty() && !fs::exists(parentDir)) {
                        fs::create_directories(parentDir);
                    }

                    std::string decodedImage = base64::from_base64(imgBase64.get<std::string>());

                    std::ofstream outFile(outputPath, std::ios::binary);
                    if (outFile.is_open()) {
                        outFile.write(decodedImage.c_str(), decodedImage.size());
                        outFile.close();
                        std::cout << "Saved image: " << outputPath << std::endl;
                    } else {
                        std::cerr << "Failed to save image: " << outputPath << std::endl;
                    }
                }
            }
        }
    } else {
        std::cerr << "Request failed." << std::endl;
        if (response) {
            std::cerr << "HTTP status: " << response->status << std::endl;
            std::cerr << "Response body: " << response->body << std::endl;
        }
        return 1;
    }

    return 0;
}
</code></pre></details>

<details><summary>Java</summary>

<pre><code class="language-java">import okhttp3.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Base64;
import java.nio.file.Paths;
import java.nio.file.Files;

public class Main {
    public static void main(String[] args) throws IOException {
        String API_URL = "http://localhost:8080/layout-parsing";
        String imagePath = "./demo.jpg";

        File file = new File(imagePath);
        byte[] fileContent = java.nio.file.Files.readAllBytes(file.toPath());
        String base64Image = Base64.getEncoder().encodeToString(fileContent);

        ObjectMapper objectMapper = new ObjectMapper();
        ObjectNode payload = objectMapper.createObjectNode();
        payload.put("file", base64Image);
        payload.put("fileType", 1);

        OkHttpClient client = new OkHttpClient();
        MediaType JSON = MediaType.get("application/json; charset=utf-8");

        RequestBody body = RequestBody.create(JSON, payload.toString());

        Request request = new Request.Builder()
                .url(API_URL)
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (response.isSuccessful()) {
                String responseBody = response.body().string();
                JsonNode root = objectMapper.readTree(responseBody);
                JsonNode result = root.get("result");

                JsonNode layoutParsingResults = result.get("layoutParsingResults");
                for (int i = 0; i < layoutParsingResults.size(); i++) {
                    JsonNode item = layoutParsingResults.get(i);
                    int finalI = i;
                    JsonNode prunedResult = item.get("prunedResult");
                    System.out.println("Pruned Result [" + i + "]: " + prunedResult.toString());

                    JsonNode outputImages = item.get("outputImages");
                    outputImages.fieldNames().forEachRemaining(imgName -> {
                        try {
                            String imgBase64 = outputImages.get(imgName).asText();
                            byte[] imgBytes = Base64.getDecoder().decode(imgBase64);
                            String imgPath = imgName + "_" + finalI + ".jpg";
                            
                            File outputFile = new File(imgPath);
                            File parentDir = outputFile.getParentFile();
                            if (parentDir != null && !parentDir.exists()) {
                                parentDir.mkdirs();
                                System.out.println("Created directory: " + parentDir.getAbsolutePath());
                            }
                            
                            try (FileOutputStream fos = new FileOutputStream(outputFile)) {
                                fos.write(imgBytes);
                                System.out.println("Saved image: " + imgPath);
                            }
                        } catch (IOException e) {
                            System.err.println("Failed to save image: " + e.getMessage());
                        }
                    });
                }
            } else {
                System.err.println("Request failed with HTTP code: " + response.code());
            }
        }
    }
}
</code></pre></details>

<details><summary>Go</summary>

<pre><code class="language-go">package main

import (
    "bytes"
    "encoding/base64"
    "encoding/json"
    "fmt"
    "io/ioutil"
    "net/http"
    "os"
    "path/filepath"
)

func main() {
    API_URL := "http://localhost:8080/layout-parsing"
    filePath := "./demo.jpg"

    fileBytes, err := ioutil.ReadFile(filePath)
    if err != nil {
        fmt.Printf("Error reading file: %v\n", err)
        return
    }
    fileData := base64.StdEncoding.EncodeToString(fileBytes)

    payload := map[string]interface{}{
        "file":     fileData,
        "fileType": 1,
    }
    payloadBytes, err := json.Marshal(payload)
    if err != nil {
        fmt.Printf("Error marshaling payload: %v\n", err)
        return
    }

    client := &http.Client{}
    req, err := http.NewRequest("POST", API_URL, bytes.NewBuffer(payloadBytes))
    if err != nil {
        fmt.Printf("Error creating request: %v\n", err)
        return
    }
    req.Header.Set("Content-Type", "application/json")

    res, err := client.Do(req)
    if err != nil {
        fmt.Printf("Error sending request: %v\n", err)
        return
    }
    defer res.Body.Close()

    if res.StatusCode != http.StatusOK {
        fmt.Printf("Unexpected status code: %d\n", res.StatusCode)
        return
    }

    body, err := ioutil.ReadAll(res.Body)
    if err != nil {
        fmt.Printf("Error reading response: %v\n", err)
        return
    }

    type Markdown struct {
        Text   string            `json:"text"`
        Images map[string]string `json:"images"`
    }

    type LayoutResult struct {
        PrunedResult map[string]interface{} `json:"prunedResult"`
        Markdown     Markdown               `json:"markdown"`
        OutputImages map[string]string      `json:"outputImages"`
        InputImage   *string                `json:"inputImage"`
    }

    type Response struct {
        Result struct {
            LayoutParsingResults []LayoutResult `json:"layoutParsingResults"`
            DataInfo             interface{}    `json:"dataInfo"`
        } `json:"result"`
    }

    var respData Response
    if err := json.Unmarshal(body, &respData); err != nil {
        fmt.Printf("Error parsing response: %v\n", err)
        return
    }

    for i, res := range respData.Result.LayoutParsingResults {
        fmt.Printf("Result %d - prunedResult: %+v\n", i, res.PrunedResult)

        mdDir := fmt.Sprintf("markdown_%d", i)
        os.MkdirAll(mdDir, 0755)
        mdFile := filepath.Join(mdDir, "doc.md")
        if err := os.WriteFile(mdFile, []byte(res.Markdown.Text), 0644); err != nil {
            fmt.Printf("Error writing markdown file: %v\n", err)
        } else {
            fmt.Printf("Markdown document saved at %s\n", mdFile)
        }

        for path, imgBase64 := range res.Markdown.Images {
            fullPath := filepath.Join(mdDir, path)
            if err := os.MkdirAll(filepath.Dir(fullPath), 0755); err != nil {
                fmt.Printf("Error creating directory for markdown image: %v\n", err)
                continue
            }
            imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
            if err != nil {
                fmt.Printf("Error decoding markdown image: %v\n", err)
                continue
            }
            if err := os.WriteFile(fullPath, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving markdown image: %v\n", err)
            }
        }

        for name, imgBase64 := range res.OutputImages {
            imgBytes, err := base64.StdEncoding.DecodeString(imgBase64)
            if err != nil {
                fmt.Printf("Error decoding output image %s: %v\n", name, err)
                continue
            }
            filename := fmt.Sprintf("%s_%d.jpg", name, i)
            
            if err := os.MkdirAll(filepath.Dir(filename), 0755); err != nil {
                fmt.Printf("Error creating directory for output image: %v\n", err)
                continue
            }
            
            if err := os.WriteFile(filename, imgBytes, 0644); err != nil {
                fmt.Printf("Error saving output image %s: %v\n", filename, err)
            } else {
                fmt.Printf("Output image saved at %s\n", filename)
            }
        }
    }
}
</code></pre></details>

<details><summary>C#</summary>

<pre><code class="language-csharp">using System;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;

class Program
{
    static readonly string API_URL = "http://localhost:8080/layout-parsing";
    static readonly string inputFilePath = "./demo.jpg";

    static async Task Main(string[] args)
    {
        var httpClient = new HttpClient();

        byte[] fileBytes = File.ReadAllBytes(inputFilePath);
        string fileData = Convert.ToBase64String(fileBytes);

        var payload = new JObject
        {
            { "file", fileData },
            { "fileType", 1 }
        };
        var content = new StringContent(payload.ToString(), Encoding.UTF8, "application/json");

        HttpResponseMessage response = await httpClient.PostAsync(API_URL, content);
        response.EnsureSuccessStatusCode();

        string responseBody = await response.Content.ReadAsStringAsync();
        JObject jsonResponse = JObject.Parse(responseBody);

        JArray layoutParsingResults = (JArray)jsonResponse["result"]["layoutParsingResults"];
        for (int i = 0; i < layoutParsingResults.Count; i++)
        {
            var res = layoutParsingResults[i];
            Console.WriteLine($"[{i}] prunedResult:\n{res["prunedResult"]}");

            JObject outputImages = res["outputImages"] as JObject;
            if (outputImages != null)
            {
                foreach (var img in outputImages)
                {
                    string imgName = img.Key;
                    string base64Img = img.Value?.ToString();
                    if (!string.IsNullOrEmpty(base64Img))
                    {
                        string imgPath = $"{imgName}_{i}.jpg";
                        byte[] imageBytes = Convert.FromBase64String(base64Img);
                        
                        string directory = Path.GetDirectoryName(imgPath);
                        if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
                        {
                            Directory.CreateDirectory(directory);
                            Console.WriteLine($"Created directory: {directory}");
                        }
                        
                        File.WriteAllBytes(imgPath, imageBytes);
                        Console.WriteLine($"Output image saved at {imgPath}");
                    }
                }
            }
        }
    }
}
</code></pre></details>

<details><summary>Node.js</summary>

<pre><code class="language-js">const axios = require('axios');
const fs = require('fs');
const path = require('path');

const API_URL = 'http://localhost:8080/layout-parsing';
const imagePath = './demo.jpg';
const fileType = 1;

function encodeImageToBase64(filePath) {
  const bitmap = fs.readFileSync(filePath);
  return Buffer.from(bitmap).toString('base64');
}

const payload = {
  file: encodeImageToBase64(imagePath),
  fileType: fileType
};

axios.post(API_URL, payload)
  .then(response => {
    const results = response.data.result.layoutParsingResults;
    results.forEach((res, index) => {
      console.log(`\n[${index}] prunedResult:`);
      console.log(res.prunedResult);

      const outputImages = res.outputImages;
      if (outputImages) {
        Object.entries(outputImages).forEach(([imgName, base64Img]) => {
          const imgPath = `${imgName}_${index}.jpg`;
          
          const directory = path.dirname(imgPath);
          if (!fs.existsSync(directory)) {
            fs.mkdirSync(directory, { recursive: true });
            console.log(`Created directory: ${directory}`);
          }
          
          fs.writeFileSync(imgPath, Buffer.from(base64Img, 'base64'));
          console.log(`Output image saved at ${imgPath}`);
        });
      } else {
        console.log(`[${index}] No outputImages.`);
      }
    });
  })
  .catch(error => {
    console.error('Error during API request:', error.message || error);
  });
</code></pre></details>

<details><summary>PHP</summary>

<pre><code class="language-php">&lt;?php

$API_URL = "http://localhost:8080/layout-parsing";
$image_path = "./demo.jpg";

$image_data = base64_encode(file_get_contents($image_path));
$payload = array("file" => $image_data, "fileType" => 1);

$ch = curl_init($API_URL);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));
curl_setopt($ch, CURLOPT_HTTPHEADER, array('Content-Type: application/json'));
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
$response = curl_exec($ch);
curl_close($ch);

$result = json_decode($response, true)["result"]["layoutParsingResults"];

foreach ($result as $i => $item) {
    echo "[$i] prunedResult:\n";
    print_r($item["prunedResult"]);

    if (!empty($item["outputImages"])) {
        foreach ($item["outputImages"] as $img_name => $img_base64) {
            $output_image_path = "{$img_name}_{$i}.jpg";
            
            $directory = dirname($output_image_path);
            if (!is_dir($directory)) {
                mkdir($directory, 0777, true);
                echo "Created directory: $directory\n";
            }
            
            file_put_contents($output_image_path, base64_decode($img_base64));
            echo "Output image saved at $output_image_path\n";
        }
    } else {
        echo "No outputImages found for item $i\n";
    }
}
?&gt;
</code></pre></details>
</details>

### 4.4 Production Configuration Adjustment Instructions

> If you do not need to adjust production configurations, you can ignore this section.

Adjusting the PaddleOCR-VL configuration for service deployment involves only three steps:

1. Generate the configuration file
2. Modify the configuration file
3. Apply the configuration file

#### 4.4.1 Generate the Configuration File

```shell
paddlex --get_pipeline_config PaddleOCR-VL
```

#### 4.4.2 Modify the Configuration File

**Enhance VLM Inference Performance Using Acceleration Frameworks**

To improve VLM inference performance using acceleration frameworks such as vLLM (refer to Section 2 for detailed instructions on starting the VLM inference service), modify the `VLRecognition.genai_config.backend` and `VLRecognition.genai_config.server_url` fields in the production configuration file, as shown below:

```yaml
VLRecognition:
  ...
  genai_config:
    backend: vllm-server
    server_url: http://127.0.0.1:8118/v1
```

**Enable Document Image Preprocessing Functionality**

The service started with default configurations does not support document preprocessing. If a client attempts to invoke this functionality, an error message will be returned. To enable document preprocessing, set `use_doc_preprocessor` to `True` in the production configuration file and start the service using the modified configuration file.

**Disable Result Visualization Functionality**

The service returns visualized results by default, which introduces additional overhead. To disable this functionality, add the following configuration to the production configuration file:

```yaml
Serving:
  visualize: False
```

Additionally, you can set the `visualize` field to `false` in the request body to disable visualization for a single request.

**Configure Return of Image URLs**

For visualized result images and images included in Markdown, the service returns them in Base64 encoding by default. To return images as URLs instead, add the following configuration to the production configuration file:

```yaml
Serving:
  extra:
    file_storage:
      type: bos
      endpoint: https://bj.bcebos.com
      bucket_name: some-bucket
      ak: xxx
      sk: xxx
      key_prefix: deploy
    return_img_urls: True
```

Currently, storing generated images in Baidu Intelligent Cloud Object Storage (BOS) and returning URLs is supported. The parameters are described as follows:

- `endpoint`: Access domain name (required).
- `ak`: Baidu Intelligent Cloud Access Key (required).
- `sk`: Baidu Intelligent Cloud Secret Key (required).
- `bucket_name`: Storage bucket name (required).
- `key_prefix`: Unified prefix for object keys.
- `connection_timeout_in_mills`: Request timeout in milliseconds.

For more information on obtaining AK/SK and other details, refer to the [Baidu Intelligent Cloud Official Documentation](https://cloud.baidu.com/doc/BOS/index.html).

**Modify PDF Parsing Page Limit**

For performance considerations, the service processes only the first 10 pages of received PDF files by default. To adjust the page limit, add the following configuration to the production configuration file:

```yaml
Serving:
  extra:
    max_num_input_imgs: <new page limit, e.g., 100>
```

Set `max_num_input_imgs` to `null` to remove the page limit.

#### 4.4.3 Apply the Configuration File

**If you deployed using Docker Compose**:

Overwrite the custom production configuration file to `/home/paddleocr/pipeline_config.yaml` in the `ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-vl` container (or the corresponding container).

**If you deployed by manually installing dependencies**:

Specify the `--pipeline` parameter as the path to the custom configuration file.

## 5. Model Fine-Tuning

If you find that PaddleOCR-VL does not meet accuracy expectations in specific business scenarios, we recommend using the [ERNIEKit suite](https://github.com/PaddlePaddle/ERNIE/tree/release/v1.4) to perform supervised fine-tuning (SFT) on the PaddleOCR-VL-0.9B model. For detailed instructions, refer to the [ERNIEKit Official Documentation](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft.md).

> Currently, fine-tuning of layout detection and ranking models is not supported.
