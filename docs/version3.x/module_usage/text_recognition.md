---
comments: true
---

# 文本识别模块使用教程

## 一、概述

文本识别模块是OCR（光学字符识别）系统中的核心部分，负责从图像中的文本区域提取出文本信息。该模块的性能直接影响到整个OCR系统的准确性和效率。文本识别模块通常接收文本检测模块输出的文本区域的边界框（Bounding Boxes）作为输入，然后通过复杂的图像处理和深度学习算法，将图像中的文本转化为可编辑和可搜索的电子文本。文本识别结果的准确性，对于后续的信息提取和数据挖掘等应用至关重要。

## 二、支持模型列表

> 推理耗时仅包含模型推理耗时，不包含前后处理耗时。

###  PP-OCRv5_server_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 81 MB  
**模型介绍：**  
PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 86.38 | 8.46 | 31.21 |
| **高性能模式** | - | 2.36 | 31.21 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/31909 ),[Hugging Face](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-OCRv5_server_rec ) |

---

###  PP-OCRv5_mobile_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 16 MB  
**模型介绍：**  
PP-OCRv5_rec 是新一代文本识别模型。该模型致力于以单一模型高效、精准地支持简体中文、繁体中文、英文、日文四种主要语言，以及手写、竖版、拼音、生僻字等复杂文本场景的识别。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 81.29 | 5.43 | 21.20 |
| **高性能模式** | - | 1.46 | 5.32 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/31872 ),[Hugging Face](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-OCRv5_mobile_rec ) |
---

###  PP-OCRv4_server_rec_doc
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 182 MB  
**模型介绍：**  
PP-OCRv4_server_rec_doc是在PP-OCRv4_server_rec的基础上，在更多中文文档数据和PP-OCR训练数据的混合数据训练而成，增加了部分繁体字、日文、特殊字符的识别能力，可支持识别的字符为1.5万+。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 86.58 | 8.69 | 37.93 |
| **高性能模式** | - | 2.78 | 37.93 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_doc_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/31859 ),[Hugging Face](https://huggingface.co/PaddlePaddle/PP-OCRv4_server_rec_doc ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-OCRv4_server_rec_doc ) |

---

###  PP-OCRv4_mobile_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 10.5 MB  
**模型介绍：**  
PP-OCRv4的轻量级识别模型，推理效率高，可以部署在包含端侧设备的多种硬件设备中。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 78.74 | 5.26 | 17.48 |
| **高性能模式** | - | 1.12 | 3.61 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/31914 ),[Hugging Face](https://huggingface.co/PaddlePaddle/PP-OCRv4_mobile_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-OCRv4_mobile_rec ) |

---

###  PP-OCRv4_server_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 173 MB  
**模型介绍：**  
PP-OCRv4的服务器端模型，推理精度高，可以部署在多种不同的服务器上。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 85.19 | 8.75 | 36.93 |
| **高性能模式** | - | 2.49 | 36.93 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/31931 ),[Hugging Face](https://huggingface.co/PaddlePaddle/PP-OCRv4_server_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/PP-OCRv4_server_rec ) |

---

###  en_PP-OCRv4_mobile_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 7.5 MB  
**模型介绍：**  
基于PP-OCRv4识别模型训练得到的超轻量英文识别模型，支持英文、数字识别。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 70.39 | 4.81 | 17.20 |
| **高性能模式** | - | 1.23 | 4.18 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/31895 ),[Hugging Face](https://huggingface.co/PaddlePaddle/en_PP-OCRv4_mobile_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/en_PP-OCRv4_mobile_rec ) |
---

> ❗ 以上列出的是文本识别模块重点支持的<b>4个核心模型</b>，该模块总共支持<b>20个全量模型</b>，包含多个多语言文本识别模型，完整的模型列表如下：

<details>
<summary> 👉模型列表详情</summary>

###  ch_SVTRv2_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 80.5 MB  
**模型介绍：**  
SVTRv2 是一种由复旦大学视觉与学习实验室（FVL）的OpenOCR团队研发的服务端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，A榜端到端识别精度相比PP-OCRv4提升6%。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 68.81 | 10.38 | 66.52 |
| **高性能模式** | - | 8.31 | 30.83 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar)|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams),[AI Studio](https://aistudio.baidu.com/modelsdetail/31887 ),[Hugging Face](https://huggingface.co/PaddlePaddle/ch_svTRv2_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/ch_svTRv2_rec ) |

---

###  ch_RepSVTR_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 48.8 MB  
**模型介绍：**  
RepSVTR 文本识别模型是一种基于SVTRv2 的移动端文本识别模型，其在PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务中荣获一等奖，B榜端到端识别精度相比PP-OCRv4提升2.5%，推理速度持平。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 65.07 | 6.29 | 20.64 |
| **高性能模式** | - | 1.57 | 5.40 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/31899 ),[Hugging Face](https://huggingface.co/PaddlePaddle/ch_RepSVTR_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/ch_RepSVTR_rec ) |

---

###  en_PP-OCRv5_mobile_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 7.5 MB  
**模型介绍：**  
基于PP-OCRv5识别模型训练得到的超轻量级英文识别模型，进一步提升英文文本的识别准确率，优化空格漏识别的问题，并提高对手写英文文本的识别效果。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) |
| :--- | :--- |
| **常规模式** | 85.25 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv5_mobile_rec_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/33810 ),[Hugging Face](https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/en_PP-OCRv5_mobile_rec ) |

---

### korean_PP-OCRv5_mobile_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 14 MB  
**模型介绍：**  
基于PP-OCRv5识别模型训练得到的超轻量韩文识别模型，支持韩文、英文和数字识别。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 88.0 | 5.43 | 21.20 |
| **高性能模式** | - | 1.46 | 5.32 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv5_mobile_rec_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv5_mobile_rec_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/31927 ),[Hugging Face](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/korean_PP-OCRv5_mobile_rec ) |

---

###  latin_PP-OCRv5_mobile_rec
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 14 MB  
**模型介绍：**  
基于PP-OCRv5识别模型训练得到的拉丁文识别模型，支持大部分拉丁字母语言、数字识别。

**性能指标：**
| 指标名称 | 识别 Avg Accuracy(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 84.7 | 5.43 | 21.20 |
| **高性能模式** | - | 1.46 | 5.32 |

**下载链接：**  
| 训练模型 |  推理模型 |
|:---: |:---: |
|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv5_mobile_rec_infer.tar )|[BOS源](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams ),[AI Studio](https://aistudio.baidu.com/modelsdetail/31861 ),[Hugging Face](https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec ),[ModelScope](https://www.modelscope.cn/models/PaddlePaddle/latin_PP-OCRv5_mobile_rec ) |

---

###  测试环境说明
**性能测试环境：**
- **硬件配置：**
  - GPU：NVIDIA Tesla T4
  - CPU：Intel Xeon Gold 6271C @ 2.60GHz
- **软件环境：**
  - Ubuntu 20.04 / CUDA 11.8 / cuDNN 8.9 / TensorRT 8.6.1.6
  - paddlepaddle 3.0.0 / paddleocr 3.0.3

**推理模式说明：**
| 模式 | GPU配置 | CPU配置 | 加速技术组合 |
| :--- | :--- | :--- | :--- |
| **常规模式** | FP32精度 / 无TRT加速 | FP32精度 / 8线程 | PaddleInference |
| **高性能模式** | 选择先验精度类型和加速策略的最优组合 | FP32精度 / 8线程 | 选择先验最优后端（Paddle/OpenVINO/TRT等） |

</details>


## 三、快速开始

> ❗ 在快速开始前，请先安装 PaddleOCR 的 wheel 包，详细请参考 [安装教程](../installation.md)。

使用一行命令即可快速体验：

```bash
paddleocr text_recognition -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png
```

<b>注：</b>PaddleOCR 官方模型默认从 HuggingFace 获取，如运行环境访问 HuggingFace 不便，可通过环境变量修改模型源为 BOS：`PADDLE_PDX_MODEL_SOURCE="BOS"`，未来将支持更多主流模型源；

您也可以将文本识别的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png)到本地。

```python
from paddleocr import TextRecognition
model = TextRecognition(model_name="PP-OCRv5_server_rec")
output = model.predict(input="general_ocr_rec_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

运行后，得到的结果为：
```bash
{'res': {'input_path': 'general_ocr_rec_001.png', 'page_index': None, 'rec_text': '绿洲仕格维花园公寓', 'rec_score': 0.9823867082595825}}
```

运行结果参数含义如下：
- `input_path`：表示输入待预测文本行图像的路径
- `page_index`：如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`
- `rec_text`：表示文本行图像的预测文本
- `rec_score`：表示文本行图像的预测置信度


可视化图片如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/modules/text_recog/general_ocr_rec_001.png"/>

相关方法、参数等说明如下：

* `TextRecognition`实例化文本识别模型（此处以`PP-OCRv5_server_rec`为例），具体说明如下：
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
<td><code>model_name</code></td>
<td>模型名称。如果设置为<code>None</code>，则使用<code>PP-OCRv5_server_rec</code>。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>模型存储路径。</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>用于推理的设备。<br/>
<b>例如：</b><code>"cpu"</code>、<code>"gpu"</code>、<code>"npu"</code>、<code>"gpu:0"</code>、<code>"gpu:0,1"</code>。<br/>
如指定多个设备，将进行并行推理。<br/>
默认情况下，优先使用 GPU 0；若不可用则使用 CPU。
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>是否启用高性能推理。</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>是否启用 Paddle Inference 的 TensorRT 子图引擎。如果模型不支持通过 TensorRT 加速，即使设置了此标志，也不会使用加速。<br/>
对于 CUDA 11.8 版本的飞桨，兼容的 TensorRT 版本为 8.x（x>=6），建议安装 TensorRT 8.6.1.6。<br/>

</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>当使用 Paddle Inference 的 TensorRT 子图引擎时设置的计算精度。<br/><b>可选项：</b><code>"fp32"</code>、<code>"fp16"</code>。</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>
是否启用 MKL-DNN 加速推理。如果 MKL-DNN 不可用或模型不支持通过 MKL-DNN 加速，即使设置了此标志，也不会使用加速。<br/>
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
<td>在 CPU 上推理时使用的线程数量。</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>input_shape</code></td>
<td>模型输入图像尺寸，格式为 <code>(C, H, W)</code>。</td>
<td><code>tuple|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

* 调用文本识别模型的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。另外，本模块还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。`predict()` 方法参数有 `input` 和 `batch_size`，具体说明如下：

<table>
<thead>
<tr>
<th>参数</th>
<th>参数说明</th>
<th>参数类型</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td><code>input</code></td>
<td>待预测数据，支持多种输入类型，必填。
<ul>
<li><b>Python Var</b>：如 <code>numpy.ndarray</code> 表示的图像数据</li>
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
<li><b>list</b>：列表元素需为上述类型数据，如<code>[numpy.ndarray, numpy.ndarray]</code>，<code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>，<code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>批大小，可设置为任意正整数。</td>
<td><code>int</code></td>
<td>1</td>
</tr>
</table>

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为图片、保存为`json`文件的操作:

<table>
<thead>
<tr>
<th>方法</th>
<th>方法说明</th>
<th>参数</th>
<th>参数类型</th>
<th>参数说明</th>
<th>默认值</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">打印结果到终端</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>是否对输出内容进行使用 <code>JSON</code> 缩进格式化</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">将结果保存为json格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>指定缩进级别，以美化输出的 <code>JSON</code> 数据，使其更具可读性，仅当 <code>format_json</code> 为 <code>True</code> 时有效</td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>控制是否将非 <code>ASCII</code> 字符转义为 <code>Unicode</code>。设置为 <code>True</code> 时，所有非 <code>ASCII</code> 字符将被转义；<code>False</code> 则保留原始字符，仅当<code>format_json</code>为<code>True</code>时有效</td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>将结果保存为图像格式的文件</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致</td>
<td>无</td>
</tr>
</table>

* 此外，也支持通过属性获取带结果的可视化图像和预测结果，具体如下：

<table>
<thead>
<tr>
<th>属性</th>
<th>属性说明</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">获取预测的<code>json</code>格式的结果</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">获取格式为<code>dict</code>的可视化图像</td>
</tr>
</table>


## 四、二次开发

如果以上模型在您的场景上效果仍然不理想，您可以尝试以下步骤进行二次开发，此处以训练 `PP-OCRv5_server_rec` 举例，其他模型替换对应配置文件即可。首先，您需要准备文本识别的数据集，可以参考[文本识别 Demo 数据](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar)的格式准备，准备好后，即可按照以下步骤进行模型训练和导出，导出后，可以将模型快速集成到上述 API 中。此处以文本识别 Demo 数据示例。在训练模型之前，请确保已经按照[安装文档](../installation.md)安装了 PaddleOCR 所需要的依赖。


## 4.1 数据集、预训练模型准备

### 4.1.1 准备数据集

```shell
# 下载示例数据集
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/ocr_rec_dataset_examples.tar
tar -xf ocr_rec_dataset_examples.tar
```

### 4.1.2 下载预训练模型

```shell
# 下载 PP-OCRv5_server_rec 预训练模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams 
```

### 4.2 模型训练

PaddleOCR 对代码进行了模块化，训练 `PP-OCRv5_server_rec` 识别模型时需要使用 `PP-OCRv5_server_rec` 的[配置文件](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml)。


训练命令如下：

```bash
#单卡训练 (默认训练方式)
python3 tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
   -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3'  tools/train.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
        -o Global.pretrained_model=./PP-OCRv5_server_rec_pretrained.pdparams
```


### 4.3 模型评估

您可以评估已经训练好的权重，如，`output/xxx/xxx.pdparams`，使用如下命令进行评估：

```bash
#注意将pretrained_model的路径设置为本地路径。若使用自行训练保存的模型，请注意修改路径和文件名为{path/to/weights}/{model_name}。
#demo 测试集评估
python3 tools/eval.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o \
Global.pretrained_model=output/xxx/xxx.pdparams
```

### 4.4 模型导出

```bash
python3 tools/export_model.py -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml -o \
Global.pretrained_model=output/xxx/xxx.pdparams \
Global.save_inference_dir="./PP-OCRv5_server_rec_infer/"
```

 导出模型后，静态图模型会存放于当前目录的`./PP-OCRv5_server_rec_infer/`中，在该目录下，您将看到如下文件：
 ```
 ./PP-OCRv5_server_rec_infer/
 ├── inference.json
 ├── inference.pdiparams
 ├── inference.yml
 ```
至此，二次开发完成，该静态图模型可以直接集成到 PaddleOCR 的 API 中。

## 五、FAQ
