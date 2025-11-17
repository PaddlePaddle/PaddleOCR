---
comments: true
---

# 表格结构识别模块使用教程

## 一、概述

表格结构识别是表格识别系统中的重要组成部分，能够将不可编辑表格图片转换为可编辑的表格形式（例如html）。表格结构识别的目标是对表格的行、列和单元格位置进行识别，该模块的性能直接影响到整个表格识别系统的准确性和效率。表格结构识别模块会输出表格区域的html代码，这些代码将作为输入传递给表格识别产线进行后续处理。

## 二、支持模型列表

> 推理耗时仅包含模型推理耗时，不包含前后处理耗时。

### 📊📊 SLANet
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 6.9 MB  
**模型介绍：**  
SLANet 是百度飞桨视觉团队自研的表格结构识别模型。该模型通过采用 CPU 友好型轻量级骨干网络 PP-LCNet、高低层特征融合模块 CSP-PAN、结构与位置信息对齐的特征解码模块 SLA Head，大幅提升了表格结构识别的精度和推理速度。

**性能指标：**
| 指标名称 | 精度(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 59.52 | 23.96 | - |
| **高性能模式** | - | 21.75 | 43.12 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams)

---

### 📊📊 SLANet_plus
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 6.9 MB  
**模型介绍：**  
SLANet_plus 是百度飞桨视觉团队自研的表格结构识别模型 SLANet 的增强版。相较于 SLANet，SLANet_plus 对无线表、复杂表格的识别能力得到了大幅提升，并降低了模型对表格定位准确性的敏感度，即使表格定位出现偏移，也能够较准确地进行识别。

**性能指标：**
| 指标名称 | 精度(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 63.69 | 23.43 | - |
| **高性能模式** | - | 22.16 | 41.80 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams)

---

### 📊📊 SLANeXt
**模型类型：** 推理模型/训练模型 | **模型存储大小：** 351 MB  
**模型介绍：**  
SLANeXt 系列是百度飞桨视觉团队自研的新一代表格结构识别模型。相较于 SLANet 和 SLANet_plus，SLANeXt 专注于对表格结构进行识别，并且对有线表格(wired)和无线表格(wireless)的识别分别训练了专用的权重，对各类型表格的识别能力都得到了明显提高，特别是对有线表格的识别能力得到了大幅提升。

#### SLANeXt_wired
**性能指标：**
| 指标名称 | 精度(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 69.65 | 85.92 | - |
| **高性能模式** | - | 85.92 | 501.66 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams)

#### SLANeXt_wireless
**性能指标：**
| 指标名称 | 精度(%) | GPU推理耗时 (ms) | CPU推理耗时 (ms) |
| :--- | :--- | :--- | :--- |
| **常规模式** | 69.65 | 85.92 | - |
| **高性能模式** | - | 85.92 | 501.66 |

**下载链接：**  
[推理模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wireless_infer.tar) | [训练模型](https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams)

---

### 🧪🧪🧪 测试环境说明
**性能测试环境：**
- **测试数据集：** 内部自建的高难度中文表格识别数据集
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



## 三、快速开始

> ❗ 在快速开始前，请先安装 PaddleOCR 的 wheel 包，详细请参考 [安装教程](../installation.md)。

使用一行命令即可快速体验：

```bash
paddleocr table_structure_recognition -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg
```

<b>注：</b>PaddleOCR 官方模型默认从 HuggingFace 获取，如运行环境访问 HuggingFace 不便，可通过环境变量修改模型源为 BOS：`PADDLE_PDX_MODEL_SOURCE="BOS"`，未来将支持更多主流模型源；

您也可以将表格结构识别的模块中的模型推理集成到您的项目中。运行以下代码前，请您下载[示例图片](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg)到本地。

```python
from paddleocr import TableStructureRecognition
model = TableStructureRecognition(model_name="SLANet")
output = model.predict(input="table_recognition.jpg", batch_size=1)
for res in output:
    res.print(json_format=False)
    res.save_to_json("./output/res.json")
```

运行后，得到的结果为：

```
{'res': {'input_path': 'table_recognition.jpg', 'page_index': None, 'bbox': [[42, 2, 390, 2, 388, 27, 40, 26], [11, 35, 89, 35, 87, 63, 11, 63], [113, 34, 192, 34, 186, 64, 109, 64], [219, 33, 399, 33, 393, 62, 212, 62], [413, 33, 544, 33, 544, 64, 407, 64], [12, 67, 98, 68, 96, 93, 12, 93], [115, 66, 205, 66, 200, 91, 111, 91], [234, 65, 390, 65, 385, 92, 227, 92], [414, 66, 537, 67, 537, 95, 409, 95], [7, 97, 106, 97, 104, 128, 7, 128], [113, 96, 206, 95, 201, 127, 109, 127], [236, 96, 386, 96, 381, 128, 230, 128], [413, 96, 534, 95, 533, 127, 408, 127]], 'structure': ['<html>', '<body>', '<table>', '<tr>', '<td', ' colspan="4"', '>', '</td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '<tr>', '<td></td>', '<td></td>', '<td></td>', '<td></td>', '</tr>', '</table>', '</body>', '</html>'], 'structure_score': 0.99948007}}
```

参数含义如下：

- `input_path`：输入的待预测表格图像的路径
- `page_index`：如果输入是PDF文件，则表示当前是PDF的第几页，否则为 `None`
- `boxes`：预测的表格单元格信息，一个列表，由预测的若干表格单元格坐标组成。特别地， SLANeXt 系列模型预测的表格单元格无效
- `structure`：预测的表格结构Html表达式，一个列表，由预测的若干Html关键字按顺序组成
- `structure_score`：预测表格结构的置信度

相关方法、参数等说明如下：

* `TableStructureRecognition`实例化表格结构识别模型（此处以`SLANet`为例），具体说明如下：
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
<td>模型名称。如果设置为<code>None</code>，则使用<code>PP-LCNet_x1_0_table_cls</code>。</td>
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
</tbody>
</table>

* 调用表格结构识别模型的 `predict()` 方法进行推理预测，该方法会返回一个结果列表。另外，本模块还提供了 `predict_iter()` 方法。两者在参数接受和结果返回方面是完全一致的，区别在于 `predict_iter()` 返回的是一个 `generator`，能够逐步处理和获取预测结果，适合处理大型数据集或希望节省内存的场景。可以根据实际需求选择使用这两种方法中的任意一种。`predict()` 方法参数有 `input` 和 `batch_size`，具体说明如下：

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
<li><b>str</b>：如图像文件或者PDF文件的本地路径：<code>/root/data/img.jpg</code>；<b>如URL链接</b>，如图像文件或PDF文件的网络URL：<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg">示例</a>；<b>如本地目录</b>，该目录下需包含待预测图像，如本地路径：<code>/root/data/</code>(当前不支持目录中包含PDF文件的预测，PDF文件需要指定到具体文件路径)</li>
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

* 对预测结果进行处理，每个样本的预测结果均为对应的Result对象，且支持打印、保存为`json`文件的操作:

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
</table>

* 此外，也支持通过属性获取结果，具体如下：

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
</table>

## 四、二次开发

如果以上模型在您的场景上效果仍然不理想，您可以尝试以下步骤进行二次开发，此处以训练 `SLANet_plus` 举例，其他模型替换对应配置文件即可。首先，您需要准备表格结构识别的数据集，可以参考[表格结构识别 Demo 数据](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_rec_dataset_examples.tar)的格式准备，准备好后，即可按照以下步骤进行模型训练和导出，导出后，可以将模型快速集成到上述 API 中。此处以表格结构识别 Demo 数据示例。在训练模型之前，请确保已经按照[安装文档](../installation.md)安装了 PaddleOCR 所需要的依赖。


### 4.1 数据集、预训练模型准备

#### 4.1.1 准备数据集

```shell
# 下载示例数据集
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_rec_dataset_examples.tar
tar -xf table_rec_dataset_examples.tar
```

#### 4.1.2 下载预训练模型

```shell
# 下载 SLANet_plus 预训练模型
wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams
```

### 4.2 模型训练

PaddleOCR 对代码进行了模块化，训练 `SLANet_plus` 识别模型时需要使用 `SLANet_plus` 的[配置文件](https://github.com/PaddlePaddle/PaddleOCR/blob/main/configs/table/SLANet_plus.yml)。


训练命令如下：

```bash
#单卡训练 (默认训练方式)
python3 tools/train.py -c configs/table/SLANet_plus.yml \
    -o Global.pretrained_model=./SLANet_plus_pretrained.pdparams
    Train.dataset.data_dir=./table_rec_dataset_examples \
    Train.dataset.label_file_list='[./table_rec_dataset_examples/train.txt]' \
    Eval.dataset.data_dir=./table_rec_dataset_examples \
    Eval.dataset.label_file_list='[./table_rec_dataset_examples/val.txt]'

#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py \
    -c configs/table/SLANet_plus.yml \
    -o Global.pretrained_model=./SLANet_plus_pretrained.pdparams
    -o Global.pretrained_model=./PP-OCRv5_server_det_pretrained.pdparams \
    Train.dataset.data_dir=./table_rec_dataset_examples \
    Train.dataset.label_file_list='[./table_rec_dataset_examples/train.txt]' \
    Eval.dataset.data_dir=./table_rec_dataset_examples \
    Eval.dataset.label_file_list='[./table_rec_dataset_examples/val.txt]'
```


### 4.3 模型评估

您可以评估已经训练好的权重，如，`output/xxx/xxx.pdparams`，使用如下命令进行评估：

```bash
# 注意将pretrained_model的路径设置为本地路径。若使用自行训练保存的模型，请注意修改路径和文件名为{path/to/weights}/{model_name}。
 # demo 测试集评估
 python3 tools/eval.py -c configs/table/SLANet_plus.yml -o \
    Global.pretrained_model=output/xxx/xxx.pdparams
    Eval.dataset.data_dir=./table_rec_dataset_examples \
    Eval.dataset.label_file_list='[./table_rec_dataset_examples/val.txt]'
```

### 4.4 模型导出

```bash
 python3 tools/export_model.py -c configs/table/SLANet_plus.yml -o \
    Global.pretrained_model=output/xxx/xxx.pdparams \
    Global.save_inference_dir="./SLANet_plus_infer/"
```

 导出模型后，静态图模型会存放于当前目录的`./SLANet_plus_infer/`中，在该目录下，您将看到如下文件：
 ```
 ./SLANet_plus_infer/
 ├── inference.json
 ├── inference.pdiparams
 ├── inference.yml
 ```
至此，二次开发完成，该静态图模型可以直接集成到 PaddleOCR 的 API 中。

## 五、FAQ
