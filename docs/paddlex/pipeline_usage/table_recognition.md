# 通用表格识别产线使用教程

## 1. 通用表格识别产线介绍
表格识别是一种自动从文档或图像中识别和提取表格内容及其结构的技术，广泛应用于数据录入、信息检索和文档分析等领域。通过使用计算机视觉和机器学习算法，表格识别能够将复杂的表格信息转换为可编辑的格式，方便用户进一步处理和分析数据。

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/01.png)

**通用****表格识别****产线中包含表格结构识别模块、版面区域分析模块、文本检测模块和文本识别模块**。

**如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。

<details>
   <summary> 👉模型列表详情</summary>

**表格识别模块模型：**

<table>
  <tr>
    <th>模型</th>
    <th>精度（%）</th>
    <th>GPU推理耗时 (ms)</th>
    <th>CPU推理耗时</th>
    <th>模型存储大小 (M)</th>
    <th>介绍</th>
  </tr>
  <tr>
    <td>SLANet</td>
    <td>59.52</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
    <td rowspan="2">SLANet 是百度飞桨视觉团队自研的表格结构识别模型。该模型通过采用CPU 友好型轻量级骨干网络PP-LCNet、高低层特征融合模块CSP-PAN、结构与位置信息对齐的特征解码模块SLA Head，大幅提升了表格结构识别的精度和推理速度。</td>
  </tr>
   <tr>
    <td>SLANet_plus</td>
    <td>63.69</td>
    <td>522.536</td>
    <td>1845.37</td>
    <td>6.9 M</td>
  </tr>
</table>

**注：以上精度指标测量PaddleX 内部自建英文表格识别数据集。所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**


**版面区域分析模块模型：**

|模型名称|mAP（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|PicoDet_layout_1x|86.8|13.036|91.2634|7.4M |
|PicoDet-L_layout_3cls|89.3|15.7425|159.771|22.6 M|
|RT-DETR-H_layout_3cls|95.9|114.644|3832.62|470.1M|
|RT-DETR-H_layout_17cls|92.6|115.126|3827.25|470.2M|

**注：以上精度指标的评估集是 PaddleX 自建的版面区域分析数据集，包含 1w 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

**文本检测模块模型：**

|模型名称|检测Hmean（%）|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|PP-OCRv4_mobile_det |77.79|10.6923|120.177|4.2 M|
|PP-OCRv4_server_det |82.69|83.3501|2434.01|100.1M|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中检测包含 500 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

**文本识别模块模型：**

|模型名称|识别Avg Accuracy(%)|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|
|-|-|-|-|-|
|PP-OCRv4_mobile_rec |78.20|7.95018|46.7868|10.6 M|
|PP-OCRv4_server_rec |79.20|7.19439|140.179|71.2 M|

**注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集 ，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 1.1w 张图片。以上所有模型 GPU 推理耗时基于 NVIDIA Tesla T4 机器，精度类型为 FP32， CPU 推理速度基于 Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz，线程数为8，精度类型为 FP32。**

</details>

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验通用图像分类产线的效果，也可以在本地使用命令行或 Python 体验通用图像分类产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/91661/webUI)通用表格识别产线的效果，用官方提供的 demo 图片进行识别，例如：

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/02.png)

如果您对产线运行的效果满意，可以直接对产线进行集成部署，如果不满意，您也可以利用私有数据**对产线中的模型进行在线微调**。

### 2.2 本地体验
在本地使用通用表格识别产线前，请确保您已经按照[PaddleX本地安装教程](../installation/installation.md)完成了PaddleX的wheel包安装。

### 2.1 命令行方式体验
一行命令即可快速体验表格识别产线效果

```
paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为表格识别产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```

在执行上述 Python 脚本时，加载的是默认的表格识别产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

<details>
   <summary> 👉点击展开</summary>

```
paddlex --get_pipeline_config table_recognition
```
执行后，表格识别产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为 `./my_path` ）：

```
paddlex --get_pipeline_config table_recognition --config_save_path ./my_path
```

获取产线配置文件后，可将 `--pipeline` 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 `./table_recognition.yaml`，只需执行：

```
paddlex --pipeline ./table_recognition.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg
```
其中，`--model`、`--device` 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

</details>

运行后，得到的结果为：

<details>
   <summary> 👉点击展开</summary>

```
{'input_path': '/root/.paddlex/predict_input/table_recognition.jpg', 'layout_result': {'input_path': '/root/.paddlex/predict_input/table_recognition.jpg', 'boxes': [{'cls_id': 3, 'label': 'Table', 'score': 0.6014542579650879, 'coordinate': [0, 21, 551, 118]}]}, 'ocr_result': {'dt_polys': [array([[37., 40.],
       [75., 40.],
       [75., 60.],
       [37., 60.]], dtype=float32), array([[123.,  37.],
       [200.,  37.],
       [200.,  59.],
       [123.,  59.]], dtype=float32), array([[227.,  37.],
       [391.,  37.],
       [391.,  58.],
       [227.,  58.]], dtype=float32), array([[416.,  36.],
       [535.,  38.],
       [535.,  61.],
       [415.,  58.]], dtype=float32), array([[35., 73.],
       [78., 73.],
       [78., 92.],
       [35., 92.]], dtype=float32), array([[287.,  73.],
       [328.,  73.],
       [328.,  92.],
       [287.,  92.]], dtype=float32), array([[453.,  72.],
       [495.,  72.],
       [495.,  94.],
       [453.,  94.]], dtype=float32), array([[ 17., 103.],
       [ 94., 103.],
       [ 94., 118.],
       [ 17., 118.]], dtype=float32), array([[145., 104.],
       [178., 104.],
       [178., 118.],
       [145., 118.]], dtype=float32), array([[277., 104.],
       [337., 102.],
       [338., 118.],
       [278., 118.]], dtype=float32), array([[446., 102.],
       [504., 104.],
       [503., 118.],
       [445., 118.]], dtype=float32)], 'rec_text': ['Dres', '连续工作3', '取出来放在网上，没想', '江、整江等八大', 'Abstr', 'rSrivi', '$709.', 'cludingGiv', '2.72', 'Ingcubic', '$744.78'], 'rec_score': [0.9934158325195312, 0.9990204572677612, 0.9967061877250671, 0.9375461935997009, 0.9947397112846375, 0.9972746968269348, 0.9904290437698364, 0.973427414894104, 0.9983080625534058, 0.993423342704773, 0.9964120984077454], 'input_path': '/root/.paddlex/predict_input/table_recognition.jpg'}, 'table_result': [{'input_path': '/root/.paddlex/predict_input/table_recognition.jpg', 'layout_bbox': [0, 21, 551, 118], 'bbox': array([[  4.395736 ,  25.238262 , 113.31014  ,  25.316246 , 115.454315 ,
         71.8867   ,   3.7177477,  71.7937   ],
       [110.727455 ,  25.94007  , 210.07187  ,  26.028755 , 209.66394  ,
         65.96484  , 109.59861  ,  66.09809  ],
       [214.45381  ,  26.027939 , 407.95276  ,  26.112846 , 409.6684   ,
         66.91336  , 215.27292  ,  67.002014 ],
       [402.81863  ,  26.123789 , 549.03656  ,  26.231564 , 549.19995  ,
         66.88339  , 404.48068  ,  66.74034  ],
       [  2.4458022,  64.68588  , 102.7665   ,  65.10228  , 105.79447  ,
         96.051254 ,   2.5367072,  95.35514  ],
       [108.85877  ,  65.80549  , 211.70216  ,  66.02091  , 210.79245  ,
         94.75581  , 107.59308  ,  94.42664  ],
       [217.05621  ,  64.98496  , 407.76328  ,  65.133484 , 406.8436   ,
         96.00133  , 214.67896  ,  95.87226  ],
       [401.73572  ,  64.60494  , 547.9967   ,  64.73921  , 548.19135  ,
         96.09901  , 402.26733  ,  95.95529  ],
       [  2.4882016,  93.589554 , 107.01325  ,  93.67592  , 107.8446   ,
        120.13259  ,   2.508764 , 119.85027  ],
       [110.773125 ,  93.98633  , 213.354    ,  94.08046  , 212.46033  ,
        120.80207  , 109.29008  , 120.613045 ],
       [216.08781  ,  94.19984  , 405.843    ,  94.28341  , 405.9974   ,
        121.33152  , 215.10301  , 121.299034 ],
       [403.92212  ,  94.44883  , 548.30963  ,  94.54982  , 548.4949   ,
        122.610176 , 404.53433  , 122.49881  ]], dtype=float32), 'img_idx': 0, 'html': '<html><body><table><tr><td>Dres</td><td>连续工作3</td><td>取出来放在网上，没想</td><td>江、整江等八大</td></tr><tr><td>Abstr</td><td></td><td>rSrivi</td><td>$709.</td></tr><tr><td>cludingGiv</td><td>2.72</td><td>Ingcubic</td><td>$744.78</td></tr></table></body></html>'}]}
```
</details>

![](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/table_recognition/03.png)

可视化图片默认保存在 `output` 目录下，您也可以通过 `--save_path` 进行自定义。

### 2.2 Python脚本方式集成
几行代码即可完成产线的快速推理，以通用表格识别产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="table_recognition")

output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv("./output/") ## 保存csv格式结果
    res.save_to_xlsx("./output/") ## 保存表格格式结果
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

（1）实例化 `create_pipeline` 实例化产线对象：具体参数说明如下：

|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|`pipeline`|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|`str`|无|
|`device`|产线模型推理设备。支持：“gpu”，“cpu”。|`str`|`gpu`|
|`enable_hpi`|是否启用高性能推理，仅当该产线支持高性能推理时可用。|`bool`|`False`|

（2）调用产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

| 参数类型      | 参数说明                                                                                                  |
|---------------|-----------------------------------------------------------------------------------------------------------|
| Python Var    | 支持直接传入Python变量，如numpy.ndarray表示的图像数据。                                               |
| str         | 支持传入待预测数据文件路径，如图像文件的本地路径：`/root/data/img.jpg`。                                   |
| str           | 支持传入待预测数据文件URL，如图像文件的网络URL：[示例](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg)。|
| str           | 支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：`/root/data/`。                               |
| dict          | 支持传入字典类型，字典的key需与具体任务对应，如图像分类任务对应\"img\"，字典的val支持上述类型数据，例如：`{\"img\": \"/root/data1\"}`。|
| list          | 支持传入列表，列表元素需为上述类型数据，如`[numpy.ndarray, numpy.ndarray]，[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]`，`[\"/root/data1\", \"/root/data2\"]`，`[{\"img\": \"/root/data1\"}, {\"img\": \"/root/data2/img.jpg\"}]`。|

（3）调用`predict`方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果。

（4）对预测结果进行处理：每个样本的预测结果均为`dict`类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：


|方法|说明|方法参数|
|-|-|-|
|save_to_csv|将结果保存为csv格式的文件|`- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|
|save_to_html|将结果保存为html格式的文件|`- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|
|save_to_xlsx|将结果保存为表格格式的文件|`- save_path`：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|

若您获取了配置文件，即可对表格识别产线各项配置进行自定义，只需要修改 `create_pipeline` 方法中的 `pipeline` 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 `./my_path/table_recognition.yaml` ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/table_recognition.yaml")
output = pipeline.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_csv("./output/") ## 保存csv格式结果
    res.save_to_xlsx("./output/") ## 保存表格格式结果
```
## 3. 开发集成/部署
如果产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将产线直接应用在您的Python项目中，可以参考 [2.2.2 Python脚本方式](#22-python脚本方式集成)中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

🚀 **高性能部署**：在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考[PaddleX高性能部署指南](../pipeline_deploy/high_performance_deploy.md)。

☁️ **服务化部署**：服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX服务化部署指南](../pipeline_deploy/service_deploy.md)。

📱 **端侧部署**：端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](../pipeline_deploy/lite_deploy.md)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用表格识别产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用表格识别产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用表格识别产线包含四个模块，模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，参考如下规则进行分析和模型微调：

* 检测到的表格结构错误（如行列识别错误、单元格位置错误），那么可能是表格结构识别模块存在不足，您需要参考[表格结构识别模块开发教程](../module_usage/table_structure_recognition.md)中的二次开发章节，使用您的私有数据集对表格结构识别模型进行微调。
* 表格区域在整体版面中定位错误，那么可能是版面区域定位模块存在不足，您需要参考[版面区域检测模块开发教程](../module_usage/layout_detection.md)中的二次开发章节，使用您的私有数据集对版面区域定位模型进行微调。
* 有较多的文本未被检测出来（即文本漏检现象），那么可能是文本检测模型存在不足，您需要参考[文本检测模块开发教程](../module_usage/text_recognition.md)中的二次开发章节，使用您的私有数据集对文本检测模型进行微调。
* 已检测到的文本中出现较多的识别错误（即识别出的文本内容与实际文本内容不符），这表明文本识别模型需要进一步改进，您需要参考[文本识别模块开发教程](../module_usage/table_structure_recognition.md)中的二次开发章节对文本识别模型进行微调。

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```python
......
 Pipeline:
  layout_model: PicoDet_layout_1x  #可修改为微调后模型的本地路径
  table_model: SLANet  #可修改为微调后模型的本地路径
  text_det_model: PP-OCRv4_mobile_det  #可修改为微调后模型的本地路径
  text_rec_model: PP-OCRv4_mobile_rec  #可修改为微调后模型的本地路径
  layout_batch_size: 1
  text_rec_batch_size: 1
  table_batch_size: 1
  device: "gpu:0"
......
```
随后， 参考本地体验中的命令行方式或 Python 脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 `--device` 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行表格识别产线的推理，使用的 Python 命令为：

```
paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 `--device` 修改为npu 即可：

```
paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device npu:0
```
若您想在更多种类的硬件上使用通用表格识别产线，请参考[PaddleX多硬件使用指南](../other_devices_support/installation_other_devices.md)。
