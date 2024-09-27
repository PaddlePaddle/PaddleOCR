# 通用OCR产线开使用教程

## 1. OCR产线介绍
OCR（光学字符识别，Optical Character Recognition）是一种将图像中的文字转换为可编辑文本的技术。它广泛应用于文档数字化、信息提取和数据处理等领域。OCR 可以识别印刷文本、手写文本，甚至某些类型的字体和符号。

通用 OCR 产线用于解决文字识别任务，提取图片中的文字信息以文本形式输出，PP-OCRv4 是一个端到端 OCR 串联系统，可实现 CPU 上毫秒级的文本内容精准预测，在通用场景上达到开源SOTA。基于该项目，产学研界多方开发者已快速落地多个 OCR 应用，使用场景覆盖通用、制造、金融、交通等各个领域。

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=24e487ee25374a109c9ff40210e8b7f2&docGuid=mFtiLCg9OMkp4x "")
**通用OCR产线中包含了文本检测模块和文本识别模块**，每个模块中包含了若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。**如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型**。

|产线模块|具体模型|精度|GPU推理耗时（ms）|CPU推理耗时|模型存储大小（M)|配置文件|
|-|-|-|-|-|-|-|
|文本检测|PP-OCRv4_mobile_det|77.79|2.719474|79.1097|15|PP-OCRv4_server_det.yaml|
||PP-OCRv4_server_det|82.69|22.20346|2662.158|198|PP-OCRv4_mobile_det.yaml|
|文本识别|PP-OCRv4_mobile_rec|78.20|2.719474|79.1097|15|PP-OCRv4_mobile_rec.yaml|
||PP-OCRv4_server_rec|79.20|22.20346|2662.158|198|PP-OCRv4_server_rec.yaml|

**注：文本检测模型精度指标为 Hmean(%)，文本识别模型精度指标为 Accuracy(%)。**

## 2. 快速开始
PaddleX 所提供的预训练的模型产线均可以快速体验效果，你可以在线体验通用OCR产线的效果，也可以在本地使用命令行或 Python 体验通用 OCR 产线的效果。

### 2.1 在线体验
您可以[在线体验](https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent)通用 OCR 产线的效果，用官方提供的 Demo 图片进行识别，例如：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=db148c34e68a47bbb895709d253db3dd&docGuid=mFtiLCg9OMkp4x "")
如果您对产线运行的效果满意，可以直接对产线进行集成部署，您可以直接从云端下载部署包，也可以使用2.2节本地集成的方式。如果不满意，您也可以利用私有数据**对产线中的模型进行在线微调**。

### 2.2 本地体验
在本地使用通用OCR产线前，请确保您已经按照[PaddleX安装教程](/docs_new/installation/installation.md)完成了PaddleX的wheel包安装。

#### 2.2.1 命令行方式体验
一行命令即可快速体验OCR产线效果

```ruby
paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0
```
参数说明：

```
--pipeline：产线名称，此处为OCR产线
--input：待处理的输入图片的本地路径或URL
--device 使用的GPU序号（例如gpu:0表示使用第0块GPU，gpu:1,2表示使用第1、2块GPU），也可选择使用CPU（--device cpu）
```
执行后，将提示选择 OCR 产线配置文件保存路径，默认保存至*当前目录*，也可 *自定义路径*。

此外，也可在执行命令时加入 -y 参数，则可跳过路径选择，直接将产线配置文件保存至当前目录。

获取产线配置文件后，可将 --pipeline 替换为配置文件保存路径，即可使配置文件生效。例如，若配置文件保存路径为 ./ocr.yaml，只需执行：

```ruby
paddlex --pipeline ./ocr.yaml --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png
```
其中，--model、--device 等参数无需指定，将使用配置文件中的参数。若依然指定了参数，将以指定的参数为准。

运行后，得到的结果为：

```ruby
The prediction result is:
['登机口于起飞前10分钟关闭']
The prediction result is:
['GATES CLOSE 1O MINUTESBEFORE DEPARTURE TIME']
The prediction result is:
['ETKT7813699238489/1']
......
```
可视化结果如下：

![](https://rte.weiyun.baidu.com/wiki/attach/image/api/imageDownloadAddress?attachId=70bc63a55911405db8f95e0f1a4adbc9&docGuid=LpZd0rg_QSm61O "")
#### 2.2.2 Python脚本方式集成 
几行代码即可完成产线的快速推理，以通用 OCR 产线为例：

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline="ocr")

output = pipeline.predict("pre_image.jpg")
for batch in output:
    for item in batch:
        res = item['result']
        res.print()
        res.save_to_img("./output/")
        res.save_to_json("./output/")
```
得到的结果与命令行方式相同。

在上述 Python 脚本中，执行了如下几个步骤：

* 实例化 `create_pipeline` 实例化 OCR 产线对象：具体参数说明如下：
  
|参数|参数说明|参数类型|默认值|
|-|-|-|-|
|pipeline|产线名称或是产线配置文件路径。如为产线名称，则必须为 PaddleX 所支持的产线。|str|无|
|device|产线模型推理设备。支持：“gpu”，“cpu”。|str|gpu|
|enable_hpi|是否启用高性能推理，仅当该产线支持高性能推理时可用。|bool|False|
* 调用OCR产线对象的 `predict` 方法进行推理预测：`predict` 方法参数为`x`，用于输入待预测数据，支持多种输入方式，具体示例如下：

| 参数类型   | 参数说明                                                                                                                                                                                                                     |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Python Var | 支持直接传入Python变量，如numpy.ndarray表示的图像数据；                                                                                                                                                                      |
| str        | 支持传入待预测数据文件路径，如图像文件的本地路径：/root/data/img.jpg；                                                                                                                                                       |
| str        | 支持传入待预测数据文件url，如图像文件的网络url：https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_001.png；                                                                                     |
| str        | 支持传入本地目录，该目录下需包含待预测数据文件，如本地路径：/root/data/；                                                                                                                                                    |
| dict       | 支持传入字典类型，字典的key需要与具体产线对应，如OCR产线为"img"，字典的val支持上述类型数据，如：{"img": "/root/data1"}；                                                                                                     |
| list       | 支持传入列表，列表元素需为上述类型数据，如[numpy.ndarray, numpy.ndarray, ]，["/root/data/img1.jpg", "/root/data/img2.jpg", ]，["/root/data1", "/root/data2", ]，[{"img": "/root/data1"}, {"img": "/root/data2/img.jpg"}, ]； |

* 调用predict方法获取预测结果：`predict` 方法为`generator`，因此需要通过调用获得预测结果，`predict`方法以batch为单位对数据进行预测，因此预测结果为list形式表示的一组预测结果
* 对预测结果进行处理：每个样本的预测结果均为dict类型，且支持打印，或保存为文件，支持保存的类型与具体产线相关，如：

|方法|说明|方法参数|
|-|-|-|
|print|打印结果到终端|format_json：bool类型，是否对输出内容进行使用json缩进格式化，默认为True； indent：int类型，json格式化设置，仅当format_json为True时有效，默认为4； ensure_ascii：bool类型，json格式化设置，仅当format_json为True时有效，默认为False；|
|save_to_json|将结果保存为json格式的文件|save_path：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致； indent：int类型，json格式化设置，默认为4； ensure_ascii：bool类型，json格式化设置，默认为False；|
|save_to_img|将结果保存为图像格式的文件|save_path：str类型，保存的文件路径，当为目录时，保存文件命名与输入文件类型命名一致；|

在执行上述Python脚本时，加载的是默认的OCR产线配置文件，若您需要自定义配置文件，可执行如下命令获取：

```ruby
paddlex --get_pipeline_yaml ocr
```

执行后，OCR产线配置文件将被保存在当前路径。若您希望自定义保存位置，可执行如下命令（假设自定义保存位置为* ./my_path*）：

```ruby
paddlex --get_pipeline_config ocr --config_save_path ./my_path
```
获取配置文件后，您即可对OCR产线各项配置进行自定义，只需要修改 create_pipeline 方法中的 pipeline 参数值为产线配置文件路径即可。

例如，若您的配置文件保存在 *./my_path/ocr.yaml* ，则只需执行：

```python
from paddlex import create_pipeline
pipeline = create_pipeline(pipeline="./my_path/ocr.yaml")
output = pipeline.predict("pre_image.jpg")
for res in output:
    res.print() # 打印预测的结构化输出
    res.save_to_img("./output/") # 保存结果可视化图像
    res.save_to_json("./output/") # 保存预测的结构化输出
```
## 3. 开发集成/部署
如果通用 OCR 产线可以达到您对产线推理速度和精度的要求，您可以直接进行开发集成/部署。

若您需要将通用 OCR 产线直接应用在您的Python项目中，可以参考 2.2.2 Python脚本方式中的示例代码。

此外，PaddleX 也提供了其他三种部署方式，详细说明如下：

* **高性能部署**:在实际生产环境中，许多应用对部署策略的性能指标（尤其是响应速度）有着较严苛的标准，以确保系统的高效运行与用户体验的流畅性。为此，PaddleX 提供高性能推理插件，旨在对模型推理及前后处理进行深度性能优化，实现端到端流程的显著提速，详细的高性能部署流程请参考[PaddleX 高性能部署指南](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/GvMbk70MZz/z0PYxETcClzAFu?source=137?t=mention&mt=doc&dt=doc)。高性能部署的benchmark指标需要给出来，这个指标需要通过FD来提供。
* **服务化部署**:服务化部署是实际生产环境中常见的一种部署形式。通过将推理功能封装为服务，客户端可以通过网络请求来访问这些服务，以获取推理结果。PaddleX 支持用户以低成本实现产线的服务化部署，详细的服务化部署流程请参考[PaddleX 服务化部署指南](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/GvMbk70MZz/CH8L_9JeqZA-nU?t=mention&mt=doc&dt=doc)。
* **端侧部署**:端侧部署是一种将计算和数据处理功能放在用户设备本身上的方式，设备可以直接处理数据，而不需要依赖远程的服务器。PaddleX 支持将模型部署在 Android 等端侧设备上，详细的端侧部署流程请参考[PaddleX端侧部署指南](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/pKzJfZczuc/GvMbk70MZz/WgkMGzkjzQlsxg?source=137?t=mention&mt=doc&dt=doc)。
您可以根据需要选择合适的方式部署模型产线，进而进行后续的 AI 应用集成。

## 4. 二次开发
如果通用 OCR 产线提供的默认模型权重在您的场景中，精度或速度不满意，您可以尝试利用**您自己拥有的特定领域或应用场景的数据**对现有模型进行进一步的**微调**，以提升通用 OCR 产线的在您的场景中的识别效果。

### 4.1 模型微调
由于通用OCR产线包含两个模块（文本检测和文本识别），模型产线的效果不及预期可能来自于其中任何一个模块。

您可以对识别效果差的图片进行分析，如果在分析过程中发现有较多的文本未被检测出来（即文本漏检现象），那么可能是文本检测模型存在不足，您需要参考[文本检测模块开发教程](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/y0mmii50BW/LEp_AuhpLJ_BhT?t=mention&mt=doc&dt=doc)中的**二次开发**章节（github可以直接链接标题），使用您的私有数据集对文本检测模型进行微调；如果在已检测到的文本中出现较多的识别错误（即识别出的文本内容与实际文本内容不符），这表明文本识别模型需要进一步改进，您需要参考[文本识别模块开发教程](https://ku.baidu-int.com/knowledge/HFVrC7hq1Q/yKeL8Lljko/y0mmii50BW/0NoIlw1Yv8fCj9?t=mention&mt=doc&dt=doc)中的**二次开发**章节（github可以直接链接标题）对文本识别模型进行微调。 

### 4.2 模型应用
当您使用私有数据集完成微调训练后，可获得本地模型权重文件。

若您需要使用微调后的模型权重，只需对产线配置文件做修改，将微调后模型权重的本地路径替换至产线配置文件中的对应位置即可：

```
......
Pipeline:
  det_model: PP-OCRv4_server_det  #可修改为微调后文本检测模型的本地路径
  det_device: "gpu"
  rec_model: PP-OCRv4_server_rec  #可修改为微调后文本识别模型的本地路径
  rec_batch_size: 1
  rec_device: "gpu"
......
```
随后， 参考 *2.2 本地体验* 中的命令行方式或Python脚本方式，加载修改后的产线配置文件即可。

##  5. 多硬件支持
PaddleX 支持英伟达 GPU、昆仑芯 XPU、昇腾 NPU和寒武纪 MLU 等多种主流硬件设备，**仅需修改 ****--device**** 参数**即可完成不同硬件之间的无缝切换。

例如，您使用英伟达 GPU 进行 OCR 产线的推理，使用的 Python 命令为：

```
paddlex --pipeline OCR --model PP-OCRv4_mobile_det PP-OCRv4_mobile_rec --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0
```
此时，若您想将硬件切换为昇腾 NPU，仅需对 Python 命令中的 --device 进行修改即可：

```
paddlex --pipeline OCR --model PP-OCRv4_mobile_det PP-OCRv4_mobile_rec --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device npu:0
```
若您想在更多种类的硬件上使用通用OCR产线，请参考[PaddleX多硬件使用指南](/docs_new/installation/installation_other_devices.md)。