# 快速开始

>**说明：**

>* 飞桨低代码开发工具[PaddleX](https://github.com/PaddlePaddle/PaddleX/tree/release/3.0-beta1)，依托于PaddleOCR的先进技术，支持了OCR领域的**低代码全流程**开发能力。通过低代码开发，可实现简单且高效的模型使用、组合与定制。

>* PaddleX 致力于实现产线级别的模型训练、推理与部署。模型产线是指一系列预定义好的、针对特定AI任务的开发流程，其中包含能够独立完成某类任务的单模型（单功能模块）组合。本文档提供**OCR相关产线**的快速推理使用，单功能模块的快速使用以及更多功能请参考[PaddleOCR低代码全流程开发](https://paddlepaddle.github.io/PaddleOCR/latest/paddlex/overview.html)中相关章节。


### 🛠️ 安装

> ❗安装PaddleX前请先确保您有基础的**Python运行环境**。
* **安装PaddlePaddle**
```bash
# cpu
python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗ 更多飞桨 Wheel 版本请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html)。

* **安装PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```

> ❗ 更多安装方式参考[PaddleX安装教程](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/installation/installation.md)

### 📝 Python 脚本使用

三行代码即可完成产线的快速推理，统一的 Python 脚本格式如下：
```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[产线名称])
output = pipeline.predict([输入图片名称])
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
执行了如下几个步骤：

* `create_pipeline()` 实例化产线对象
* 传入图片并调用产线对象的 `predict` 方法进行推理预测
* 对预测结果进行处理

其他产线的 Python 脚本使用，只需将 `create_pipeline()` 方法的 `pipeline` 参数调整为相应产线的名称。下面列出了每个产线对应的参数名称及详细的使用解释：


<b>👉 更多产线的Python脚本使用</b>

| 产线名称           | 对应参数               | 详细说明                                                                                                      |
|--------------------|------------------------|---------------------------------------------------------------------------------------------------------------|
| 文档场景信息抽取v3   | `PP-ChatOCRv3-doc` | [文档场景信息抽取v3产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction.md#22-本地体验) |
| 通用OCR            | `OCR` | [通用OCR产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/OCR.md#222-python脚本方式集成) |
| 通用表格识别       | `table_recognition` | [通用表格识别产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/table_recognition.md#22-python脚本方式集成) |
| 通用版面解析       | `layout_parsing`                | [通用版面解析产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.md#22-python脚本方式集成)                                   |
| 公式识别       | `formula_recognition`                | [公式识别产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.md#22-python脚本方式集成)                                   |
| 印章文本识别       | `seal_recognition`                | [印章文本识别产线Python脚本使用说明](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.md#22-python脚本方式集成)                                          |

### 💻 命令行使用

一行命令即可快速体验产线效果，统一的命令行格式为：

```bash
paddlex --pipeline [产线名称] --input [输入图片] --device [运行设备]
```

只需指定三个参数：
* `pipeline`：产线名称
* `input`：待处理的输入文件（如图片）的本地路径或 URL
* `device`: 使用的 GPU 序号（例如`gpu:0`表示使用第 0 块 GPU），也可选择使用 CPU（`cpu`）


以通用 OCR 产线为例：
```bash
paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0
```

运行结果如下：

```bash
{'img_path': '/root/.paddlex/predict_input/general_ocr_002.png', 'dt_polys': [[[5, 12], [88, 10], [88, 29], [5, 31]], [[208, 14], [249, 14], [249, 22], [208, 22]], [[695, 15], [824, 15], [824, 60], [695, 60]], [[158, 27], [355, 23], [356, 70], [159, 73]], [[421, 25], [659, 19], [660, 59], [422, 64]], [[337, 104], [460, 102], [460, 127], [337, 129]], [[486, 103], [650, 100], [650, 125], [486, 128]], [[675, 98], [835, 94], [835, 119], [675, 124]], [[64, 114], [192, 110], [192, 131], [64, 134]], [[210, 108], [318, 106], [318, 128], [210, 130]], [[82, 140], [214, 138], [214, 163], [82, 165]], [[226, 136], [328, 136], [328, 161], [226, 161]], [[404, 134], [432, 134], [432, 161], [404, 161]], [[509, 131], [570, 131], [570, 158], [509, 158]], [[730, 138], [771, 138], [771, 154], [730, 154]], [[806, 136], [817, 136], [817, 146], [806, 146]], [[342, 175], [470, 173], [470, 197], [342, 199]], [[486, 173], [616, 171], [616, 196], [486, 198]], [[677, 169], [813, 166], [813, 191], [677, 194]], [[65, 181], [170, 177], [171, 202], [66, 205]], [[96, 208], [171, 205], [172, 230], [97, 232]], [[336, 220], [476, 215], [476, 237], [336, 242]], [[507, 217], [554, 217], [554, 236], [507, 236]], [[87, 229], [204, 227], [204, 251], [87, 254]], [[344, 240], [483, 236], [483, 258], [344, 262]], [[66, 252], [174, 249], [174, 271], [66, 273]], [[75, 279], [264, 272], [265, 297], [76, 303]], [[459, 297], [581, 295], [581, 320], [459, 322]], [[101, 314], [210, 311], [210, 337], [101, 339]], [[68, 344], [165, 340], [166, 365], [69, 368]], [[345, 350], [662, 346], [662, 368], [345, 371]], [[100, 459], [832, 444], [832, 465], [100, 480]]], 'dt_scores': [0.8183103704439653, 0.7609575621092027, 0.8662357274035412, 0.8619508290334809, 0.8495855993183273, 0.8676840017933314, 0.8807986687956436, 0.822308525056085, 0.8686617037621976, 0.8279022169854463, 0.952332847006758, 0.8742692553015098, 0.8477013022907575, 0.8528771493227294, 0.7622965906848765, 0.8492388224448705, 0.8344203789965632, 0.8078477124353284, 0.6300434587457232, 0.8359967356998494, 0.7618617265751318, 0.9481573079350023, 0.8712182945408912, 0.837416955846334, 0.8292475059403851, 0.7860382856406026, 0.7350527486717117, 0.8701022267947695, 0.87172526903969, 0.8779847108088126, 0.7020437651809734, 0.6611684983372949], 'rec_text': ['www.997', '151', 'PASS', '登机牌', 'BOARDING', '舱位 CLASS', '序号SERIALNO.', '座位号SEATNO', '航班 FLIGHT', '日期DATE', 'MU 2379', '03DEC', 'W', '035', 'F', '1', '始发地FROM', '登机口 GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO.', '姓名NAME', 'ZHANGQIWEI', '票号TKTNO.', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭GATESCLOSE1OMINUTESBEFOREDEPARTURETIME'], 'rec_score': [0.9617719054222107, 0.4199012815952301, 0.9652514457702637, 0.9978302121162415, 0.9853208661079407, 0.9445787072181702, 0.9714463949203491, 0.9841841459274292, 0.9564052224159241, 0.9959094524383545, 0.9386572241783142, 0.9825271368026733, 0.9356589317321777, 0.9985442161560059, 0.3965512812137604, 0.15236201882362366, 0.9976775050163269, 0.9547433257102966, 0.9974752068519592, 0.9646636843681335, 0.9907559156417847, 0.9895358681678772, 0.9374122023582458, 0.9909093379974365, 0.9796401262283325, 0.9899340271949768, 0.992210865020752, 0.9478569626808167, 0.9982215762138367, 0.9924325942993164, 0.9941263794898987, 0.96443772315979]}
......
```

可视化结果如下：

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png)


其他产线的命令行使用，只需将 `pipeline` 参数调整为相应产线的名称。下面列出了每个产线对应的命令：

<b>👉 更多产线的命令行使用</b>

| 产线名称      | 使用命令                                                                                                                                                                                             |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 通用表格识别    | `paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device gpu:0`                                             |
| 通用版面解析       | `paddlex --pipeline layout_parsing --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png --device gpu:0`                                      |
| 公式识别       | `paddlex --pipeline formula_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png --device gpu:0`                                      |
| 印章文本识别       | `paddlex --pipeline seal_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png --device gpu:0`                             |
