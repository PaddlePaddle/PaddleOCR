# Quick Start

> **Note:**
>* The All-in-One development tool [PaddleX](https://github.com/PaddlePaddle/PaddleX/tree/release/3.0-beta1), based on the advanced technology of PaddleOCR, supports **all-in-one** development capabilities in the OCR field. Through all-in-one development, simple and efficient model use, combination, and customization can be achieved.
>* PaddleX is committed to achieving pipeline-level model training, inference, and deployment. A model pipeline refers to a series of predefined development processes for specific AI tasks, including combinations of single models (single-function modules) that can independently complete a type of task. This document provides quick inference usage of the **OCR-related pipelines**. For quick usage of single-function modules and more features, please refer to the relevant sections in [All-in-One Development of PaddleOCR](https://paddlepaddle.github.io/PaddleOCR/latest/en/paddlex/overview.html).

### 🛠️ Installation

> ❗Before installing PaddleX, please ensure you have a basic **Python environment** (Note: Currently supports Python 3.8 to Python 3.10, with more Python versions being adapted).
* **Installing PaddlePaddle**
```bash
# cpu
python -m pip install paddlepaddle==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，该命令仅适用于 CUDA 版本为 11.8 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，该命令仅适用于 CUDA 版本为 12.3 的机器环境
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗For more PaddlePaddle versions, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html). 
* **Installing PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0b1-py3-none-any.whl
```

> ❗For more installation methods, refer to the [PaddleX Installation Guide](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/installation/installation_en.md).
### 📝 Python Script Usage

A few lines of code can complete the quick inference of the pipeline, the unified Python script format is as follows:

```python
from paddlex import create_pipeline

pipeline = create_pipeline(pipeline=[Pipeline Name])
output = pipeline.predict([Input Image Name])
for res in output:
    res.print()
    res.save_to_img("./output/")
    res.save_to_json("./output/")
```
The following steps are executed:

* `create_pipeline()` instantiates the pipeline object
* Passes the image and calls the `predict` method of the pipeline object for inference prediction
* Processes the prediction results

For other pipelines in Python scripts, just adjust the `pipeline` parameter of the `create_pipeline()` method to the corresponding name of the pipeline. Below is a list of each pipeline's corresponding parameter name and detailed usage explanation:

<b>👉 More Python Script Usages for Pipelines</b>

| Pipeline Name          | Corresponding Parameter  | Detailed Explanation                                                                                                     |
|------------------------|--------------------------|---------------------------------------------------------------------------------------------------------------------------|
| PP-ChatOCRv3-doc  | `PP-ChatOCRv3-doc` | [Python Script Usage for PP-ChatOCRv3-doc Pipeline](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/information_extration_pipelines/document_scene_information_extraction_en.md#22-local-experience) |
| General OCR            | `OCR` | [Python Script Usage for General OCR Pipeline](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/OCR_en.md#22-local-experience) |
| Table Recognition       | `table_recognition` | [Python Script Usage for Table Recognition Pipeline](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/table_recognition_en.md#22-local-experience) |
| Layout Parsing       | `layout_parsing`                | [Python Script Usage for Layout Parsing Pipeline](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/layout_parsing_en.md#22-local-experience)                                   |
| Formula Recognition       | `formula_recognition`                | [Python Script Usage for Formula Recognition Pipeline](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/formula_recognition_en.md#2-quick-start)                                   |
| Seal Recognition       | `seal_recognition`                | [Python Script Usage for Formula Recognition Pipeline](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/ocr_pipelines/seal_recognition_en.md#2--quick-start)                                          |

### 💻 Command Line Usage

You can quickly experience the pipeline effect with a single command. The unified command line format is:

```bash
paddlex --pipeline [pipeline_name] --input [input_image] --device [device]
```

You only need to specify three parameters:
* `pipeline`: the name of the pipeline
* `input`: the local path or URL of the input file (e.g., image) to be processed
* `device`: the GPU index to use (e.g., `gpu:0` means using the first GPU), or you can choose to use the CPU (`cpu`)

For example, for the General OCR pipeline:
```bash
paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --device gpu:0
```

The result of the operation is as follows:

```bash
{'input_path': '/root/.paddlex/predict_input/general_ocr_002.png', 'dt_polys': [[[5, 12], [88, 10], [88, 29], [5, 31]], [[208, 14], [249, 14], [249, 22], [208, 22]], [[695, 15], [824, 15], [824, 60], [695, 60]], [[158, 27], [355, 23], [356, 70], [159, 73]], [[421, 25], [659, 19], [660, 59], [422, 64]], [[337, 104], [460, 102], [460, 127], [337, 129]], [[486, 103], [650, 100], [650, 125], [486, 128]], [[675, 98], [835, 94], [835, 119], [675, 124]], [[64, 114], [192, 110], [192, 131], [64, 134]], [[210, 108], [318, 106], [318, 128], [210, 130]], [[82, 140], [214, 138], [214, 163], [82, 165]], [[226, 136], [328, 136], [328, 161], [226, 161]], [[404, 134], [432, 134], [432, 161], [404, 161]], [[509, 131], [570, 131], [570, 158], [509, 158]], [[730, 138], [771, 138], [771, 154], [730, 154]], [[806, 136], [817, 136], [817, 146], [806, 146]], [[342, 175], [470, 173], [470, 197], [342, 199]], [[486, 173], [616, 171], [616, 196], [486, 198]], [[677, 169], [813, 166], [813, 191], [677, 194]], [[65, 181], [170, 177], [171, 202], [66, 205]], [[96, 208], [171, 205], [172, 230], [97, 232]], [[336, 220], [476, 215], [476, 237], [336, 242]], [[507, 217], [554, 217], [554, 236], [507, 236]], [[87, 229], [204, 227], [204, 251], [87, 254]], [[344, 240], [483, 236], [483, 258], [344, 262]], [[66, 252], [174, 249], [174, 271], [66, 273]], [[75, 279], [264, 272], [265, 297], [76, 303]], [[459, 297], [581, 295], [581, 320], [459, 322]], [[101, 314], [210, 311], [210, 337], [101, 339]], [[68, 344], [165, 340], [166, 365], [69, 368]], [[345, 350], [662, 346], [662, 368], [345, 371]], [[100, 459], [832, 444], [832, 465], [100, 480]]], 'dt_scores': [0.8183103704439653, 0.7609575621092027, 0.8662357274035412, 0.8619508290334809, 0.8495855993183273, 0.8676840017933314, 0.8807986687956436, 0.822308525056085, 0.8686617037621976, 0.8279022169854463, 0.952332847006758, 0.8742692553015098, 0.8477013022907575, 0.8528771493227294, 0.7622965906848765, 0.8492388224448705, 0.8344203789965632, 0.8078477124353284, 0.6300434587457232, 0.8359967356998494, 0.7618617265751318, 0.9481573079350023, 0.8712182945408912, 0.837416955846334, 0.8292475059403851, 0.7860382856406026, 0.7350527486717117, 0.8701022267947695, 0.87172526903969, 0.8779847108088126, 0.7020437651809734, 0.6611684983372949], 'rec_text': ['www.997', '151', 'PASS', '登机牌', 'BOARDING', '舱位 CLASS', '序号SERIALNO.', '座位号SEATNO', '航班 FLIGHT', '日期DATE', 'MU 2379', '03DEC', 'W', '035', 'F', '1', '始发地FROM', '登机口 GATE', '登机时间BDT', '目的地TO', '福州', 'TAIYUAN', 'G11', 'FUZHOU', '身份识别IDNO.', '姓名NAME', 'ZHANGQIWEI', '票号TKTNO.', '张祺伟', '票价FARE', 'ETKT7813699238489/1', '登机口于起飞前10分钟关闭GATESCLOSE1OMINUTESBEFOREDEPARTURETIME'], 'rec_score': [0.9617719054222107, 0.4199012815952301, 0.9652514457702637, 0.9978302121162415, 0.9853208661079407, 0.9445787072181702, 0.9714463949203491, 0.9841841459274292, 0.9564052224159241, 0.9959094524383545, 0.9386572241783142, 0.9825271368026733, 0.9356589317321777, 0.9985442161560059, 0.3965512812137604, 0.15236201882362366, 0.9976775050163269, 0.9547433257102966, 0.9974752068519592, 0.9646636843681335, 0.9907559156417847, 0.9895358681678772, 0.9374122023582458, 0.9909093379974365, 0.9796401262283325, 0.9899340271949768, 0.992210865020752, 0.9478569626808167, 0.9982215762138367, 0.9924325942993164, 0.9941263794898987, 0.96443772315979]}
......
```

The visualization result is as follows:

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png)

For command line usage of other pipelines, simply adjust the `pipeline` parameter to the name of the respective pipeline. Below are the commands corresponding to each pipeline:

<b>👉 More Command Line Usages for Pipelines</b>

| Pipeline Name     | Command                                                                                                                                                                                               |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Table Recognition | `paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --device gpu:0`                                             |
| Layout Parsing       | `paddlex --pipeline layout_parsing --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png --device gpu:0`                                      |
| Formula Recognition       | `paddlex --pipeline formula_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png --device gpu:0`                                      |
| Seal Recognition       | `paddlex --pipeline seal_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png --device gpu:0`     
