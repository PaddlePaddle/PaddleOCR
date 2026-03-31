---
comments: true
hide:
  - toc
---

# 快速开始

>**说明：**

>* 飞桨低代码开发工具[PaddleX](https://github.com/PaddlePaddle/PaddleX)，依托于PaddleOCR的先进技术，支持了OCR领域的**低代码全流程**开发能力。通过低代码开发，可实现简单且高效的模型使用、组合与定制。

>* PaddleX 致力于实现产线级别的模型训练、推理与部署。模型产线是指一系列预定义好的、针对特定AI任务的开发流程，其中包含能够独立完成某类任务的单模型（单功能模块）组合。本文档提供**OCR相关产线**的快速推理使用，单功能模块的快速使用以及更多功能请参考[PaddleOCR低代码全流程开发](https://www.paddleocr.ai/latest/version3.x/paddlex/overview.html)中相关章节。

>* 关于 PaddleOCR 和 PaddleX 的关系，可以查看[ PaddleOCR 与 PaddleX 关系介绍](https://www.paddleocr.ai/latest/version3.x/paddleocr_and_paddlex.html)。


### 🛠️ 安装

> ❗在安装 PaddleX 之前，请确保您已具备基本的 **Python 运行环境**（注：目前支持 Python 3.8 至 Python 3.13）。PaddleX 3.2 版本依赖的 PaddlePaddle 版本为 3.0.0 以上版本。

* **安装 PaddlePaddle**
```bash
# CPU 版本
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# GPU 版本，需显卡驱动程序版本 ≥450.80.02（Linux）或 ≥452.39（Windows）
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# GPU 版本，需显卡驱动程序版本 ≥550.54.14（Linux）或 ≥550.54.14（Windows）
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```
> ❗无需关注物理机上的 CUDA 版本，只需关注显卡驱动程序版本。更多飞桨 Wheel 版本信息，请参考[飞桨官网](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/zh/install/pip/linux-pip.html)。

* **安装PaddleX**

```bash
pip install "paddlex[ocr]"
```

> ❗ 更多安装方式参考 [PaddleX 安装教程](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html)

### 💻 命令行使用

一行命令即可快速体验产线效果，统一的命令行格式为：

```bash
paddlex --pipeline [产线名称] --input [输入图片] --device [运行设备]
```

PaddleX的每一条产线对应特定的参数，您可以在各自的产线文档中查看具体的参数说明。每条产线需指定必要的三个参数：
* `pipeline`：产线名称或产线配置文件
* `input`：待处理的输入文件（如图片）的本地路径、目录或 URL
* `device`：使用的硬件设备及序号（例如`gpu:0`表示使用第 0 块 GPU），也可选择使用 NPU(`npu:0`)、 XPU(`xpu:0`)、CPU(`cpu`)等。


以通用 OCR 产线为例：
```bash
paddlex --pipeline OCR \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device gpu:0
```

<b>👉 点击查看运行结果 </b><

```bash
{'res': {'input_path': 'general_ocr_002.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': False}, 'angle': 0},'dt_polys': [array([[ 3, 10],
       [82, 10],
       [82, 33],
       [ 3, 33]], dtype=int16), ...], 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': [-1, ...], 'text_rec_score_thresh': 0.0, 'rec_texts': ['www.99*', ...], 'rec_scores': [0.8980069160461426,  ...], 'rec_polys': [array([[ 3, 10],
       [82, 10],
       [82, 33],
       [ 3, 33]], dtype=int16), ...], 'rec_boxes': array([[  3,  10,  82,  33], ...], dtype=int16)}}
```

可视化结果如下：

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png)


其他产线的命令行使用，只需将 `pipeline` 参数调整为相应产线的名称，参数调整为对应的产线的参数即可。下面列出了每个产线对应的命令：

<b>👉 更多产线的命令行使用</b>

| 产线名称           | 使用命令                                                                                                                                                                                    |
|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 文档图像预处理            | `paddlex --pipeline doc_preprocessor --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg --use_doc_orientation_classify True --use_doc_unwarping True --save_path ./output --device gpu:0`                                                      |
| 通用OCR            | `paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --save_path ./output --device gpu:0`                                                      |
| 通用表格识别       | `paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --save_path ./output --device gpu:0`                                      |
| 通用表格识别v2       | `paddlex --pipeline table_recognition_v2 --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --save_path ./output --device gpu:0`                                      |
| 公式识别       | `paddlex --pipeline formula_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png --use_layout_detection True --use_doc_orientation_classify False --use_doc_unwarping False --layout_threshold 0.5 --layout_nms True --layout_unclip_ratio  1.0 --layout_merge_bboxes_mode large --save_path ./output --device gpu:0`                                      |
| 印章文本识别       | `paddlex --pipeline seal_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png --use_doc_orientation_classify False --use_doc_unwarping False --device gpu:0 --save_path ./output`                                      |
| 通用版面解析       | `paddlex --pipeline layout_parsing --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --save_path ./output --device gpu:0`                                      |
| 通用版面解析v3       | `paddlex --pipeline PP-StructureV3 --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --save_path ./output --device gpu:0`                                      |


### 📝 Python 脚本使用

几行代码即可完成产线的快速推理，统一的 Python 脚本格式如下：

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
* 传入图片并调用产线对象的 `predict()` 方法进行推理预测
* 对预测结果进行处理

其他产线的 Python 脚本使用，只需将 `create_pipeline()` 方法的 `pipeline` 参数调整为相应产线的名称，参数调整为对应的产线的参数即可。下面列出了每个产线对应的参数名称及详细的使用解释：
<b>👉 更多产线的Python脚本使用</b>

| 产线名称           | 对应参数                           | 详细说明                                                                                                                                                         |
|--------------------|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 通用OCR            | `OCR`                              | [通用OCR产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/OCR.html#222-python)                                                     |
| 文档图像预处理            | `doc_preprocessor`                              | [文档图像预处理产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html#212-python脚)                       |
| 通用表格识别       | `table_recognition`                | [通用表格识别产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html#22-python脚)                                   |
| 通用表格识别v2      | `table_recognition_v2`                | [通用表格识别v2产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html#22-python脚)                                   |
| 公式识别       | `formula_recognition`                | [公式识别产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html#222-python)                                   |
| 印章文本识别       | `seal_recognition`                | [印章文本识别产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html#222-python)                                   |
| 通用版面解析       | `layout_parsing`                | [通用版面解析产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html#22-python)                                   |
| 通用版面解析v3      | `PP-StructureV3`                | [通用版面解析v3产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/PP-StructureV3.html#22-python)                                   |
| 文档场景信息抽取v3   | `PP-ChatOCRv3-doc`                 | [文档场景信息抽取v3产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.html#22) |
| 文档场景信息抽取v4   | `PP-ChatOCRv4-doc`                 | [文档场景信息抽取v4产线Python脚本使用说明](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.html#21) |
