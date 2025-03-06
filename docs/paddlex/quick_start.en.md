# ⏭️ Quick Start

## 🛠️ Installation

> ❗Before installing PaddleX, please ensure you have a basic **Python runtime environment** (Note: Currently supports running under Python 3.8 to Python 3.10, with more Python versions under adaptation). The PaddlePaddle version required by PaddleX

* **Installing PaddlePaddle**

```bash
# CPU
python -m pip install paddlepaddle==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# gpu，requires GPU driver version ≥450.80.02 (Linux) or ≥452.39 (Windows)
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# gpu，requires GPU driver version ≥545.23.06 (Linux) or ≥545.84 (Windows)
python -m pip install paddlepaddle-gpu==3.0.0rc0 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```
> ❗No need to focus on the CUDA version on the physical machine, only the GPU driver version needs attention. For more information on PaddlePaddle Wheel versions, please refer to the [PaddlePaddle Official Website](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation./docs/en/install/pip/linux-pip.html).

* **Installing PaddleX**

```bash
pip install https://paddle-model-ecology.bj.bcebos.com/paddlex/whl/paddlex-3.0.0rc0-py3-none-any.whl
```

> ❗For more installation methods, refer to the [PaddleX Installation Guide](https://paddlepaddle.github.io/PaddleX/latest/en/installation/installation.html).


## 💻 CLI Usage

One command can quickly experience the pipeline effect, the unified CLI format is:

```bash
paddlex --pipeline [Pipeline Name] --input [Input Image] --device [Running Device]
```

Each Pipeline in PaddleX corresponds to specific parameters, which you can view in the respective Pipeline documentation for detailed explanations. Each Pipeline requires specifying three necessary parameters:
* `pipeline`: The name of the Pipeline or the configuration file of the Pipeline
* `input`: The local path, directory, or URL of the input file (e.g., an image) to be processed
* `device`: The hardware device and its index to use (e.g., `gpu:0` indicates using the 0th GPU), or you can choose to use NPU (`npu:0`), XPU (`xpu:0`), CPU (`cpu`), etc.

For example, using the  OCR pipeline:
```bash
paddlex --pipeline OCR \
        --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png \
        --use_doc_orientation_classify False \
        --use_doc_unwarping False \
        --use_textline_orientation False \
        --save_path ./output \
        --device gpu:0
```
  <summary><b>👉 Click to view the running result</b></summary>

```bash
{'res': {'input_path': 'general_ocr_002.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': False}, 'angle': 0},'dt_polys': [array([[ 3, 10],
       [82, 10],
       [82, 33],
       [ 3, 33]], dtype=int16), ...], 'text_det_params': {'limit_side_len': 960, 'limit_type': 'max', 'thresh': 0.3, 'box_thresh': 0.6, 'unclip_ratio': 2.0}, 'text_type': 'general', 'textline_orientation_angles': [-1, ...], 'text_rec_score_thresh': 0.0, 'rec_texts': ['www.99*', ...], 'rec_scores': [0.8980069160461426,  ...], 'rec_polys': [array([[ 3, 10],
       [82, 10],
       [82, 33],
       [ 3, 33]], dtype=int16), ...], 'rec_boxes': array([[  3,  10,  82,  33], ...], dtype=int16)}}
```

The visualization result is as follows:

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/boardingpass.png)


To use the command line for other pipelines, simply adjust the `pipeline` parameter to the name of the corresponding pipeline and modify the parameters accordingly. Below are the commands for each pipeline:

  <summary><b>👉 More CLI usage for pipelines</b></summary>

| Pipeline Name                | Command                                                                                                                                                                                    |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| OCR                      | `paddlex --pipeline OCR --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --save_path ./output --device gpu:0`                                      |
| Document Image Preprocessor      | `paddlex --pipeline doc_preprocessor --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/doc_test_rotated.jpg --use_doc_orientation_classify True --use_doc_unwarping True --save_path ./output --device gpu:0`                                                      |
| Table Recognition        | `paddlex --pipeline table_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --save_path ./output --device gpu:0`                                      |
| Table Recognition v2     | `paddlex --pipeline table_recognition_v2 --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/table_recognition.jpg --save_path ./output --device gpu:0`                                      |
| Formula Recognition              | `paddlex --pipeline formula_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/general_formula_recognition.png --use_layout_detection True --use_doc_orientation_classify False --use_doc_unwarping False --layout_threshold 0.5 --layout_nms True --layout_unclip_ratio  1.0 --layout_merge_bboxes_mode large --save_path ./output --device gpu:0`                                      |
| Seal Recognition            | `paddlex --pipeline seal_recognition --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/seal_text_det.png --use_doc_orientation_classify False --use_doc_unwarping False --device gpu:0 --save_path ./output`                                      |
| Layout Parsing           | `paddlex --pipeline layout_parsing --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --save_path ./output --device gpu:0`                      |
| Layout Parsing v2        | `paddlex --pipeline layout_parsing_v2 --input https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout_parsing_v2_demo.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False --save_path ./output --device gpu:0`                      |



## 📝 Python Script Usage

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
* Passes the image and calls the `predict()` method of the pipeline object for inference prediction
* Processes the prediction results

To use the Python script for other pipelines, simply adjust the `pipeline` parameter in the `create_pipeline()` method to the name of the corresponding pipeline and modify the parameters accordingly. Below are the parameter names and detailed usage explanations for each pipeline:

👉 More Python script usage for pipelines

| pipeline Name           | Corresponding Parameter               | Detailed Explanation                                                                                                      |
|-------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------|
| OCR | `OCR` | [Instructions for Using the General OCR Pipeline Python Script](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/OCR.html#222-python-script-integration) |
| Document Image Preprocessing | `doc_preprocessor` | [Instructions for Using the Document Image Preprocessing Pipeline Python Script](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html#212-python-script-integration) |
| Table Recognition | `table_recognition` | [Instructions for Using the General Table Recognition Pipeline Python Script](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html#22-python-script-integration) |
| Table Recognition v2 | `table_recognition_v2` | [Instructions for Using the General Table Recognition v2 Pipeline Python Script](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html#22-python-script-integration) |
| Formula Recognition | `formula_recognition` | [Instructions for Using the Formula Recognition Pipeline Python Script](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html#22-python-script-integration) |
| Seal Recognition | `seal_recognition` | [Instructions for Using the Seal Text Recognition Pipeline Python Script](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html#22-python-script-integration) |
| Layout Parsing | `layout_parsing` | [Instructions for Using the General Layout Parsing Pipeline Python Script](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html#22-python-script-integration) |
| Layout Parsing v2 | `layout_parsing_v2` | [Instructions for Using the General Layout Parsing v2 Pipeline Python Script](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/layout_parsing_v2.html#22-python-script-integration) |
| PP-ChatOCRv3-doc   | `PP-ChatOCRv3-doc` | [PP-ChatOCRv3-doc Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.html) |
| PP-ChatOCRv4-doc | `PP-ChatOCRv4-doc` | [PP-ChatOCRv4-doc Pipeline Python Script Usage Instructions](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.html) |
