## 1. Introduction to All-in-One Development

The All-in-One development tool [PaddleX](https://github.com/PaddlePaddle/PaddleX/tree/release/3.0-beta1), based on the advanced technology of PaddleOCR, supports **low-code full-process** development capabilities in the OCR field. Through low-code development, simple and efficient model use, combination, and customization can be achieved. This will significantly **reduce the time consumption** of model development, **lower its development difficulty**, and greatly accelerate the application and promotion speed of models in the industry. Features include:

* 🎨 [**Rich Model One-Click Call**](https://paddlepaddle.github.io/PaddleOCR/latest/en/paddlex/quick_start.html): Integrates **48 models** related to text image intelligent analysis, general OCR, general layout parsing, table recognition, formula recognition, and seal recognition into 10 pipelines, which can be quickly experienced through a simple **Python API one-click call**. In addition, the same set of APIs also supports a total of **200+ models** in image classification, object detection, image segmentation, and time series forcasting, forming 30+ single-function modules, making it convenient for developers to use **model combinations**.

* 🚀 [**High Efficiency and Low barrier of entry**](https://paddlepaddle.github.io/PaddleOCR/latest/en/paddlex/overview.html): Provides two methods based on **unified commands** and **GUI** to achieve simple and efficient use, combination, and customization of models. Supports multiple deployment methods such as **high-performance inference, service-oriented deployment, and edge deployment**. Additionally, for various mainstream hardware such as **NVIDIA GPU, Kunlunxin XPU, Ascend NPU, Cambricon MLU, and Haiguang DCU**, models can be developed with **seamless switching**.

> **Note**: PaddleX is committed to achieving pipeline-level model training, inference, and deployment. A model pipeline refers to a series of predefined development processes for specific AI tasks, including combinations of single models (single-function modules) that can independently complete a type of task.
## 2. OCR-Related Capability Support

In PaddleX, all 6 OCR-related pipelines support **local inference**, and some pipelines support **online experience**. You can quickly experience the pre-trained model effects of each pipeline. If you are satisfied with the pre-trained model effects of a pipeline, you can directly proceed with [high-performance inference](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_deploy/high_performance_inference_en.md)/[service-oriented deployment](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_deploy/service_deploy_en.md)/[edge deployment](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_deploy/edge_deploy_en.md). If not satisfied, you can also use the **custom development** capabilities of the pipeline to improve the effects. For the complete pipeline development process, please refer to [PaddleX Pipeline Usage Overview](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/pipeline_develop_guide_en.md) or the tutorials for each pipeline.

In addition, PaddleX provides developers with a full-process efficient model training and deployment tool based on a [cloud-based GUI](https://aistudio.baidu.com/pipeline/mine). Developers **do not need code development**, just need to prepare a dataset that meets the pipeline requirements to **quickly start model training**. For details, please refer to the tutorial ["Developing Industrial-level AI Models with Zero Barrier"](https://aistudio.baidu.com/practical/introduce/546656605663301).

<table>
    <tr>
        <th>Pipeline</th>
        <th>Online Experience</th>
        <th>Local Inference</th>
        <th>High-Performance Inference</th>
        <th>Service-Oriented Deployment</th>
        <th>Edge Deployment</th>
        <th>Custom Development</th>
        <th><a href="https://aistudio.baidu.com/pipeline/mine">No-Code Development On AI Studio</a></td> 
    </tr>
   <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html">Document Image Preprocessing</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html">OCR</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/91660/webUI?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html">Table Recognition</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/91661?source=appMineRecent">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html">Table Recognition V2</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
   </tr>
        <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html">Formula Recognition</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/387976/webUI?source=appCenter">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html">Seal Recognition</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/387977/webUI?source=appCenter">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html">Layout Parsing</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing_v2.html">Layout Parsing v2</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
        <td>🚧</td>
        <td>🚧</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html">PP-ChatOCRv3-doc</a></td>
        <td><a href = "https://aistudio.baidu.com/community/app/182491/webUI?source=appCenter">Link</a></td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
    </tr>
    <tr>
        <td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html">PP-ChatOCRv4-doc</a></td>
        <td>🚧</td>
        <td>✅</td>
        <td>✅</td>
        <td>✅</td>
        <td>🚧</td>
        <td>✅</td>
        <td>🚧</td>
    </tr>
</table>


</table>


> ❗Note: The above capabilities are implemented based on GPU/CPU. PaddleX can also perform local inference and custom development on mainstream hardware such as Kunlunxin, Ascend, Cambricon, and Haiguang. The table below details the support status of the pipelines. For specific supported model lists, please refer to the [Model List (Kunlunxin XPU)](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/support_list/model_list_xpu_en.md)/[Model List (Ascend NPU)](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/support_list/model_list_npu_en.md)/[Model List (Cambricon MLU)](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/support_list/model_list_mlu_en.md)/[Model List (Haiguang DCU)](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/support_list/model_list_dcu_en.md). We are continuously adapting more models and promoting the implementation of high-performance and service-oriented deployment on mainstream hardware.
**🚀 Support for Domestic Hardware Capabilities**

<table>
  <tr>
    <th>Pipeline Name</th>
    <th>Ascend 910B</th>
    <th>Kunlunxin XPU</th>
    <th>Cambricon MLU</th>
    <th>Haiguang DCU</th>
  </tr>
  <tr>
    <td>General OCR</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
    <td>🚧</td>
  </tr>
  <tr>
    <td>Table Recognition</td>
    <td>✅</td>
    <td>🚧</td>
    <td>🚧</td>
    <td>🚧</td>
  </tr>
</table>

## 3. List and Tutorials of OCR-Related Model Pipelines

- **OCR Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html)
- **Table Recognition Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html)
- **PP-ChatOCRv3-doc Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.html)
- **Layout Parsing Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html)
- **Formula Recognition Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html)
- **Seal Recognition Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html)

## 4. List and Tutorials of OCR-Related Modules

- **Text Detection Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_detection.html)
- **Seal Detection Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/seal_text_detection.html)
- **Text Recognition Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html)
- **Formula Recognition Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html)
- **Table Structure Recognition Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html)
- **Text Image Unwarping Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_image_unwarping.html)
- **Layout Detection Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)
- **Document Image Orientation Classification Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html)




## 3. List of OCR-related Pipeline Models and Tutorials

- **Document Image Preprocessing Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html)
- **OCR Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/OCR.html)
- **Table Recognition Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition.html)
- **Table Recognition v2 Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html)
- **Layout Parsing Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing.html)
- **Layout Parsing v2 Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing_v2.html)
- **Formula Recognition**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/formula_recognition.html)
- **Seal Recognition**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/seal_recognition.html)
- **PP-ChatOCRv3-doc Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v3.html)
- **PP-ChatOCRv4-doc Pipeline**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.html)

## 4. List of OCR-related Single Function Modules and Tutorials

- **Text Detection Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_detection.html)
- **Seal Detection Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/seal_text_detection.html)
- **Textline Orientation Classification Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html)
- **Text Recognition Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html)
- **Formula Recognition Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html)
- **Table Structure Recognition Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html)
- **Text Image Unwarping Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_image_unwarping.html)
- **Layout Detection Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)
- **Document Image Orientation Classification Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html)
- **Table Cells Detection Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_cells_detection.html)
- **Table Classification Module**: [Tutorial](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_classification.html)
