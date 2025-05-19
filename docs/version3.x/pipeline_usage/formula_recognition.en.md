---
comments: true
---

# Formula Recognition Pipeline Tutorial

## 1. Introduction to Formula Recognition Pipeline

Formula recognition is a technology that automatically identifies and extracts LaTeX formula content and structure from documents or images. It is widely used in fields such as mathematics, physics, and computer science for document editing and data analysis. By using computer vision and machine learning algorithms, formula recognition can convert complex mathematical formula information into editable LaTeX format, facilitating further processing and analysis of data.

The formula recognition pipeline is designed to solve formula recognition tasks by extracting formula information from images and outputting it in LaTeX source code format. This pipeline integrates the advanced formula recognition model PP-FormulaNet developed by the PaddlePaddle Vision Team and the well-known formula recognition model UniMERNet. It is an end-to-end formula recognition system that supports the recognition of simple printed formulas, complex printed formulas, and handwritten formulas. Additionally, it includes functions for image orientation correction and distortion correction. Based on this pipeline, precise formula content prediction can be achieved, covering various application scenarios in education, research, finance, manufacturing, and other fields. The pipeline also provides flexible deployment options, supporting multiple hardware devices and programming languages. Moreover, it offers the capability for custom development. You can train and optimize the pipeline on your own dataset, and the trained model can be seamlessly integrated.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/03.png" style="width: 70%"/>

<b>
The formula recognition pipeline includes the following four modules. Each module can be trained and inferred independently and contains multiple models. For more details, please click on the respective module to view the documentation.</b>

- [Formula Recognition Module](../module_usage/formula_recognition.en.md)
- [Layout Detection Module](../module_usage/layout_detection.en.md)（Optional）
- [Document Image Orientation Classification Module](../module_usage/doc_img_orientation_classification.en.md) （Optional）
- [Text Image Correction Module](../module_usage/text_image_unwarping.en.md) （Optional）


In this pipeline, you can choose the model you want to use based on the benchmark data provided below.

<details>
<summary><b>Document Image Orientation Classification Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>Top-1 Acc (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">Training Model</a></td>
<td>99.06</td>
<td>2.31 / 0.43</td>
<td>3.37 / 1.27</td>
<td>7</td>
<td>A document image classification model based on PP-LCNet_x1_0, with four categories: 0 degrees, 90 degrees, 180 degrees, and 270 degrees.</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Text Image Correction Module (Optional):</b></summary>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>CER</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">Training Model</a></td>
<td>0.179</td>
<td>30.3 M</td>
<td>High-precision text image correction model</td>
</tr>
</tbody>
</table>
</details>

<details>
<summary><b>Layout Detection Module (Optional):</b></summary>

* <b>The layout detection model includes 20 common categories: document title, paragraph title, text, page number, abstract, table, references, footnotes, header, footer, algorithm, formula, formula number, image, table, seal, figure_table title, chart, and sidebar text and lists of references</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">Training Model</a></td>
<td>83.2</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
<td>126.01 M</td>
<td>A higher-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, PPT, multi-layout magazines, contracts, books, exams, ancient books and research reports using RT-DETR-L</td>
</tr>
<tr>
</tbody>
</table>

* <b>The layout detection model includes 23 common categories: document title, paragraph title, text, page number, abstract, table of contents, references, footnotes, header, footer, algorithm, formula, formula number, image, figure caption, table, table caption, seal, figure title, figure, header image, footer image, and sidebar text</b>
<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / -</td>
<td>123.76 M</td>
<td>A high-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>A layout area localization model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
<td>4.834</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-S.</td>
</tr>
</tbody>
</table>

>❗ The above list includes the <b>4 core models</b> that are key supported by the layout detection module. The module actually supports a total of <b>7 full models</b>, including several predefined models with different categories. The complete model list is as follows:

<details><summary> 👉Details of Model List</summary>

* <b>Layout Detection Model, including 17 common layout categories: Paragraph Title, Image, Text, Number, Abstract, Content, Figure Caption, Formula, Table, Table Caption, References, Document Title, Footnote, Header, Algorithm, Footer, and Stamp</b>

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>87.4</td>
<td>9.11 / 2.12</td>
<td>15.42 / 9.12</td>
<td>4.8</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-S.</td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>89.0</td>
<td>13.50 / 4.69</td>
<td>43.32 / 43.32</td>
<td>22.6</td>
<td>A balanced efficiency and precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">Training Model</a></td>
<td>98.3</td>
<td>115.29 / 104.09</td>
<td>995.27 / 995.27</td>
<td>470.2</td>
<td>A high-precision layout area localization model trained on a self-built dataset of Chinese and English papers, magazines, and research reports using RT-DETR-H.</td>
</tr>
</tbody></table>

* <b>Layout Detection Model, including 23 common layout categories: Document Title, Section Title, Text, Page Number, Abstract, Table of Contents, References, Footnotes, Header, Footer, Algorithm, Formula, Formula Number, Image, Figure Caption, Table, Table Caption, Seal, Chart Caption, Chart, Header Image, Footer Image, Sidebar Text</b>

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">Training Model</a></td>
<td>90.4</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / -</td>
<td>123.76 M</td>
<td>A high-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using RT-DETR-L.</td>
</tr>
<tr>
<td>PP-DocLayout-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">Training Model</a></td>
<td>75.2</td>
<td>13.3259 / 4.8685</td>
<td>44.0680 / 44.0680</td>
<td>22.578</td>
<td>A layout area localization model with balanced precision and efficiency, trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-L.</td>
</tr>
<tr>
<td>PP-DocLayout-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">Training Model</a></td>
<td>70.9</td>
<td>8.3008 / 2.3794</td>
<td>10.0623 / 9.9296</td>
<td>4.834</td>
<td>A high-efficiency layout area localization model trained on a self-built dataset containing Chinese and English papers, magazines, contracts, books, exams, and research reports using PicoDet-S.</td>
</tr>
</tbody>
</table>

* <b>The layout detection model includes 20 common categories: document title, paragraph title, text, page number, abstract, table, references, footnotes, header, footer, algorithm, formula, formula number, image, table, seal, figure_table title, chart, and sidebar text and lists of references</b>

<table>
<thead>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>mAP(0.5) (%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">Training Model</a></td>
<td>83.2</td>
<td>34.6244 / 10.3945</td>
<td>510.57 / - </td>
<td>126.01 M</td>
<td>A higher-precision layout area localization model trained on a self-built dataset containing Chinese and English papers, PPT, multi-layout magazines, contracts, books, exams, ancient books and research reports using RT-DETR-L</td>
</tr>
<tr>
</tbody>
</table>


</details>
</details>

<details>
<summary><b>Formula Recognition Module ：</b></summary>
<table>
<tr>
<th>Model</th><th>Model Download Link</th>
<th>En-BLEU(%)</th>
<th>Zh-BLEU(%)</th>
<th>GPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Normal Mode / High-Performance Mode]</th>
<th>Model Storage Size (M)</th>
<th>Introduction</th>
</tr>
<td>UniMERNet</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">Training Model</a></td>
<td>85.91</td>
<td>43.50</td>
<td>2266.96/-</td>
<td>-/-</td>
<td>1.53 G</td>
<td>UniMERNet is a formula recognition model developed by Shanghai AI Lab. It uses Donut Swin as the encoder and MBartDecoder as the decoder. The model is trained on a dataset of one million samples, including simple formulas, complex formulas, scanned formulas, and handwritten formulas, significantly improving the recognition accuracy of real-world formulas.</td>
<tr>
<td>PP-FormulaNet-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">Training Model</a></td>
<td>87.00</td>
<td>45.71</td>
<td>202.25/-</td>
<td>-/-</td>
<td>224 M</td>
<td rowspan="2">PP-FormulaNet is an advanced formula recognition model developed by the Baidu PaddlePaddle Vision Team. The PP-FormulaNet-S version uses PP-HGNetV2-B4 as its backbone network. Through parallel masking and model distillation techniques, it significantly improves inference speed while maintaining high recognition accuracy, making it suitable for applications requiring fast inference. The PP-FormulaNet-L version, on the other hand, uses Vary_VIT_B as its backbone network and is trained on a large-scale formula dataset, showing significant improvements in recognizing complex formulas compared to PP-FormulaNet-S.</td>
</tr>
<td>PP-FormulaNet-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">Training Model</a></td>
<td>90.36</td>
<td>45.78</td>
<td>1976.52/-</td>
<td>-/-</td>
<td>695 M</td>
<tr>
<td>PP-FormulaNet_plus-S</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-S_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-S_pretrained.pdparams">Training Model</a></td>
<td>88.71</td>
<td>53.32</td>
<td>191.69/-</td>
<td>-/-</td>
<td>248 M</td>
<td rowspan="3">PP-FormulaNet_plus is an enhanced version of the formula recognition model developed by the Baidu PaddlePaddle Vision Team, building upon the original PP-FormulaNet. Compared to the original version, PP-FormulaNet_plus utilizes a more diverse formula dataset during training, including sources such as Chinese dissertations, professional books, textbooks, exam papers, and mathematics journals. This expansion significantly improves the model’s recognition capabilities. Among the models, PP-FormulaNet_plus-M and PP-FormulaNet_plus-L have added support for Chinese formulas and increased the maximum number of predicted tokens for formulas from 1,024 to 2,560, greatly enhancing the recognition performance for complex formulas. Meanwhile, the PP-FormulaNet_plus-S model focuses on improving the recognition of English formulas. With these improvements, the PP-FormulaNet_plus series models perform exceptionally well in handling complex and diverse formula recognition tasks. </td>
</tr>
<tr>
<td>PP-FormulaNet_plus-M</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-M_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-M_pretrained.pdparams">Training Model</a></td>
<td>91.45</td>
<td>89.76</td>
<td>1301.56/-</td>
<td>-/-</td>
<td>592 M</td>
</tr>
<tr>
<td>PP-FormulaNet_plus-L</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-L_pretrained.pdparams">Training Model</a></td>
<td>92.22</td>
<td>90.64</td>
<td>1745.25/-</td>
<td>-/-</td>
<td>698 M</td>
</tr>
<tr>
<td>LaTeX_OCR_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">Training Model</a></td>
<td>74.55</td>
<td>39.96</td>
<td>1244.61/-</td>
<td>-/-</td>
<td>99 M</td>
<td>LaTeX-OCR is a formula recognition algorithm based on an autoregressive large model. It uses Hybrid ViT as the backbone network and a transformer as the decoder, significantly improving the accuracy of formula recognition.</td>
</tr>
</table>
</details>

<details>
<summary><b>Test Environment Description: </b></summary>

  <ul>
      <li><b>Performance Test Environment</b>
          <ul>
            <li><strong>Test Dataset：
             </strong>
                <ul>
                  <li>Document Image Orientation Classification Module: A self-built dataset using PaddleX, covering multiple scenarios such as ID cards and documents, containing 1000 images.</li>
                  <li> Text Image Rectification Module: <a href="https://www3.cs.stonybrook.edu/~cvl/docunet.html">DocUNet</a>。</li>
                  <li>Layout Region Detection Module: A self-built layout region detection dataset using PaddleOCR, including 500 images of common document types such as Chinese and English papers, magazines, contracts, books, exam papers, and research reports.</li>
                  <li>17-Class Region Detection Model: A self-built layout region detection dataset using PaddleOCR, including 892 images of common document types such as Chinese and English papers, magazines, and research reports.</li>
                  <li>Formula Recognition Module: A self-built formula recognition test set using PaddleX.</li>
                </ul>
             </li>
             <li><strong>Hardware Configuration：</strong>
                  <ul>
                      <li>GPU: NVIDIA Tesla T4</li>
                      <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
                      <li>Other Environments: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
                  </ul>
              </li>
          </ul>
      </li>
      <li><b>Inference Mode Description</b></li>
  </ul>

<table border="1">
    <thead>
        <tr>
            <th>Mode</th>
            <th>GPU Configuration </th>
            <th>CPU Configuration </th>
            <th>Acceleration Technology Combination</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Normal Mode</td>
            <td>FP32 Precision / No TRT Acceleration</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>PaddleInference</td>
        </tr>
        <tr>
            <td>High-Performance Mode</td>
            <td>Optimal combination of pre-selected precision types and acceleration strategies</td>
            <td>FP32 Precision / 8 Threads</td>
            <td>Pre-selected optimal backend (Paddle/OpenVINO/TRT, etc.)</td>
        </tr>
    </tbody>
</table>
</details>
<br />
<b>If you prioritize model accuracy, choose a model with higher precision; if you care more about inference speed, choose a faster model; if you are concerned about model storage size, choose a smaller model.</b>

## 2. Quick Start

Before using the formula recognition pipeline locally, please ensure that you have completed the wheel package installation according to the [installation guide](../ppocr/installation.en.md). Once installed, you can experience it locally via the command line or integrate it with Python.


### 2.1 Command Line Experience

You can quickly experience the effect of the formula recognition pipeline with one command：

```bash
paddleocr formula_recognition_pipeline -i https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png

# Specify whether to use the document orientation classification model with --use_doc_orientation_classify.
paddleocr formula_recognition_pipeline -i ./general_formula_recognition_001.png --use_doc_orientation_classify True

# Specify whether to use the text image unwarping module with --use_doc_unwarping.
paddleocr formula_recognition_pipeline -i ./general_formula_recognition_001.png --use_doc_unwarping True

# Specify the use of GPU for model inference with --device.
paddleocr formula_recognition_pipeline -i ./general_formula_recognition_001.png --device gpu
```

<details><summary><b>The command line supports more parameter settings. Click to expand for detailed descriptions of the command line parameters.</b></summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td> 
The name of the document orientation classification model. If set to <code>None</code>, the default model in pipeline will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>The directory path of the document orientation classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_batch_size</code></td>
<td>The batch size of the document orientation classification model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.
</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td> The name of the text image unwarping model. If set to <code>None</code>, the default model in pipeline will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td> The directory path of the  text image unwarping model. If set to <code>None</code>, the official model will be downloaded.
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_batch_size</code></td>
<td>The batch size of the text image unwarping model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>
The name of the layout detection model. If set to <code>None</code>, the default model in pipeline will be used. </td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td> The directory path of the  layout detection model. If set to <code>None</code>, the official model will be downloaded.
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Threshold for layout detection, used to filter out predictions with low confidence.
<ul>
<li><b>float</b>， such as 0.2, indicates filtering out all bounding boxes with a confidence score less than 0.2.</li>
<li><b>Dictionary</b>, with <b>int</b> keys representing <code>cls_id</code> and <b>float</b> values as thresholds. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> indicates applying a threshold of 0.45 for class ID 0, 0.48 for class ID 2, and 0.4 for class ID 7</li>
<li><b>None</b>, If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>
Whether to use NMS (Non-Maximum Suppression) post-processing for layout region detection to filter out overlapping boxes. If set to <code>None</code>, the default configuration of the official model will be used.
</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>
The scaling factor for the side length of the detection boxes in layout region detection.
<ul>
<li><b>float</b>: A positive float number, e.g., 1.1, indicating that the center of the bounding box remains unchanged while the width and height are both scaled up by a factor of 1.1</li>
<li><b>List</b>: e.g., [1.2, 1.5], indicating that the center of the bounding box remains unchanged while the width is scaled up by a factor of 1.2 and the height by a factor of 1.5</li>
<li><b>None</b>: If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>float|list</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The merging mode for the detection boxes output by the model in layout region detection.
<ul>
<li><b>large</b>: When set to "large", only the largest outer bounding box will be retained for overlapping bounding boxes, and the inner overlapping boxes will be removed.</li>
<li><b>small</b>: When set to "small", only the smallest inner bounding boxes will be retained for overlapping bounding boxes, and the outer overlapping boxes will be removed.</li>
<li><b>union</b>: No filtering of bounding boxes will be performed, and both inner and outer boxes will be retained.</li>
<li><b>None</b>: If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_batch_size</code></td>
<td>The batch size for the layout region detection model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load the document orientation classification module. If set to <code>None</code>, the parameter will default to the value initialized in the pipeline, which is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>
Whether to load the text image unwarping module. If set to <code>None</code>, the parameter will default to the value initialized in the pipeline, which is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>
Whether to load the layout detection module. If set to <code>None</code>, the parameter will default to the value initialized in the pipeline, which is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>
The name of the formula recognition model. If set to <code>None</code>, the default model from the pipeline will be used.
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>The directory path of the formula recognition model. If set to <code>None</code>, the official model will be downloaded.
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>The batch size for the formula recognition model. If set to  <code>None</code>, the batch size will default to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types, required.
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., network URL of image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png">Example</a>; <b>Local directory</b>, the directory should contain images to be predicted, e.g., local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with a specific file path)</li>
<li><b>List</b>: Elements of the list must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>
Specify the path to save the inference results file. If set to <code>None</code>, the inference results will not be saved locally.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. You can specify a particular card number.
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> indicates using the 1st GPU for inference;</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> indicates using the 1st NPU for inference;</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> indicates using the 1st XPU for inference;</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> indicates using the 1st MLU for inference;</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> indicates using the 1st DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used. During initialization, the local GPU 0 will be prioritized; if unavailable, the CPU will be used.</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable the high-performance inference plugin.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to use TensorRT for inference acceleration.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>Minimum subgraph size for optimizing the computation of model subgraphs. </td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Compute precision, such as FP32 or FP16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable the MKL-DNN acceleration library. If set to <code>None</code>, it will be enabled by default.
</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>
The number of threads to use when performing inference on the CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
</tbody>
</table>
</details>
<br />


The results of running the default configuration of the formula recognition pipeline will be printed to the terminal as follows:

```bash
{'res': {'input_path': '/root/.paddlex/predict_input/general_formula_recognition_001.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_layout_detection': True}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 0}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.985754132270813, 'coordinate': [89.7193, 1086.395, 658.7713, 1553.3423]}, {'cls_id': 2, 'label': 'text', 'score': 0.9843020439147949, 'coordinate': [92.333496, 128.53197, 665.28827, 396.77502]}, {'cls_id': 2, 'label': 'text', 'score': 0.9766702651977539, 'coordinate': [698.1615, 590.9516, 1293.0541, 747.957]}, {'cls_id': 2, 'label': 'text', 'score': 0.9720445275306702, 'coordinate': [697.04236, 752.37866, 1289.7733, 883.6316]}, {'cls_id': 2, 'label': 'text', 'score': 0.969851016998291, 'coordinate': [92.62311, 799.51917, 660.5987, 901.7046]}, {'cls_id': 2, 'label': 'text', 'score': 0.968906819820404, 'coordinate': [703.46436, 81.138016, 1304.8857, 187.78355]}, {'cls_id': 2, 'label': 'text', 'score': 0.9686803221702576, 'coordinate': [691.15967, 1513.7944, 1283.3694, 1639.1626]}, {'cls_id': 2, 'label': 'text', 'score': 0.9676252007484436, 'coordinate': [700.59705, 287.55557, 1299.9479, 391.25064]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9654340147972107, 'coordinate': [727.97864, 441.2702, 1221.3971, 570.22736]}, {'cls_id': 2, 'label': 'text', 'score': 0.962298572063446, 'coordinate': [696.6504, 958.38, 1288.2308, 1033.8015]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9593353867530823, 'coordinate': [155.30962, 924.0272, 598.61615, 1036.6716]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9584437012672424, 'coordinate': [811.00867, 1058.013, 1176.5062, 1118.1985]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9583907127380371, 'coordinate': [776.84436, 208.44116, 1224.5082, 267.0984]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9574459195137024, 'coordinate': [756.9298, 1211.8248, 1190.2643, 1267.3693]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9567654728889465, 'coordinate': [723.22705, 1332.768, 1254.1936, 1469.2213]}, {'cls_id': 2, 'label': 'text', 'score': 0.9535155296325684, 'coordinate': [87.32236, 1557.9272, 656.71436, 1632.439]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9531306028366089, 'coordinate': [116.53526, 714.33014, 613.72314, 773.89496]}, {'cls_id': 2, 'label': 'text', 'score': 0.9499222040176392, 'coordinate': [95.88785, 479.01178, 663.25146, 536.5941]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9488246440887451, 'coordinate': [165.14, 558.2904, 597.4905, 613.77295]}, {'cls_id': 2, 'label': 'text', 'score': 0.9445527791976929, 'coordinate': [96.62344, 639.164, 662.406, 693.54376]}, {'cls_id': 2, 'label': 'text', 'score': 0.9438745975494385, 'coordinate': [695.2748, 1138.9849, 1286.6161, 1188.8252]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9253508448600769, 'coordinate': [195.446, 425.10272, 567.61505, 452.4903]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9198653101921082, 'coordinate': [853.006, 908.8241, 1132.3086, 933.7346]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9072502255439758, 'coordinate': [165.8695, 129.74162, 512.8529, 156.56209]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.9058157205581665, 'coordinate': [1246.4065, 1078.4541, 1287.0457, 1104.9424]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.9032580256462097, 'coordinate': [1247.0944, 1229.402, 1287.0751, 1255.4117]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.9020437598228455, 'coordinate': [1247.0134, 908.5498, 1288.0088, 934.7777]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.9014809131622314, 'coordinate': [1252.9968, 492.4516, 1294.864, 518.0589]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9007395505905151, 'coordinate': [95.99875, 236.36539, 296.8511, 266.53656]}, {'cls_id': 2, 'label': 'text', 'score': 0.899124801158905, 'coordinate': [725.15186, 395.68433, 1263.7633, 423.32642]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8968735337257385, 'coordinate': [1241.6221, 1473.1089, 1283.1995, 1498.9155]}, {'cls_id': 2, 'label': 'text', 'score': 0.891890823841095, 'coordinate': [696.93774, 1286.4127, 1083.1498, 1310.9156]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8909994959831238, 'coordinate': [1270.2683, 219.9446, 1300.0896, 246.45982]}, {'cls_id': 2, 'label': 'text', 'score': 0.8866966366767883, 'coordinate': [94.425, 1058.3392, 441.26416, 1082.0751]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8848764300346375, 'coordinate': [94.911865, 1319.9253, 263.36142, 1345.916]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8829801678657532, 'coordinate': [634.6257, 427.7292, 661.91315, 453.2417]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8760316967964172, 'coordinate': [630.8843, 939.32605, 658.2267, 965.29675]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8740068078041077, 'coordinate': [634.3623, 576.0576, 660.5403, 601.4944]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8715350031852722, 'coordinate': [633.26575, 730.2727, 660.3563, 755.6184]}, {'cls_id': 19, 'label': 'formula_number', 'score': 0.8699929714202881, 'coordinate': [630.99963, 1001.1361, 657.9286, 1025.9573]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8555153012275696, 'coordinate': [1089.9109, 1598.5446, 1277.5623, 1622.1991]}, {'cls_id': 7, 'label': 'formula', 'score': 0.833438515663147, 'coordinate': [694.6742, 1611.7349, 861.1708, 1635.6787]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7923287749290466, 'coordinate': [365.27258, 268.35327, 515.08936, 296.99475]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7592059969902039, 'coordinate': [917.18024, 1618.9021, 1009.52045, 1640.4705]}, {'cls_id': 3, 'label': 'number', 'score': 0.7468197345733643, 'coordinate': [1297.7268, 5.963439, 1310.3665, 26.294968]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7404024004936218, 'coordinate': [538.54333, 479.8123, 662.3668, 508.62253]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7284969091415405, 'coordinate': [99.5916, 508.4211, 253.29228, 535.67163]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7198922634124756, 'coordinate': [1116.627, 1572.7815, 1191.6616, 1594.5166]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7081565260887146, 'coordinate': [244.82803, 162.53033, 313.66757, 187.39536]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5778979659080505, 'coordinate': [255.87213, 323.67505, 326.8396, 349.7248]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5756691098213196, 'coordinate': [695.4659, 1561.6521, 900.0931, 1585.8818]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5596396923065186, 'coordinate': [175.38367, 350.68616, 242.63516, 376.44427]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5586691498756409, 'coordinate': [787.9116, 349.50732, 812.71045, 370.09338]}, {'cls_id': 7, 'label': 'formula', 'score': 0.546517550945282, 'coordinate': [1262.5737, 314.87128, 1296.2644, 338.0655]}, {'cls_id': 7, 'label': 'formula', 'score': 0.541178822517395, 'coordinate': [774.1763, 595.4717, 801.0121, 618.29297]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5409464836120605, 'coordinate': [848.8351, 619.38025, 959.8961, 646.0126]}]}, 'formula_res_list': [{'rec_formula': '\\small\\begin{aligned}{\\psi_{0}(M)-\\psi(M,z)=}&{{}\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\operatorname{E}}^{2}\\operatorname{ln}(10)}\\times}\\\\ {}&{{}\\int_{0}^{z}d z^{\\prime}\\frac{d t}{d z^{\\prime}}\\left.\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\operatorname{E}}},}\\\\ \\end{aligned}', 'formula_region_id': 1, 'dt_polys': ([727.97864, 441.2702, 1221.3971, 570.22736],)}, {'rec_formula': '\\begin{aligned}{\\rho_{\\mathrm{BH}}}&{{}=\\int d M\\psi(M)M}\\\\ {}&{{}=\\frac{1-\\epsilon_{r}}{\\epsilon_{r}c^{2}}\\int_{0}^{\\infty}d z\\frac{d t}{d z}\\int d\\log_{10}L\\phi(L,z)L,}\\\\ \\end{aligned}', 'formula_region_id': 2, 'dt_polys': ([155.30962, 924.0272, 598.61615, 1036.6716],)}, {'rec_formula': '{\\frac{d n}{d\\sigma}}d\\sigma=\\psi_{*}\\left({\\frac{\\sigma}{\\sigma_{*}}}\\right)^{\\alpha}{\\frac{e^{-(\\sigma/\\sigma_{*})^{\\beta}}}{\\Gamma(\\alpha/\\beta)}}\\beta{\\frac{d\\sigma}{\\sigma}}.', 'formula_region_id': 3, 'dt_polys': ([811.00867, 1058.013, 1176.5062, 1118.1985],)}, {'rec_formula': '\\phi(L)\\equiv\\frac{d n}{d\\log_{10}L}=\\frac{\\phi_{*}}{(L/L_{*})^{\\gamma_{1}}+(L/L_{*})^{\\gamma_{2}}}.', 'formula_region_id': 4, 'dt_polys': ([776.84436, 208.44116, 1224.5082, 267.0984],)}, {'rec_formula': '\\psi_{0}(M)=\\int d\\sigma\\frac{p(\\log_{10}M|\\log_{10}\\sigma)}{M\\log(10)}\\frac{d n}{d\\sigma}(\\sigma),', 'formula_region_id': 5, 'dt_polys': ([756.9298, 1211.8248, 1190.2643, 1267.3693],)}, {'rec_formula': '\\small\\begin{aligned}{p(\\operatorname{log}_{10}}&{{}M|\\operatorname{log}_{10}\\sigma)=\\frac{1}{\\sqrt{2\\pi}\\epsilon_{0}}}\\\\ {}&{{}\\times\\operatorname{exp}\\left[-\\frac{1}{2}\\left(\\frac{\\operatorname{log}_{10}M-a_{\\bullet}-b_{\\bullet}\\operatorname{log}_{10}\\sigma}{\\epsilon_{0}}\\right)^{2}\\right].}\\\\ \\end{aligned}', 'formula_region_id': 6, 'dt_polys': ([723.22705, 1332.768, 1254.1936, 1469.2213],)}, {'rec_formula': '\\frac{\\partial\\psi}{\\partial t}(M,t)+\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\mathrm{E}}^{2}\\ln(10)}\\left.\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\mathrm{v}}}=0,', 'formula_region_id': 7, 'dt_polys': ([116.53526, 714.33014, 613.72314, 773.89496],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle\\psi(M,t)=\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}c^{2}\\operatorname{ln}(10)}\\phi(L,t)\\frac{d L}{d M}.', 'formula_region_id': 8, 'dt_polys': ([165.14, 558.2904, 597.4905, 613.77295],)}, {'rec_formula': '\\small\\begin{aligned}{\\phi(L,t)d\\operatorname{log}_{10}L=\\delta(M,t)\\psi(M,t)d M.}\\\\ \\end{aligned}', 'formula_region_id': 9, 'dt_polys': ([195.446, 425.10272, 567.61505, 452.4903],)}, {'rec_formula': '\\log_{10}M=a_{\\bullet}+b_{\\bullet}\\log_{10}X.', 'formula_region_id': 10, 'dt_polys': ([853.006, 908.8241, 1132.3086, 933.7346],)}, {'rec_formula': 't_{E}\\,=\\,\\sigma_{T}c/4\\pi G m_{v}\\,=\\,4.5\\times10^{8}\\mathrm{yr}', 'formula_region_id': 11, 'dt_polys': ([165.8695, 129.74162, 512.8529, 156.56209],)}, {'rec_formula': '\\dot{M}\\:=\\:(1\\:-\\:\\epsilon_{r})\\dot{M}_{\\mathrm{acc}}', 'formula_region_id': 12, 'dt_polys': ([95.99875, 236.36539, 296.8511, 266.53656],)}, {'rec_formula': 'M_{*}=L_{*}t_{E}/\\breve{\\lambda}c^{2}', 'formula_region_id': 13, 'dt_polys': ([94.911865, 1319.9253, 263.36142, 1345.916],)}, {'rec_formula': 'a_{\\bullet}\\,=\\,8.32\\pm0.05', 'formula_region_id': 14, 'dt_polys': ([1089.9109, 1598.5446, 1277.5623, 1622.1991],)}, {'rec_formula': 'b_{\\bullet}=5.64\\,\\dot{\\pm\\,0.32}', 'formula_region_id': 15, 'dt_polys': ([694.6742, 1611.7349, 861.1708, 1635.6787],)}, {'rec_formula': '\\phi(L,t)d\\operatorname{log}_{10}L', 'formula_region_id': 16, 'dt_polys': ([365.27258, 268.35327, 515.08936, 296.99475],)}, {'rec_formula': '\\epsilon_{0}=0.38', 'formula_region_id': 17, 'dt_polys': ([917.18024, 1618.9021, 1009.52045, 1640.4705],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle=', 'formula_region_id': 18, 'dt_polys': ([538.54333, 479.8123, 662.3668, 508.62253],)}, {'rec_formula': '\\delta(M,t)\\dot{M}(M,t)', 'formula_region_id': 19, 'dt_polys': ([99.5916, 508.4211, 253.29228, 535.67163],)}, {'rec_formula': 'M\\mathrm{~-~}\\sigma', 'formula_region_id': 20, 'dt_polys': ([1116.627, 1572.7815, 1191.6616, 1594.5166],)}, {'rec_formula': '\\epsilon_{r}\\dot{M}_{\\mathrm{acc}}', 'formula_region_id': 21, 'dt_polys': ([244.82803, 162.53033, 313.66757, 187.39536],)}, {'rec_formula': '\\delta(M,t)', 'formula_region_id': 22, 'dt_polys': ([255.87213, 323.67505, 326.8396, 349.7248],)}, {'rec_formula': 'X\\:=\\:\\sigma/200\\mathrm{km}\\:\\:\\mathrm{s}^{-1}', 'formula_region_id': 23, 'dt_polys': ([695.4659, 1561.6521, 900.0931, 1585.8818],)}, {'rec_formula': '\\phi(L,t)', 'formula_region_id': 24, 'dt_polys': ([175.38367, 350.68616, 242.63516, 376.44427],)}, {'rec_formula': '\\gamma_{2}', 'formula_region_id': 25, 'dt_polys': ([787.9116, 349.50732, 812.71045, 370.09338],)}, {'rec_formula': 'L_{*}.', 'formula_region_id': 26, 'dt_polys': ([1262.5737, 314.87128, 1296.2644, 338.0655],)}, {'rec_formula': '\\psi_{0}', 'formula_region_id': 27, 'dt_polys': ([774.1763, 595.4717, 801.0121, 618.29297],)}, {'rec_formula': 'z,\\ \\psi(M,z)', 'formula_region_id': 28, 'dt_polys': ([848.8351, 619.38025, 959.8961, 646.0126],)}]}}
```

The explanation of the running result parameters can refer to the result interpretation in [2.2 Integration via Python Script](#22-integration-via-python-script).

The visualization results are saved under `save_path`, where the visualization result of formula recognition is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/04.png" style="width: 70%"/>

<b> If you need to visualize the formula recognition pipeline, you need to run the following command to install the LaTeX rendering environment. Currently, visualization of the formula recognition pipeline only supports the Ubuntu environment, and other environments are not supported. For complex formulas, the LaTeX result may contain some advanced representations that may not be successfully displayed in environments such as Markdown:</b>

```bash
sudo apt-get update
sudo apt-get install texlive texlive-latex-base texlive-xetex latex-cjk-all texlive-latex-extra -y
```

<b>Note</b>: Due to the need to render each formula image during the formula recognition visualization process, the process takes a long time. Please be patient.

### 2.2 Python Script Integration

Using the command line is a quick way to experience and check the results. Generally, in a project, you often need to integrate it through code. You can perform quick inference with just a few lines of code. The inference code is as follows:

```python
from paddleocr import FormulaRecognitionPipeline

pipeline = FormulaRecognitionPipeline()
# ocr = FormulaRecognitionPipeline(use_doc_orientation_classify=True) # Specify whether to use the document orientation classification model with use_doc_orientation_classify.
# ocr = FormulaRecognitionPipeline(use_doc_unwarping=True) # Specify whether to use the text image unwarping module with use_doc_unwarping.
# ocr = FormulaRecognitionPipeline(device="gpu") # Specify the use of GPU for model inference with device.
output = pipeline.predict("./general_formula_recognition_001.png")
for res in output:
    res.print() ##  Print the structured output of the prediction
    res.save_to_json(save_path="output") ## Save the structured JSON result of the current image
```

In the above Python script, the following steps are executed:

（1）Instantiate the formula recognition pipeline object through `create_pipeline()`, with specific parameters as follows:

<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>doc_orientation_classify_model_name</code></td>
<td>The name of the document orientation classification model. If set to <code>None</code>, the default model in pipeline will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_model_dir</code></td>
<td>The directory path of the document orientation classification model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_orientation_classify_batch_size</code></td>
<td>The batch size of the document orientation classification model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_name</code></td>
<td>The name of the text image unwarping model. If set to <code>None</code>, the default model in pipeline will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_model_dir</code></td>
<td>The directory path of the  text image unwarping model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>doc_unwarping_batch_size</code></td>
<td>The batch size of the text image unwarping model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_name</code></td>
<td>The name of the layout detection model. If set to <code>None</code>, the default model in pipeline will be used. </td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_model_dir</code></td>
<td>The directory path of the  layout detection model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>Threshold for layout detection, used to filter out predictions with low confidence.
<ul>
<li><b>float</b>， such as 0.2, indicates filtering out all bounding boxes with a confidence score less than 0.2.</li>
<li><b>Dictionary</b>, with <b>int</b> keys representing <code>cls_id</code> and <b>float</b> values as thresholds. For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> indicates applying a threshold of 0.45 for class ID 0, 0.48 for class ID 2, and 0.4 for class ID 7</li>
<li><b>None</b>, If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS (Non-Maximum Suppression) post-processing for layout region detection to filter out overlapping boxes. If set to <code>None</code>, the default configuration of the official model will be used.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>The scaling factor for the side length of the detection boxes in layout region detection.
<ul>
<li><b>float</b>: A positive float number, e.g., 1.1, indicating that the center of the bounding box remains unchanged while the width and height are both scaled up by a factor of 1.1</li>
<li><b>List</b>: e.g., [1.2, 1.5], indicating that the center of the bounding box remains unchanged while the width is scaled up by a factor of 1.2 and the height by a factor of 1.5</li>
<li><b>None</b>: If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>float|list</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The merging mode for the detection boxes output by the model in layout region detection.
<ul>
<li><b>large</b>: When set to "large", only the largest outer bounding box will be retained for overlapping bounding boxes, and the inner overlapping boxes will be removed.</li>
<li><b>small</b>: When set to "small", only the smallest inner bounding boxes will be retained for overlapping bounding boxes, and the outer overlapping boxes will be removed.</li>
<li><b>union</b>: No filtering of bounding boxes will be performed, and both inner and outer boxes will be retained.</li>
<li><b>None</b>: If not specified, the default PaddleX official model configuration will be used</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_detection_batch_size</code></td>
<td>The batch size for the layout region detection model. If set to <code>None</code>, the default batch size will be set to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>Whether to load the document orientation classification module. If set to <code>None</code>, the parameter will default to the value initialized in the pipeline, which is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to load the text image unwarping module. If set to <code>None</code>, the parameter will default to the value initialized in the pipeline, which is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>Whether to load the layout detection module. If set to <code>None</code>, the parameter will default to the value initialized in the pipeline, which is <code>True</code>.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_name</code></td>
<td>The name of the formula recognition model. If set to <code>None</code>, the default model from the pipeline will be used.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_model_dir</code></td>
<td>The directory path of the formula recognition model. If set to <code>None</code>, the official model will be downloaded.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>formula_recognition_batch_size</code></td>
<td>The batch size for the formula recognition model. If set to  <code>None</code>, the batch size will default to <code>1</code>.</td>
<td><code>int</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>The device used for inference. You can specify a particular card number.
<ul>
<li><b>CPU</b>: e.g., <code>cpu</code> indicates using CPU for inference;</li>
<li><b>GPU</b>: e.g., <code>gpu:0</code> indicates using the 1st GPU for inference;</li>
<li><b>NPU</b>: e.g., <code>npu:0</code> indicates using the 1st NPU for inference;</li>
<li><b>XPU</b>: e.g., <code>xpu:0</code> indicates using the 1st XPU for inference;</li>
<li><b>MLU</b>: e.g., <code>mlu:0</code> indicates using the 1st MLU for inference;</li>
<li><b>DCU</b>: e.g., <code>dcu:0</code> indicates using the 1st DCU for inference;</li>
<li><b>None</b>: If set to <code>None</code>, the default value initialized by the pipeline will be used. During initialization, the local GPU 0 will be prioritized; if unavailable, the CPU will be used.</li>
</ul>
</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable the high-performance inference plugin.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to use TensorRT for inference acceleration. </td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>min_subgraph_size</code></td>
<td>Minimum subgraph size for optimizing the computation of model subgraphs. </td>
<td><code>int</code></td>
<td><code>3</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Compute precision, such as FP32 or FP16.</td>
<td><code>str</code></td>
<td><code>fp32</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>Whether to enable the MKL-DNN acceleration library. If set to <code>None</code>, it will be enabled by default.
</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>The number of threads to use when performing inference on the CPU.</td>
<td><code>int</code></td>
<td><code>8</code></td>
</tr>
</tbody>
</table>

（2）
Call the `predict()` method of the formula recognition pipeline object to perform inference prediction. This method will return a list of results.

Additionally, the pipeline also provides the `predict_iter()` method. Both methods are completely consistent in terms of parameter acceptance and result return. The difference is that `predict_iter()` returns a `generator`, which allows for step-by-step processing and retrieval of prediction results. This is suitable for handling large datasets or scenarios where memory saving is desired. You can choose to use either of these methods based on your actual needs.

Here are the parameters of the `predict()` method and their descriptions:
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types, required.
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., network URL of image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png">Example</a>; <b>Local directory</b>, the directory should contain images to be predicted, e.g., local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with a specific file path)</li>
<li><b>List</b>: Elements of the list must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td><code>None</code></td>
<tr>
<td><code>device</code></td>
<td>The parameters are the same as those used during instantiation.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_layout_detection</code></td>
<td>
Whether to use the layout detection module during inference. </td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_orientation_classify</code></td>
<td>
Whether to use the document orientation classification module during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>use_doc_unwarping</code></td>
<td>Whether to use the text image unwarping module during inference.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_threshold</code></td>
<td>The parameters are the same as those used during instantiation.</td>
<td><code>float|dict</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>The parameters are the same as those used during instantiation.</td>
<td><code>bool</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>The parameters are the same as those used during instantiation.</td>
<td><code>float|list</code></td>
<td><code>None</code></td>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>The parameters are the same as those used during instantiation.</td>
<td><code>string</code></td>
<td><code>None</code></td>
</tr>
</tr></tr></tbody>
</table>

（3）Process the prediction results, where the prediction result for each sample corresponds to a Result object, and supports operations such as printing, saving as an image, and saving as a JSON file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print results to terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Whether to format the output content using <code>JSON</code> indentation</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save results as a JSON file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file. If it is a directory, the saved file will be named the same as the input file type</td>
<td>无</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to beautify the output <code>JSON</code> data, making it more readable. Effective only when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non-<code>ASCII</code> characters to <code>Unicode</code>. When set to <code>True</code>, all non-<code>ASCII</code> characters will be escaped; <code>False</code> retains the original characters. Effective only when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save results as an image file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>Path to save the file, supports directory or file path</td>
<td>无</td>
</tr>
</table>

- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:

    - `input_path`: `(str)` The input path of the image to be predicted.

    - `page_index`: `(Union[int, None])` If the input is a PDF file, this indicates the current page number of the PDF. Otherwise, it is `None`

    - `model_settings`: `(Dict[str, bool])` The model parameters required for the pipeline configuration.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-pipeline.
        - `use_layout_detection`: `(bool)` Controls whether to enable the layout area detection module.

    - `doc_preprocessor_res`: `(Dict[str, Union[str, Dict[str, bool], int]])` The output result of the document preprocessing sub-pipeline. It exists only when `use_doc_preprocessor=True`.
        - `input_path`: `(Union[str, None])` The image path accepted by the image preprocessing sub-pipeline. When the input is a `numpy.ndarray`, it is saved as `None`.
        - `model_settings`: `(Dict)` The model configuration parameters of the preprocessing sub-pipeline.
            - `use_doc_orientation_classify`: `(bool)` Controls whether to enable document orientation classification.
            - `use_doc_unwarping`: `(bool)` Controls whether to enable document distortion correction.
        - `angle`: `(int)` The prediction result of document orientation classification. When enabled, it takes values from [0,1,2,3], corresponding to [0°,90°,180°,270°]; when disabled, it is -1.
    - `layout_det_res`: `(Dict[str, List[Dict]])` The output result of the layout area detection module. It exists only when `use_layout_detection=True`.
        - `input_path`: `(Union[str, None])` The image path accepted by the layout area detection module. When the input is a `numpy.ndarray`, it is saved as `None`.
        - `boxes`: `(List[Dict[int, str, float, List[float]]])` A list of layout area detection prediction results.
            - `cls_id`: `(int)` The class ID predicted by layout area detection.
            - `label`: `(str)` The class label predicted by layout area detection.
            - `score`: `(float)` The confidence score of the predicted class.
            - `coordinate`: `(List[float])` The bounding box coordinates predicted by layout area detection, in the format [x_min, y_min, x_max, y_max], where (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner.
    - `formula_res_list`: `(List[Dict[str, int, List[float]]])` A list of formula recognition prediction results.
        - `rec_formula`: `(str)` The LaTeX source code predicted by formula recognition.
        - `formula_region_id`: `(int)` The ID number predicted by formula recognition.
        - `dt_polys`: `(List[float])` The bounding box coordinates predicted by formula recognition, in the format [x_min, y_min, x_max, y_max], where (x_min, y_min) is the top-left corner and (x_max, y_max) is the bottom-right corner.

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since JSON files do not support saving numpy arrays, `numpy.array` types will be converted to list format.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_formula_res_img.{your_img_extension}`. If a file is specified, it will be saved directly to that file. (The pipeline usually contains many result images, so it is not recommended to specify a specific file path directly, otherwise multiple images will be overwritten and only the last one will be retained.)

* In addition, you can also obtain the visualization image with results and the prediction results through attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction results in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="2"><code>img</code></td>
<td rowspan="2">Get the visualization image in <code>dict</code> format</td>
</tr>
</table>



- The prediction result obtained from the `json` attribute is of the dict type, and its content is consistent with what is saved by calling the `save_to_json()` method.
- The prediction result returned by the  `img` attribute is a dictionary-type data. The keys are  `preprocessed_img`、 `layout_det_res` and `formula_res_img`, and the corresponding values are three `Image.Image` objects: the first one is used to display the visualization of image preprocessing, the second one is for displaying the visualization of layout region detection, and the third one is for displaying the visualization of formula recognition. If the image preprocessing submodule is not used, the dictionary will not contain the `preprocessed_img` key. Similarly, if the layout region detection submodule is not used, the dictionary will not contain the `layout_det_res` key.

## 3. Development Integration/Deployment

If the formula recognition pipeline meets your requirements for inference speed and accuracy, you can proceed directly with development integration/deployment.

If you need to integrate the formula recognition pipeline into your Python project, you can refer to the example code in [ 2.2 Integration via Python Script](#22-integration-via-python-script).

In addition, PaddleOCR also provides two other deployment methods, which are detailed as follows:

🚀 High-Performance Inference: In real-world production environments, many applications have stringent standards for performance metrics of deployment strategies, particularly regarding response speed, to ensure efficient system operation and a smooth user experience. To address this, PaddleOCR offers high-performance inference capabilities designed to deeply optimize the performance of model inference and pre/post-processing, significantly accelerating the end-to-end process. For detailed information on the high-performance inference process, please refer to the [High-Performance Inference Guide](../deployment/high_performance_inference.en.md).


☁️ Service-Based Deployment：
Service-Based Deployment is a common deployment form in real-world production environments. By encapsulating inference capabilities as a service, clients can access these services via network requests to obtain inference results. For detailed instructions on Service-Based Deployment in production lines, please refer to the [Service-Based Deployment Guide](../deployment/serving.md).

Below are the API references for basic service-based deployment and multi-language service invocation examples:


<details><summary>API Reference</summary>
<p>For the main operations provided by the service:</p>
<ul>
<li>The HTTP request method is POST.</li>
<li>Both the request body and response body are JSON data (JSON objects).</li>
<li>When the request is processed successfully, the response status code is <code>200</code>, and the attributes of the response body are as follows:</li>
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
<td>Error message. Fixed as <code>"Success"</code>.</td>
</tr>
<tr>
<td><code>result</code></td>
<td><code>object</code></td>
<td>The result of the operation.</td>
</tr>
</tbody>
</table>
<ul>
<li>When the request is not processed successfully, the attributes of the response body are as follows:</li>
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
<td>Error message.</td>
</tr>
</tbody>
</table>
<p>The main operations provided by the service are as follows:</p>
<ul>
<li><b><code>infer</code></b></li>
</ul>
<p>Obtain the formula recognition results from images.</p>
<p><code>POST /formula-recognition</code></p>
<ul>
<li>The attributes of the request body are as follows:</li>
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
<td>The URL of an image or PDF file accessible by the server, or the Base64-encoded content of the file. By default, for PDF files exceeding 10 pages, only the first 10 pages will be processed.<br />
To remove the page limit, please add the following configuration to the pipeline configuration file:
<pre><code>Serving:
  extra:
    max_num_input_imgs: null
</code></pre>
</td>
<td>Yes</td>
</tr>
<tr>
<td><code>fileType</code></td>
<td><code>integer</code> | <code>null</code></td>
<td>The type of the file. <code>0</code> for PDF files, <code>1</code> for image files. If this attribute is missing, the file type will be inferred from the URL.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocOrientationClassify</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_orientation_classify</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useDocUnwarping</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_doc_unwarping</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>useLayoutDetection</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>use_layout_detection</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutThreshold</code></td>
<td><code>number</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_threshold</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutNms</code></td>
<td><code>boolean</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_nms</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutUnclipRatio</code></td>
<td><code>number</code> | <code>array</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_unclip_ratio</code> parameter of the pipeline object's <code>predict</code> method.</td>
<td>No</td>
</tr>
<tr>
<td><code>layoutMergeBboxesMode</code></td>
<td><code>string</code> | <code>null</code></td>
<td>Please refer to the description of the <code>layout_merge_bboxes_mode</code> parameter of the pipeline object's <code>predict</code> method.</td>
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
<th>名称</th>
<th>类型</th>
<th>含义</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>formulaRecResults</code></td>
<td><code>object</code></td>
<td>The formula recognition results. The array length is 1 (for image input) or the actual number of document pages processed (for PDF input). For PDF input, each element in the array represents the result of each page actually processed in the PDF file.</td>
</tr>
<tr>
<td><code>dataInfo</code></td>
<td><code>object</code></td>
<td>Information about the input data.</td>
</tr>
</tbody>
</table>
<p>Each element in <code>formulaRecResults</code> is an <code>object</code> with the following attributes:</p>
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
<td><code>prunedResult</code></td>
<td><code>object</code></td>
<td>A simplified version of the <code>res</code> field in the JSON representation of the result generated by the pipeline object's <code>predict</code> method, excluding the <code>input_path</code> and the <code>page_index</code> fields.</td>
</tr>
</tr>
<tr>
<td><code>outputImages</code></td>
<td><code>object</code> | <code>null</code></td>
<td>See the description of the <code>img</code> attribute of the result of the pipeline prediction. The images are in JPEG format and are Base64-encoded.</td>
</tr>
<tr>
<td><code>inputImage</code> | <code>null</code></td>
<td><code>string</code></td>
<td>The input image. The image is in JPEG format and is Base64-encoded.</td>
</tr>
</tbody>
</table>
</details>
<details><summary>Multi-language Service Invocation Example</summary>
<details>
<summary>Python</summary>

<pre><code class="language-python">import base64
import requests

API_URL = "http://localhost:8080/formula-recognition"
file_path = "./demo.jpg"

with open(file_path, "rb") as file:
    file_bytes = file.read()
    file_data = base64.b64encode(file_bytes).decode("ascii")

payload = {"file": file_data, "fileType": 1}

response = requests.post(API_URL, json=payload)

assert response.status_code == 200
result = response.json()["result"]
for i, res in enumerate(result["formulaRecResults"]):
    print(res["prunedResult"])
    for img_name, img in res["outputImages"].items():
        img_path = f"{img_name}_{i}.jpg"
        with open(img_path, "wb") as f:
            f.write(base64.b64decode(img))
        print(f"Output image saved at {img_path}")
</code></pre></details>
</details>
<br/>


## 4. Custom Development

If the default model weights provided by the formula recognition pipeline do not meet your requirements in terms of accuracy or speed, you can try to <b>fine-tune</b> the existing models using <b>your own domain-specific or application-specific data</b> to improve the recognition performance of the formula recognition pipeline in your scenario.

Since the formula recognition pipeline consists of several modules, if the pipeline's performance is not satisfactory, the issue may arise from any one of these modules. You can analyze the poorly recognized images to determine which module is problematic and refer to the corresponding fine-tuning tutorial links in the table below for model fine-tuning.

<table>
<thead>
<tr>
<th>Scenario</th>
<th>Fine-Tuning Module</th>
<th>Reference Link</th>
</tr>
</thead>
<tbody>
<tr>
<td>Formulas are missing</td>
<td>Layout Detection Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html">Link</a></td>
</tr>
<tr>
<td>Formula content is inaccurate</td>
<td>Formula Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html">Link</a></td>
</tr>
<tr>
<td>Whole-image rotation correction is inaccurate</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html">Link</a></td>
</tr>
<tr>
<td>Image distortion correction is inaccurate</td>
<td>Text Image Correction Module</td>
<td>Fine-tuning not supported</td>
</tr>
</tbody>
</table>
