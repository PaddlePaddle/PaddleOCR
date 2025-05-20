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
                  <li>Document Image Orientation Classification Module: A self-built dataset using PaddleOCR, covering multiple scenarios such as ID cards and documents, containing 1000 images.</li>
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

Before using the formula recognition pipeline locally, please ensure that you have completed the wheel package installation according to the [installation guide](../installation.en.md). Once installed, you can experience it locally via the command line or integrate it with Python.


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
<td><code>input</code></td>
<td>Data to be predicted, supporting multiple input types, required.
<ul>
<li><b>Python Var</b>: Image data represented by <code>numpy.ndarray</code></li>
<li><b>str</b>: Local path of image or PDF file, e.g., <code>/root/data/img.jpg</code>; <b>URL link</b>, e.g., network URL of image or PDF file: <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/demo_image/pipelines/general_formula_recognition_001.png">Example</a>; <b>Local directory</b>, the directory should contain images to be predicted, e.g., local path: <code>/root/data/</code> (currently does not support prediction of PDF files in directories; PDF files must be specified with a specific file path)</li>
<li><b>List</b>: Elements of the list must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>[\"/root/data/img1.jpg\", \"/root/data/img2.jpg\"]</code>, <code>[\"/root/data1\", \"/root/data2\"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>save_path</code></td>
<td>
Specify the path to save the inference results file. If set to <code>None</code>, the inference results will not be saved locally.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
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
<tr>
<td><code>paddlex_config</code></td>
<td>Path to PaddleX pipeline configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>
</details>
<br />


The results of running the default configuration of the formula recognition pipeline will be printed to the terminal as follows:

```bash
{'res': {'input_path': './general_formula_recognition_001.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_layout_detection': True}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': True, 'use_doc_unwarping': True}, 'angle': 0}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 2, 'label': 'text', 'score': 0.9855189323425293, 'coordinate': [90.56131, 1086.7773, 658.8992, 1553.2681]}, {'cls_id': 2, 'label': 'text', 'score': 0.9814704060554504, 'coordinate': [93.04651, 127.988556, 664.8587, 396.60892]}, {'cls_id': 2, 'label': 'text', 'score': 0.9767388105392456, 'coordinate': [698.4391, 591.0454, 1293.3676, 748.28345]}, {'cls_id': 2, 'label': 'text', 'score': 0.9712911248207092, 'coordinate': [701.4946, 286.61566, 1299.0099, 391.87457]}, {'cls_id': 2, 'label': 'text', 'score': 0.9709068536758423, 'coordinate': [697.0126, 751.93604, 1290.2236, 883.64453]}, {'cls_id': 2, 'label': 'text', 'score': 0.9689271450042725, 'coordinate': [704.01196, 79.645935, 1304.7493, 187.96674]}, {'cls_id': 2, 'label': 'text', 'score': 0.9683637619018555, 'coordinate': [93.063385, 799.3567, 660.6935, 902.0344]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9660536646842957, 'coordinate': [728.5045, 440.9215, 1224.0634, 570.8518]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9616329669952393, 'coordinate': [722.9789, 1333.5085, 1257.1136, 1468.0432]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9610316753387451, 'coordinate': [756.4525, 1211.323, 1188.0428, 1268.2336]}, {'cls_id': 7, 'label': 'formula', 'score': 0.960993230342865, 'coordinate': [777.51355, 207.87927, 1222.8966, 267.33014]}, {'cls_id': 2, 'label': 'text', 'score': 0.9594196677207947, 'coordinate': [697.5154, 957.6764, 1288.6238, 1033.5211]}, {'cls_id': 2, 'label': 'text', 'score': 0.9593432545661926, 'coordinate': [691.333, 1511.8015, 1282.0968, 1642.5906]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9589930176734924, 'coordinate': [153.89856, 924.2046, 601.0946, 1036.9038]}, {'cls_id': 2, 'label': 'text', 'score': 0.9582098722457886, 'coordinate': [87.02347, 1557.2971, 655.9584, 1632.6912]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9579620957374573, 'coordinate': [810.86975, 1057.0771, 1175.101, 1117.6631]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9557801485061646, 'coordinate': [165.26271, 557.8495, 598.1803, 614.35]}, {'cls_id': 7, 'label': 'formula', 'score': 0.953873872756958, 'coordinate': [116.48187, 713.88416, 614.2181, 774.02576]}, {'cls_id': 2, 'label': 'text', 'score': 0.9521227478981018, 'coordinate': [96.6882, 478.32745, 662.573, 536.5877]}, {'cls_id': 2, 'label': 'text', 'score': 0.944242000579834, 'coordinate': [96.12866, 639.1591, 661.7959, 692.4849]}, {'cls_id': 2, 'label': 'text', 'score': 0.9403323531150818, 'coordinate': [695.9436, 1138.6748, 1286.7242, 1188.0049]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9249663949012756, 'coordinate': [852.90137, 908.64386, 1131.1882, 933.81793]}, {'cls_id': 7, 'label': 'formula', 'score': 0.9249223470687866, 'coordinate': [195.28397, 424.81024, 567.697, 451.1291]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.9173304438591003, 'coordinate': [1246.2393, 1079.0535, 1286.3281, 1104.3323]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.9169507026672363, 'coordinate': [1246.9003, 908.6482, 1288.2013, 934.61426]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.915979266166687, 'coordinate': [1247.0374, 1229.1572, 1287.094, 1254.9805]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.9085646867752075, 'coordinate': [1252.864, 492.1079, 1294.6238, 518.47095]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.9017605781555176, 'coordinate': [1242.1719, 1473.6951, 1283.02, 1498.6316]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8999755382537842, 'coordinate': [1269.8164, 220.34933, 1299.8589, 247.01102]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8965252041816711, 'coordinate': [96.00711, 235.49493, 295.43823, 265.60016]}, {'cls_id': 2, 'label': 'text', 'score': 0.8954343199729919, 'coordinate': [696.85693, 1286.2236, 1083.3921, 1310.8643]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8952110409736633, 'coordinate': [166.60979, 129.20242, 511.65692, 156.29672]}, {'cls_id': 2, 'label': 'text', 'score': 0.893648624420166, 'coordinate': [725.64575, 396.18964, 1263.0391, 422.76813]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8922948837280273, 'coordinate': [634.14124, 427.77087, 661.1686, 454.10022]}, {'cls_id': 2, 'label': 'text', 'score': 0.8892256617546082, 'coordinate': [94.483246, 1058.7595, 441.92313, 1082.4875]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8878197073936462, 'coordinate': [630.4175, 939.3015, 657.7135, 965.36426]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8831961154937744, 'coordinate': [630.5835, 1000.95715, 657.4309, 1026.2128]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8767948150634766, 'coordinate': [634.1024, 575.3833, 660.59094, 601.1677]}, {'cls_id': 7, 'label': 'formula', 'score': 0.873543918132782, 'coordinate': [95.29655, 1320.3627, 264.93008, 1345.8473]}, {'cls_id': 17, 'label': 'formula_number', 'score': 0.8702306151390076, 'coordinate': [633.82825, 730.31525, 659.83215, 755.5485]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8387619853019714, 'coordinate': [365.19897, 268.29675, 515.7938, 296.07013]}, {'cls_id': 7, 'label': 'formula', 'score': 0.8314349055290222, 'coordinate': [1090.509, 1599.1382, 1276.6736, 1622.156]}, {'cls_id': 7, 'label': 'formula', 'score': 0.817135751247406, 'coordinate': [246.175, 161.22958, 314.3764, 186.40591]}, {'cls_id': 3, 'label': 'number', 'score': 0.8042846322059631, 'coordinate': [1297.4036, 7.1497707, 1310.5969, 27.737753]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7970448136329651, 'coordinate': [538.45593, 478.09354, 661.8812, 508.50778]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7644855976104736, 'coordinate': [916.51746, 1618.5188, 1009.62537, 1640.8206]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7423419952392578, 'coordinate': [694.8439, 1612.2507, 861.05334, 1635.9768]}, {'cls_id': 7, 'label': 'formula', 'score': 0.7072376608848572, 'coordinate': [99.72007, 508.21167, 254.91953, 535.74744]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6976271867752075, 'coordinate': [696.8011, 1561.4375, 899.79584, 1586.7349]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6707713007926941, 'coordinate': [1117.0862, 1571.9763, 1191.502, 1594.742]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6338322162628174, 'coordinate': [577.33484, 1274.4131, 602.5636, 1296.7021]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6199935674667358, 'coordinate': [175.28284, 349.82376, 241.24683, 376.6708]}, {'cls_id': 7, 'label': 'formula', 'score': 0.612853467464447, 'coordinate': [773.06287, 595.202, 800.43884, 617.3812]}, {'cls_id': 7, 'label': 'formula', 'score': 0.6107096672058105, 'coordinate': [706.6776, 316.87082, 736.69714, 339.9352]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5520269870758057, 'coordinate': [1263.9711, 314.65167, 1292.7728, 337.3896]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5346108675003052, 'coordinate': [1219.2955, 316.599, 1243.9181, 339.71802]}, {'cls_id': 7, 'label': 'formula', 'score': 0.5195119380950928, 'coordinate': [254.65729, 323.6553, 326.57758, 349.53494]}, {'cls_id': 7, 'label': 'formula', 'score': 0.501812219619751, 'coordinate': [255.8518, 1350.6472, 301.74304, 1375.5286]}]}, 'formula_res_list': [{'rec_formula': '\\begin{aligned}{\\psi_{0}(M)-\\psi_{}(M,z)=}&{{}\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\operatorname{E}}^{2}\\operatorname{l n}(10)}\\times}\\\\ {}&{{}\\int_{0}^{z}d z^{\\prime}\\frac{d t}{d z^{\\prime}}\\left.\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\operatorname{E}}},}\\\\ \\end{aligned}', 'formula_region_id': 1, 'dt_polys': ([728.5045, 440.9215, 1224.0634, 570.8518],)}, {'rec_formula': '\\begin{aligned}{p(\\operatorname{l o g}_{10}}&{{}M|\\operatorname{l o g}_{10}\\sigma)=\\frac{1}{\\sqrt{2\\pi}\\epsilon_{0}}}\\\\ {}&{{}\\times\\operatorname{e x p}\\left[-\\frac{1}{2}\\left(\\frac{\\operatorname{l o g}_{10}M-a_{\\bullet}-b_{\\bullet}\\operatorname{l o g}_{10}\\sigma}{\\epsilon_{0}}\\right)^{2}\\right].}\\\\ \\end{aligned}', 'formula_region_id': 2, 'dt_polys': ([722.9789, 1333.5085, 1257.1136, 1468.0432],)}, {'rec_formula': '\\psi_{0}(M)=\\int d\\sigma\\frac{p(\\operatorname{l o g}_{10}M|\\operatorname{l o g}_{10}\\sigma)}{M\\operatorname{l o g}(10)}\\frac{d n}{d\\sigma}(\\sigma),', 'formula_region_id': 3, 'dt_polys': ([756.4525, 1211.323, 1188.0428, 1268.2336],)}, {'rec_formula': '\\phi(L)\\equiv\\frac{d n}{d\\operatorname{l o g}_{10}L}=\\frac{\\phi_{*}}{(L/L_{*})^{\\gamma_{1}}+(L/L_{*})^{\\gamma_{2}}}.', 'formula_region_id': 4, 'dt_polys': ([777.51355, 207.87927, 1222.8966, 267.33014],)}, {'rec_formula': '\\begin{aligned}{\\rho_{\\operatorname{B H}}}&{{}=\\int d M\\psi(M)M}\\\\ {}&{{}=\\frac{1-\\epsilon_{r}}{\\epsilon_{r}c^{2}}\\int_{0}^{\\infty}d z\\frac{d t}{d z}\\int d\\operatorname{l o g}_{10}L\\phi(L,z)L,}\\\\ \\end{aligned}', 'formula_region_id': 5, 'dt_polys': ([153.89856, 924.2046, 601.0946, 1036.9038],)}, {'rec_formula': '\\frac{d n}{d\\sigma}d\\sigma=\\psi_{*}\\left(\\frac{\\sigma}{\\sigma_{*}}\\right)^{\\alpha}\\frac{e^{-(\\sigma/\\sigma_{*})^{\\beta}}}{\\Gamma(\\alpha/\\beta)}\\beta\\frac{d\\sigma}{\\sigma}.', 'formula_region_id': 6, 'dt_polys': ([810.86975, 1057.0771, 1175.101, 1117.6631],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle\\psi(M,t)=\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}c^{2}\\operatorname{l n}(10)}\\phi(L,t)\\frac{d L}{d M}.', 'formula_region_id': 7, 'dt_polys': ([165.26271, 557.8495, 598.1803, 614.35],)}, {'rec_formula': '\\frac{\\partial\\psi}{\\partial t}(M,t)+\\frac{(1-\\epsilon_{r})}{\\epsilon_{r}}\\frac{\\lambda^{2}c^{2}}{t_{\\operatorname{E}}^{2}\\operatorname{l n}(10)}\\left.\\frac{\\partial\\phi}{\\partial L}\\right|_{L=\\lambda M c^{2}/t_{\\operatorname{E}}}=0,', 'formula_region_id': 8, 'dt_polys': ([116.48187, 713.88416, 614.2181, 774.02576],)}, {'rec_formula': '\\operatorname{l o g}_{10}M=a_{\\bullet}+b_{\\bullet}\\operatorname{l o g}_{10}X.', 'formula_region_id': 9, 'dt_polys': ([852.90137, 908.64386, 1131.1882, 933.81793],)}, {'rec_formula': '\\phi(L,t)d\\operatorname{l o g}_{10}L=\\delta(M,t)\\psi(M,t)d M.', 'formula_region_id': 10, 'dt_polys': ([195.28397, 424.81024, 567.697, 451.1291],)}, {'rec_formula': '\\dot{M}\\:=\\:(1\\:-\\:\\epsilon_{r})\\dot{M}_{\\mathrm{a c c}}^{\\mathrm{~\\tiny~\\cdot~}}', 'formula_region_id': 11, 'dt_polys': ([96.00711, 235.49493, 295.43823, 265.60016],)}, {'rec_formula': 't_{E}=\\sigma_{T}c/4\\pi G m_{p}=4.5\\times10^{8}\\mathrm{y r}', 'formula_region_id': 12, 'dt_polys': ([166.60979, 129.20242, 511.65692, 156.29672],)}, {'rec_formula': 'M_{*}=L_{*}t_{E}/\\tilde{\\lambda}c^{2}', 'formula_region_id': 13, 'dt_polys': ([95.29655, 1320.3627, 264.93008, 1345.8473],)}, {'rec_formula': '\\phi(L,t)d\\operatorname{l o g}_{10}L', 'formula_region_id': 14, 'dt_polys': ([365.19897, 268.29675, 515.7938, 296.07013],)}, {'rec_formula': 'a_{\\bullet}=8.32\\pm0.05', 'formula_region_id': 15, 'dt_polys': ([1090.509, 1599.1382, 1276.6736, 1622.156],)}, {'rec_formula': '\\epsilon_{r}\\dot{M}_{\\mathrm{a c c}}', 'formula_region_id': 16, 'dt_polys': ([246.175, 161.22958, 314.3764, 186.40591],)}, {'rec_formula': '\\langle\\dot{M}(M,t)\\rangle=', 'formula_region_id': 17, 'dt_polys': ([538.45593, 478.09354, 661.8812, 508.50778],)}, {'rec_formula': '\\epsilon_{0}=0.38', 'formula_region_id': 18, 'dt_polys': ([916.51746, 1618.5188, 1009.62537, 1640.8206],)}, {'rec_formula': 'b_{\\bullet}=5.64\\dot{\\pm}\\dot{0.32}', 'formula_region_id': 19, 'dt_polys': ([694.8439, 1612.2507, 861.05334, 1635.9768],)}, {'rec_formula': '\\delta(M,t)\\dot{M}(M,t)', 'formula_region_id': 20, 'dt_polys': ([99.72007, 508.21167, 254.91953, 535.74744],)}, {'rec_formula': 'X=\\sigma/200\\mathrm{k m}\\mathrm{~s^{-1}~}', 'formula_region_id': 21, 'dt_polys': ([696.8011, 1561.4375, 899.79584, 1586.7349],)}, {'rec_formula': 'M-\\sigma', 'formula_region_id': 22, 'dt_polys': ([1117.0862, 1571.9763, 1191.502, 1594.742],)}, {'rec_formula': 'L_{*}', 'formula_region_id': 23, 'dt_polys': ([577.33484, 1274.4131, 602.5636, 1296.7021],)}, {'rec_formula': '\\phi(L,t)', 'formula_region_id': 24, 'dt_polys': ([175.28284, 349.82376, 241.24683, 376.6708],)}, {'rec_formula': '\\psi_{0}', 'formula_region_id': 25, 'dt_polys': ([773.06287, 595.202, 800.43884, 617.3812],)}, {'rec_formula': '\\mathrm{A^{\\prime\\prime}}', 'formula_region_id': 26, 'dt_polys': ([706.6776, 316.87082, 736.69714, 339.9352],)}, {'rec_formula': 'L_{*}', 'formula_region_id': 27, 'dt_polys': ([1263.9711, 314.65167, 1292.7728, 337.3896],)}, {'rec_formula': '\\phi_{*}', 'formula_region_id': 28, 'dt_polys': ([1219.2955, 316.599, 1243.9181, 339.71802],)}, {'rec_formula': '\\delta(M,t)', 'formula_region_id': 29, 'dt_polys': ([254.65729, 323.6553, 326.57758, 349.53494],)}, {'rec_formula': '\\phi(L)', 'formula_region_id': 30, 'dt_polys': ([255.8518, 1350.6472, 301.74304, 1375.5286],)}]}}
```

The explanation of the running result parameters can refer to the result interpretation in [ 2.2 Python Script Integration](#22-python-script-integration).

The visualization results are saved under `save_path`, where the visualization result of formula recognition is as follows:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/formula_recognition/04_paddleocr3.png" style="width: 70%"/>

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
    res.save_to_img(save_path="output") ## Save the formula visualization result of the current image.
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
<tr>
<td><code>paddlex_config</code></td>
<td>Path to PaddleX pipeline configuration file.</td>
<td><code>str</code></td>
<td><code>None</code></td>
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
<td></td>
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

（3）Process the prediction results, where the prediction result for each sample corresponds to a Result object, and supports operations such as printing, saving as an image, and saving as a `json` file:

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

If you need to integrate the formula recognition pipeline into your Python project, you can refer to the example code in [ 2.2 Python Script Integration](#22-python-script-integration).

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
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Formula content is inaccurate</td>
<td>Formula Recognition Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleOCR/main/en/version3.x/module_usage/formula_recognition.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Whole-image rotation correction is inaccurate</td>
<td>Document Image Orientation Classification Module</td>
<td><a href="https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.html#iv-custom-development">Link</a></td>
</tr>
<tr>
<td>Image distortion correction is inaccurate</td>
<td>Text Image Correction Module</td>
<td>Fine-tuning not supported</td>
</tr>
</tbody>
</table>
