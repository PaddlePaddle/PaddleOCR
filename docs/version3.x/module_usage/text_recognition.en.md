---
comments: true
---

# Text Recognition Module User Guide 

## 1.Overview

The text recognition module is a core component of the OCR (Optical Character Recognition) system. It is responsible for extracting textual information from the detected text regions in images. The performance of this module directly affects the accuracy and efficiency of the entire OCR pipeline. Typically, it receives bounding boxes from the text detection module as input and utilizes complex image processing and deep learning algorithms to convert image-based text into editable and searchable electronic text. Accurate recognition results are crucial for subsequent tasks such as information extraction and data mining.

## 2.Supported Models List

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Pretrained Model</a></td>
<td>81.53</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc is based on PP-OCRv4_server_rec and is trained on a mixed dataset comprising more Chinese document data and standard PP-OCR training data. It enhances the recognition capabilities for traditional Chinese,Japanese, and special characters, supporting over 15,000 characters. In addition to improved document-specific recognition, it also boosts general text recognition performance.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>78.74</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>A lightweight text recognition model from PP-OCRv4 with high inference efficiency, suitable for deployment on various hardware devices including edge devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec </td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Pretrained Model</a></td>
td>
<td>80.61 </td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>A server-side model of PP-OCRv4 with high inference accuracy, deployable across various types of servers.</td>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>An ultra-lightweight English text recognition model trained based on the PP-OCRv4 architecture, supporting English and numeric character recognition.</td>
</tr>
</table>

> ❗  The models listed above are the <b>4 core models</b>primarily supported by the text recognition module. In total,this module supports<b>18 full models</b>, including multiple multilingual text recognition models. The complete list is as follows:
<details><summary> 👉 Full Model List</summary>

<b>Chinese Text Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_doc_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_doc_pretrained.pdparams">Pretrained Model</a></td>
<td>81.53</td>
<td>6.65 / 2.38</td>
<td>32.92 / 32.92</td>
<td>74.7 M</td>
<td>PP-OCRv4_server_rec_doc is based on PP-OCRv4_server_rec and trained on a mixture of additional Chinese document data and PP-OCR training datasets. It enhances recognition of traditional Chinese, Japanese, and special characters, supporting over 15,000 characters. Besides improving document-related recognition, it also boosts general text recognition capabilities.</td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>78.74</td>
<td>4.82 / 1.20</td>
<td>16.74 / 4.64</td>
<td>10.6 M</td>
<td>A lightweight version of PP-OCRv4 recognition model with high inference efficiency, suitable for deployment on various edge and embedded devices.</td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>80.61</td>
<td>6.58 / 2.43</td>
<td>33.17 / 33.17</td>
<td>71.2 M</td>
<td>The server-side version of PP-OCRv4 with high recognition accuracy, suitable for deployment across various server environments.</td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>72.96</td>
<td>5.87 / 1.19</td>
<td>9.07 / 4.28</td>
<td>9.2 M</td>
<td>A lightweight version of the PP-OCRv3 recognition model with high inference efficiency, deployable on a variety of edge and embedded devices.</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>68.81</td>
<td>8.08 / 2.74</td>
<td>50.17 / 42.50</td>
<td>73.9 M</td>
<td rowspan="1">
SVTRv2 is a server-side text recognition model developed by the OpenOCR team of the Visual and Learning Lab (FVL) at Fudan University. It won first prize in Task 1 of the PaddleOCR Algorithm Model Challenge. On the A leaderboard, its end-to-end recognition accuracy is 6% higher than PP-OCRv4.
</td>
</tr>
</table>

<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>CPU Inference Time (ms)<br/>[Standard / High Performance]</th>
<th>Model Size (MB)</th>
<th>Description</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>65.07</td>
<td>5.93 / 1.62</td>
<td>20.73 / 7.32</td>
<td>22.1 M</td>
<td rowspan="1">RepSVTR is a mobile-end text recognition model based on SVTRv2. It also won first prize in Task 1 of the PaddleOCR Algorithm Model Challenge. On the B leaderboard, it achieves 2.5% higher end-to-end accuracy compared to PP-OCRv4, with similar inference speed.</td>
</tr>
</table>

* <b>English Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv4_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv4_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.39</td>
<td>4.81 / 0.75</td>
<td>16.10 / 5.31</td>
<td>6.8 M</td>
<td>Ultra-lightweight English recognition model based on PP-OCRv4, supports English and digit recognition</td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>70.69</td>
<td>5.44 / 0.75</td>
<td>8.65 / 5.57</td>
<td>7.8 M</td>
<td>Ultra-lightweight English recognition model based on PP-OCRv3, supports English and digit recognition</td>
</tr>
</table>

<b>Multilingual Recognition Models</b>
<table>
<tr>
<th>Model</th><th>Download Link</th>
<th>Avg Accuracy (%)</th>
<th>GPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>CPU Inference Time (ms)<br/>[Standard Mode / High-Performance Mode]</th>
<th>Model Size (M)</th>
<th>Description</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>60.21</td>
<td>5.40 / 0.97</td>
<td>9.11 / 4.05</td>
<td>8.6 M</td>
<td>Ultra-lightweight Korean recognition model based on PP-OCRv3, supports Korean and digit recognition</td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/japan_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/japan_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>45.69</td>
<td>5.70 / 1.02</td>
<td>8.48 / 4.07</td>
<td>8.8 M</td>
<td>Ultra-lightweight Japanese recognition model based on PP-OCRv3, supports Japanese and digit recognition</td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/chinese_cht_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/chinese_cht_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>82.06</td>
<td>5.90 / 1.28</td>
<td>9.28 / 4.34</td>
<td>9.7 M</td>
<td>Ultra-lightweight Traditional Chinese recognition model based on PP-OCRv3, supports Traditional Chinese and digit recognition</td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/te_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>95.88</td>
<td>5.42 / 0.82</td>
<td>8.10 / 6.91</td>
<td>7.8 M</td>
<td>Ultra-lightweight Telugu recognition model based on PP-OCRv3, supports Telugu and digit recognition</td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ka_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ka_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>96.96</td>
<td>5.25 / 0.79</td>
<td>9.09 / 3.86</td>
<td>8.0 M</td>
<td>Ultra-lightweight Kannada recognition model based on PP-OCRv3, supports Kannada and digit recognition</td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ta_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>76.83</td>
<td>5.23 / 0.75</td>
<td>10.13 / 4.30</td>
<td>8.0 M</td>
<td>Ultra-lightweight Tamil recognition model based on PP-OCRv3, supports Tamil and digit recognition</td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>76.93</td>
<td>5.20 / 0.79</td>
<td>8.83 / 7.15</td>
<td>7.8 M</td>
<td>Ultra-lightweight Latin recognition model based on PP-OCRv3, supports Latin and digit recognition</td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/arabic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>73.55</td>
<td>5.35 / 0.79</td>
<td>8.80 / 4.56</td>
<td>7.8 M</td>
<td>Ultra-lightweight Arabic recognition model based on PP-OCRv3, supports Arabic and digit recognition</td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/cyrillic_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>94.28</td>
<td>5.23 / 0.76</td>
<td>8.89 / 3.88</td>
<td>7.9 M</td>
<td>Ultra-lightweight Cyrillic recognition model based on PP-OCRv3, supports Cyrillic and digit recognition</td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td><td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/devanagari_PP-OCRv3_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv3_mobile_rec_pretrained.pdparams">Pretrained Model</a></td>
<td>96.44</td>
<td>5.22 / 0.79</td>
<td>8.56 / 4.06</td>
<td>7.9 M</td>
<td>Ultra-lightweight Devanagari recognition model based on PP-OCRv3, supports Devanagari and digit recognition</td>
</tr>
</table>

<strong>Test Environment Description:</strong>
<ul>
  <li><b>Performance Test Environment</b>
    <ul>
      <li><strong>Test Datasets:</strong>
        <ul>
          <li>Chinese recognition model: PaddleOCR internal Chinese dataset, covering street view, web images, documents, and handwriting; 11k images in text recognition.</li>
          <li>ch_SVTRv2_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Challenge - Task 1: End-to-End OCR Recognition (A-Set)</a></li>
          <li>ch_RepSVTR_rec: <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR Algorithm Challenge - Task 1: End-to-End OCR Recognition (B-Set)</a></li>
          <li>English recognition models: PaddleX internal English dataset.</li>
          <li>Multilingual models: PaddleX internal multilingual dataset.</li>
        </ul>
      </li>
      <li><strong>Hardware Configuration:</strong>
        <ul>
          <li>GPU: NVIDIA Tesla T4</li>
          <li>CPU: Intel Xeon Gold 6271C @ 2.60GHz</li>
          <li>Other: Ubuntu 20.04 / cuDNN 8.6 / TensorRT 8.5.2.2</li>
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
<th>GPU Configuration</th>
<th>CPU Configuration</th>
<th>Acceleration Strategy</th>
</tr>
</thead>
<tbody>
<tr>
<td>Standard Mode</td>
<td>FP32 precision / No TRT acceleration</td>
<td>FP32 precision / 8 threads</td>
<td>PaddleInference</td>
</tr>
<tr>
<td>High-Performance Mode</td>
<td>Best precision type and strategy chosen in advance</td>
<td>FP32 precision / 8 threads</td>
<td>Best backend selected in advance (Paddle/OpenVINO/TRT, etc.)</td>
</tr>
</tbody>
</table>
</details>

## 3. Quick Start

> ❗ Before getting started, please install the PaddleOCR wheel package. For details, please refer to the  [Installation Guide](../ppocr/installation.md)。

You can quickly try it out with a single command:

```bash
paddleocr text_recognition -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png
```
You can also integrate the model inference of the text recognition module into your project. Before running the following code, please download the sample image (https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_rec_001.png)to your local machine.

```python
from paddleocr import TextRecognition
model = TextRecognition(model_name="PP-OCRv4_mobile_rec")
output = model.predict(input="general_ocr_rec_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After execution, the result obtained is:

```bash
{'res': {'input_path': 'general_ocr_rec_001.png', 'page_index': None, 'rec_text': '绿洲仕格维花园公寓', 'rec_score': 0.9875142574310303}}
```

The meanings of the parameters in the result are as follows:

- 'input_path: indicates the path to the input image file for prediction.
- page_index: if the input is a PDF file, this represents which page it is in the PDF; otherwise, it is None.
- rec_text: represents the predicted text from the input image.
- `rec_score` represents the confidence score of the predicted text from the input image.
The visualized image is shown below: