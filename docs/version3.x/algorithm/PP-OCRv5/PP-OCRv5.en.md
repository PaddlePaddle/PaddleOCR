# Introduction to PP-OCRv5

**PP-OCRv5** is the new generation text recognition solution of PP-OCR, focusing on multi-scenario and multi-text type recognition. In terms of text types, PP-OCRv5 supports 5 major mainstream text types: Simplified Chinese, Chinese Pinyin, Traditional Chinese, English, and Japanese. For scenarios, PP-OCRv5 has upgraded recognition capabilities for challenging scenarios such as complex Chinese and English handwriting, vertical text, and uncommon characters. On internal complex evaluation sets across multiple scenarios, PP-OCRv5 achieved a 13 percentage point end-to-end improvement over PP-OCRv4.

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/algorithm_ppocrv5.png" width="600"/>
</div>

# Key Metrics

### 1. Text Detection Metrics
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Handwritten Chinese</th>
      <th>Handwritten English</th>
      <th>Printed Chinese</th>
      <th>Printed English</th>
      <th>Traditional Chinese</th>
      <th>Ancient Text</th>
      <th>Japanese</th>
      <th>General Scenario</th>
      <th>Pinyin</th>
      <th>Rotation</th>
      <th>Distortion</th>
      <th>Artistic Text</th>
      <th>Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>PP-OCRv5_server_det</b></td>
      <td><b>0.803</b></td>
      <td><b>0.841</b></td>
      <td><b>0.945</b></td>
      <td><b>0.917</b></td>
      <td><b>0.815</b></td>
      <td><b>0.676</b></td>
      <td><b>0.772</b></td>
      <td><b>0.797</b></td>
      <td><b>0.671</b></td>
      <td><b>0.8</b></td>
      <td><b>0.876</b></td>
      <td><b>0.673</b></td>
      <td><b>0.827</b></td>
    </tr>
    <tr>
      <td>PP-OCRv4_server_det</td>
      <td>0.706</td>
      <td>0.249</td>
      <td>0.888</td>
      <td>0.690</td>
      <td>0.759</td>
      <td>0.473</td>
      <td>0.685</td>
      <td>0.715</td>
      <td>0.542</td>
      <td>0.366</td>
      <td>0.775</td>
      <td>0.583</td>
      <td>0.662</td>
    </tr>
    <tr>
      <td><b>PP-OCRv5_mobile_det</b></td>
      <td><b>0.744</b></td>
      <td><b>0.777</b></td>
      <td><b>0.905</b></td>
      <td><b>0.910</b></td>
      <td><b>0.823</b></td>
      <td><b>0.581</b></td>
      <td><b>0.727</b></td>
      <td><b>0.721</b></td>
      <td><b>0.575</b></td>
      <td><b>0.647</b></td>
      <td><b>0.827</b></td>
      <td>0.525</td>
      <td><b>0.770</b></td>
    </tr>
    <tr>
      <td>PP-OCRv4_mobile_det</td>
      <td>0.583</td>
      <td>0.369</td>
      <td>0.872</td>
      <td>0.773</td>
      <td>0.663</td>
      <td>0.231</td>
      <td>0.634</td>
      <td>0.710</td>
      <td>0.430</td>
      <td>0.299</td>
      <td>0.715</td>
      <td><b>0.549</b></td>
      <td>0.624</td>
    </tr>
  </tbody>
</table>

Compared to PP-OCRv4, PP-OCRv5 shows significant improvement in all detection scenarios, especially in handwriting, ancient texts, and Japanese detection capabilities.

### 2. Text Recognition Metrics

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/ocrv5_rec_acc.png" width="600"/>
</div>

<table>
  <thead>
    <tr>
      <th>Evaluation Set Category</th>
      <th>Handwritten Chinese</th>
      <th>Handwritten English</th>
      <th>Printed Chinese</th>
      <th>Printed English</th>
      <th>Traditional Chinese</th>
      <th>Ancient Text</th>
      <th>Japanese</th>
      <th>Confusable Characters</th>
      <th>General Scenario</th>
      <th>Pinyin</th>
      <th>Vertical Text</th>
      <th>Artistic Text</th>
      <th>Weighted Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PP-OCRv5_server_rec</td>
      <td><b>0.5807</b></td>
      <td><b>0.5806</b></td>
      <td><b>0.9013</b></td>
      <td><b>0.8679</b></td>
      <td><b>0.7472</b></td>
      <td><b>0.6039</b></td>
      <td><b>0.7372</b></td>
      <td><b>0.5946</b></td>
      <td><b>0.8384</b></td>
      <td><b>0.7435</b></td>
      <td><b>0.9314</b></td>
      <td><b>0.6397</b></td>
      <td><b>0.8401</b></td>
    </tr>
    <tr>
      <td>PP-OCRv4_server_rec</td>
      <td>0.3626</td>
      <td>0.2661</td>
      <td>0.8486</td>
      <td>0.6677</td>
      <td>0.4097</td>
      <td>0.3080</td>
      <td>0.4623</td>
      <td>0.5028</td>
      <td>0.8362</td>
      <td>0.2694</td>
      <td>0.5455</td>
      <td>0.5892</td>
      <td>0.5735</td>
    </tr>
    <tr>
      <td>PP-OCRv5_mobile_rec</td>
      <td><b>0.4166</b></td>
      <td><b>0.4944</b></td>
      <td><b>0.8605</b></td>
      <td><b>0.8753</b></td>
      <td><b>0.7199</b></td>
      <td><b>0.5786</b></td>
      <td><b>0.7577</b></td>
      <td><b>0.5570</b></td>
      <td>0.7703</td>
      <td><b>0.7248</b></td>
      <td><b>0.8089</b></td>
      <td>0.5398</td>
      <td><b>0.8015</b></td>
    </tr>
    <tr>
      <td>PP-OCRv4_mobile_rec</td>
      <td>0.2980</td>
      <td>0.2550</td>
      <td>0.8398</td>
      <td>0.6598</td>
      <td>0.3218</td>
      <td>0.2593</td>
      <td>0.4724</td>
      <td>0.4599</td>
      <td><b>0.8106</b></td>
      <td>0.2593</td>
      <td>0.5924</td>
      <td><b>0.5555</b></td>
      <td>0.5301</td>
    </tr>
  </tbody>
</table>

A single model can cover multiple languages and text types, with recognition accuracy significantly ahead of previous generation products and mainstream open-source solutions.

# PP-OCRv5 Demo Examples

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/algorithm_ppocrv5_demo1.png" width="600"/>
</div>

<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/PP-OCRv5/algorithm_ppocrv5_demo.pdf">More Demos</a>

# Deployment and Secondary Development
* **Multiple System Support**: Compatible with mainstream operating systems including Windows, Linux, and Mac.
* **Multiple Hardware Support**: Besides NVIDIA GPUs, it also supports inference and deployment on Intel CPU, Kunlun chips, Ascend, and other new hardware.
* **High-Performance Inference Plugin**: Recommended to combine with high-performance inference plugins to further improve inference speed. See [High-Performance Inference Guide](../../deployment/high_performance_inference.md) for details.
* **Service Deployment**: Supports highly stable service deployment solutions. See [Service Deployment Guide](../../deployment/serving.md) for details.
* **Secondary Development Capability**: Supports custom dataset training, dictionary extension, and model fine-tuning. Example: To add Korean recognition, you can extend the dictionary and fine-tune the model, seamlessly integrating into existing production lines. See [Text Detection Module Usage Tutorial](../../module_usage/text_detection.en.md) and [Text Recognition Module Usage Tutorial](../../module_usage/text_recognition.en.md) for details.
