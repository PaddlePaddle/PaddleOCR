# 1. Introduction to PP-OCRv5
**PP-OCRv5** is the new generation text recognition solution of the PP-OCR series, focusing on multi-scenario and multi-type text recognition. In terms of text types, PP-OCRv5 supports five mainstream types: Simplified Chinese, Chinese Pinyin, Traditional Chinese, English, and Japanese. For scenarios, PP-OCRv5 has enhanced recognition for complex handwritten Chinese and English, vertical text, rare characters, and other challenging cases. On the internal multi-scenario complex benchmark, PP-OCRv5 achieves a 13 percentage point end-to-end improvement over PP-OCRv4.

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/algorithm_ppocrv5.png" width="400"/>

# 2. Key Metrics
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
      <th>Ancient Books</th>
      <th>Japanese</th>
      <th>General Scenario</th>
      <th>Pinyin</th>
      <th>Rotated</th>
      <th>Distorted</th>
      <th>Art Text</th>
      <th>Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PP-OCRv5_server_det</td>
      <td>0.803</td>
      <td>0.841</td>
      <td>0.945</td>
      <td>0.917</td>
      <td>0.815</td>
      <td>0.676</td>
      <td>0.772</td>
      <td>0.797</td>
      <td>0.671</td>
      <td>0.8</td>
      <td>0.876</td>
      <td>0.673</td>
      <td>0.827</td>
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
      <td>PP-OCRv5_mobile_det</td>
      <td>0.744</td>
      <td>0.777</td>
      <td>0.905</td>
      <td>0.910</td>
      <td>0.823</td>
      <td>0.581</td>
      <td>0.727</td>
      <td>0.721</td>
      <td>0.575</td>
      <td>0.647</td>
      <td>0.827</td>
      <td>0.525</td>
      <td>0.770</td>
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
      <td>0.549</td>
      <td>0.624</td>
    </tr>
  </tbody>
</table>

Compared to PP-OCRv4, PP-OCRv5 shows significant improvements in all detection scenarios, especially in handwritten, ancient books, and Japanese detection.

### 2. Text Recognition Metrics

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/ocrv5_rec_acc.png" width="400"/>

<table>
  <thead>
    <tr>
      <th>Evaluation Set</th>
      <th>Handwritten Chinese</th>
      <th>Handwritten English</th>
      <th>Printed Chinese</th>
      <th>Printed English</th>
      <th>Traditional Chinese</th>
      <th>Ancient Books</th>
      <th>Japanese</th>
      <th>Easily Confused Characters</th>
      <th>General Scenario</th>
      <th>Pinyin</th>
      <th>Vertical Text</th>
      <th>Art Text</th>
      <th>Weighted Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PP-OCRv5_server_rec</td>
      <td>0.5807</td>
      <td>0.5806</td>
      <td>0.9013</td>
      <td>0.8679</td>
      <td>0.7472</td>
      <td>0.6039</td>
      <td>0.7372</td>
      <td>0.5946</td>
      <td>0.8384</td>
      <td>0.7435</td>
      <td>0.9314</td>
      <td>0.6397</td>
      <td>0.8401</td>
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
      <td>0.4166</td>
      <td>0.4944</td>
      <td>0.8605</td>
      <td>0.8753</td>
      <td>0.7199</td>
      <td>0.5786</td>
      <td>0.7577</td>
      <td>0.5570</td>
      <td>0.7703</td>
      <td>0.7248</td>
      <td>0.8089</td>
      <td>0.5398</td>
      <td>0.8015</td>
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
      <td>0.8106</td>
      <td>0.2593</td>
      <td>0.5924</td>
      <td>0.5555</td>
      <td>0.5301</td>
    </tr>
  </tbody>
</table>

A single model can cover multiple languages and text types, with accuracy far ahead of previous products and mainstream open-source solutions.

# 4. End-to-End Inference Benchmark
Dependencies: Paddle Framework 3.0, CUDA 11.8, cuDNN 8.9.

Input data: 1185 images, covering both general and document scenarios.

During testing, the document image preprocessing model was loaded but not used (disabled via API parameter).

Local inference:

1. Test hardware: NVIDIA Tesla V100 + Intel Xeon Gold 6271C

<table>
  <thead>
    <tr>
      <th>Pipeline Config</th>
      <th>Avg Time/Image (s)</th>
      <th>Avg CPU Usage (%)</th>
      <th>Peak RAM (MB)</th>
      <th>Avg RAM (MB)</th>
      <th>Avg GPU Usage (%)</th>
      <th>Peak VRAM (MB)</th>
      <th>Avg VRAM (MB)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PP-OCRv5_server</td>
      <td>0.50</td>
      <td>103.4</td>
      <td>2446.0</td>
      <td>1985.4</td>
      <td>20.9</td>
      <td>2816.0</td>
      <td>2668.6</td>
    </tr>
    <tr>
      <td>PP-OCRv5_mobile</td>
      <td>0.62</td>
      <td>103.0</td>
      <td>2466.3</td>
      <td>1972.1</td>
      <td>33.5</td>
      <td>3178.0</td>
      <td>2978.0</td>
    </tr>
  </tbody>
</table>

2. Test hardware: NVIDIA A100 + Intel Xeon Platinum 8350C

<table>
  <thead>
    <tr>
      <th>Pipeline Config</th>
      <th>Avg Time/Image (s)</th>
      <th>Avg CPU Usage (%)</th>
      <th>Peak RAM (MB)</th>
      <th>Avg RAM (MB)</th>
      <th>Avg GPU Usage (%)</th>
      <th>Peak VRAM (MB)</th>
      <th>Avg VRAM (MB)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PP-OCRv5_server</td>
      <td>0.32</td>
      <td>101.9</td>
      <td>5461.6</td>
      <td>5032.0</td>
      <td>15.5</td>
      <td>4028.0</td>
      <td>3877.1</td>
    </tr>
    <tr>
      <td>PP-OCRv5_mobile</td>
      <td>0.38</td>
      <td>101.8</td>
      <td>5450.7</td>
      <td>4998.4</td>
      <td>24.0</td>
      <td>5462.0</td>
      <td>5172.8</td>
    </tr>
  </tbody>
</table>

Service deployment:

Test hardware: NVIDIA A100 + Intel Xeon Platinum 8350C

<table>
  <thead>
    <tr>
      <th>Pipeline Config</th>
      <th>Instance Count</th>
      <th>Concurrent Requests</th>
      <th>Throughput</th>
      <th>Avg Latency (s)</th>
      <th>Success/Total Requests</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">PP-OCRv5_server</td>
      <td>4 GPUs * 1</td>
      <td>4</td>
      <td>7.20</td>
      <td>0.55</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>4 GPUs * 4</td>
      <td>16</td>
      <td>21.78</td>
      <td>0.73</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>4 GPUs * 8</td>
      <td>32</td>
      <td>28.57</td>
      <td>1.11</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td rowspan="3">PP-OCRv5_mobile</td>
      <td>4 GPUs * 1</td>
      <td>4</td>
      <td>7.95</td>
      <td>0.50</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>4 GPUs * 4</td>
      <td>16</td>
      <td>24.94</td>
      <td>0.64</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>4 GPUs * 8</td>
      <td>32</td>
      <td>29.92</td>
      <td>1.05</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>

# 5. Key Demo Example of PP-OCRv5

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/ocrv5_demo.gif" width="400"/>

# 6. Deployment and Secondary Development
* **Multi-system support**: Compatible with Windows, Linux, Mac and other mainstream operating systems.
* **Multi-hardware support**: In addition to Nvidia GPUs, inference and deployment on Intel CPUs, Kunlunxin, Ascend, etc. are supported.
* **High-performance inference plugin**: It is recommended to use the high-performance inference plugin to further improve the inference speed. See [High Performance Inference Guide](../../deployment/high_performance_inference.en.md) for details.
* **Service deployment**: Supports highly stable service deployment solutions, see [Service Deployment Guide](../../deployment/serving.en.md) for details.
* **Secondary development capability**: Supports custom dataset training, dictionary expansion, and model fine-tuning. For example, to add Korean recognition, you can expand the dictionary and fine-tune the model, seamlessly integrating into the existing pipeline. See [Text Recognition Module Usage Tutorial](../../module_usage/text_recognition.en.md) for details.
