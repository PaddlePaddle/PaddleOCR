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

## Reference Data for Inference Performance

Test Environment:

- NVIDIA Tesla V100
- Intel Xeon Gold 6271C
- PaddlePaddle 3.0.0

Tested on 200 images (including both general and document images). During testing, images are read from disk, so the image reading time and other associated overhead are also included in the total time consumption. If the images are preloaded into memory, the average time per image can be further reduced by approximately 25 ms.

Unless otherwise specified:

- PP-OCRv4_mobile_det and PP-OCRv4_mobile_rec models are used.
- Document orientation classification, image correction, and text line orientation classification are not used.
- `text_det_limit_type` is set to `"min"` and `text_det_limit_side_len` to `732`.

### 1. Comparison of Inference Performance Between PP-OCRv5 and PP-OCRv4

| Config | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| v5_mobile      | Uses PP-OCRv5_mobile_det and PP-OCRv5_mobile_rec models. |
| v4_mobile      | Uses PP-OCRv4_mobile_det and PP-OCRv4_mobile_rec models. |
| v5_server      | Uses PP-OCRv5_server_det and PP-OCRv5_server_rec models. |
| v4_server      | Uses PP-OCRv4_server_det and PP-OCRv4_server_rec models. |

**GPU, without high-performance inference:**

| Config     | Avg Time/Image (s) | Avg Chars/sec | Avg CPU Usage (%) | Peak RAM (MB) | Avg RAM (MB) | Peak VRAM (MB) | Avg VRAM (MB) |
| ---------- | ------------------ | ------------- | ----------------- | ------------- | ------------ | -------------- | ------------- |
| v5_mobile | 0.56               | 1162          | 106.02            | 1576.43       | 1420.83      | 4342.00        | 3258.95       |
| v4_mobile | 0.27               | 2246          | 111.20            | 1392.22       | 1318.76      | 1304.00        | 1166.46       |
| v5_server | 0.70               | 929           | 105.31            | 1634.85       | 1428.55      | 5402.00        | 4685.13       |
| v4_server | 0.44               | 1418          | 106.96            | 1455.34       | 1346.95      | 6760.00        | 5817.46       |

**GPU, with high-performance inference:**

| Config     | Avg Time/Image (s) | Avg Chars/sec | Avg CPU Usage (%) | Peak RAM (MB) | Avg RAM (MB) | Peak VRAM (MB) | Avg VRAM (MB) |
| ---------- | ------------------ | ------------- | ----------------- | ------------- | ------------ | -------------- | ------------- |
| v5_mobile | 0.50               | 1301          | 106.50            | 1338.12       | 1155.86      | 4112.00        | 3536.36       |
| v4_mobile | 0.21               | 2887          | 114.09            | 1113.27       | 1054.46      | 2072.00        | 1840.59       |
| v5_server | 0.60               | 1084          | 105.73            | 1980.73       | 1776.20      | 12150.00       | 11849.40      |
| v4_server | 0.36               | 1687          | 104.15            | 1186.42       | 1065.67      | 13058.00       | 12679.00      |

**CPU, without high-performance inference:**

| Config     | Avg Time/Image (s) | Avg Chars/sec | Avg CPU Usage (%) | Peak RAM (MB) | Avg RAM (MB) |
| ---------- | ------------------ | ------------- | ----------------- | ------------- | ------------ |
| v5_mobile | 1.43               | 455           | 798.93            | 11695.40      | 6829.09      |
| v4_mobile | 1.09               | 556           | 813.16            | 11996.30      | 6834.25      |
| v5_server | 3.79               | 172           | 799.24            | 50216.00      | 27902.40     |
| v4_server | 4.22               | 148           | 803.74            | 51428.70      | 28593.60     |

**CPU, with high-performance inference:**

| Config     | Avg Time/Image (s) | Avg Chars/sec | Avg CPU Usage (%) | Peak RAM (MB) | Avg RAM (MB) |
| ---------- | ------------------ | ------------- | ----------------- | ------------- | ------------ |
| v5_mobile | 1.14               | 571           | 339.68            | 3245.17       | 2560.55      |
| v4_mobile | 0.68               | 892           | 443.00            | 3057.38       | 2329.44      |
| v5_server | 3.56               | 183           | 797.03            | 45664.70      | 26905.90     |
| v4_server | 4.22               | 148           | 803.74            | 51428.70      | 28593.60     |

> Note: PP-OCRv5 uses a larger dictionary in the recognition model, which increases inference time and causes slower performance compared to PP-OCRv4.

### 2. Impact of Auxiliary Features on PP-OCRv5 Inference Performance

| Config | Description                                                                                               |
| --------------- | --------------------------------------------------------------------------------------------------------- |
| base            | No document orientation classification, no image correction, no text line orientation classification.     |
| with_textline  | Includes text line orientation classification only.                                                       |
| with_all       | Includes document orientation classification, image correction, and text line orientation classification. |

**GPU, without high-performance inference:**

| Config         | Avg Time/Image (s) | Avg Chars/sec | Avg CPU Usage (%) | Peak RAM (MB) | Avg RAM (MB) | Peak VRAM (MB) | Avg VRAM (MB) |
| -------------- | ------------------ | ------------- | ----------------- | ------------- | ------------ | -------------- | ------------- |
| base           | 0.56               | 1162          | 106.02            | 1576.43       | 1420.83      | 4342.00        | 3258.95       |
| with_textline | 0.60               | 1083          | 105.59            | 1715.65       | 1510.83      | 4342.00        | 3266.05       |
| with_all      | 1.01               | 605           | 104.89            | 1949.11       | 1612.00      | 2624.00        | 2210.15       |

**CPU, without high-performance inference:**

| Config         | Avg Time/Image (s) | Avg Chars/sec | Avg CPU Usage (%) | Peak RAM (MB) | Avg RAM (MB) |
| -------------- | ------------------ | ------------- | ----------------- | ------------- | ------------ |
| base           | 1.43               | 455           | 798.93            | 11695.40      | 6829.09      |
| with_textline | 1.43               | 454           | 801.90            | 11994.30      | 6947.94      |
| with_all      | 1.90               | 320           | 642.48            | 11710.80      | 6944.01      |

> Note: Auxiliary features such as image unwarping can impact inference accuracy. More features do not necessarily yield better results and may increase resource usage.

### 3. Impact of Input Scaling Strategy in Text Detection Module on PP-OCRv5 Inference Performance

| Config            | Description                                                                            |
| ----------------- | -------------------------------------------------------------------------------------- |
| mobile_min_1280 | Uses `min` limit type and `text_det_limit_side_len=1280` with PP-OCRv5_mobile models. |
| mobile_min_736  | Same as default, `min`, `side_len=736`.                                                |
| mobile_max_960  | Uses `max` limit type and `side_len=960`.                                              |
| mobile_max_640  | Uses `max` limit type and `side_len=640`.                                              |
| server_min_1280 | Uses `min`, `side_len=1280` with PP-OCRv5_server models.                              |
| server_min_736  | Same as default, `min`, `side_len=736`.                                                |
| server_max_960  | Uses `max`, `side_len=960`.                                                            |
| server_max_640  | Uses `max`, `side_len=640`.                                                            |

**GPU, without high-performance inference:**

| Config            | Avg Time/Image (s) | Avg Chars/sec | Avg CPU Usage (%) | Peak RAM (MB) | Avg RAM (MB) | Peak VRAM (MB) | Avg VRAM (MB) |
| ----------------- | ------------------ | ------------- | ----------------- | ------------- | ------------ | -------------- | ------------- |
| mobile_min_1280 | 0.61               | 1071          | 109.12            | 1663.71       | 1439.72      | 4202.00        | 3550.32       |
| mobile_min_736  | 0.56               | 1162          | 106.02            | 1576.43       | 1420.83      | 4342.00        | 3258.95       |
| mobile_max_960  | 0.48               | 1313          | 103.49            | 1587.25       | 1395.48      | 2642.00        | 2319.03       |
| mobile_max_640  | 0.42               | 1436          | 103.07            | 1651.14       | 1422.62      | 2530.00        | 2149.11       |
| server_min_1280 | 0.82               | 795           | 107.17            | 1678.16       | 1428.94      | 10368.00       | 8320.43       |
| server_min_736  | 0.70               | 929           | 105.31            | 1634.85       | 1428.55      | 5402.00        | 4685.13       |
| server_max_960  | 0.59               | 1073          | 103.03            | 1590.19       | 1383.62      | 2928.00        | 2079.47       |
| server_max_640  | 0.54               | 1099          | 102.63            | 1602.09       | 1416.49      | 3152.00        | 2737.81       |

**CPU, without high-performance inference:**

| Config            | Avg Time/Image (s) | Avg Chars/sec | Avg CPU Usage (%) | Peak RAM (MB) | Avg RAM (MB) |
| ----------------- | ------------------ | ------------- | ----------------- | ------------- | ------------ |
| mobile_min_1280 | 1.64               | 398           | 799.45            | 12344.10      | 7100.60      |
| mobile_min_736  | 1.43               | 455           | 798.93            | 11695.40      | 6829.09      |
| mobile_max_960  | 1.21               | 521           | 800.13            | 11099.10      | 6369.49      |
| mobile_max_640  | 1.01               | 597           | 802.52            | 9585.48       | 5573.52      |
| server_min_1280 | 4.48               | 145           | 800.49            | 50683.10      | 28273.30     |
| server_min_736  | 3.79               | 172           | 799.24            | 50216.00      | 27902.40     |
| server_max_960  | 2.67               | 237           | 797.63            | 49362.50      | 26075.60     |
| server_max_640  | 2.36               | 251           | 795.18            | 45656.10      | 24900.80     |

# Deployment and Secondary Development
* **Multiple System Support**: Compatible with mainstream operating systems including Windows, Linux, and Mac.
* **Multiple Hardware Support**: Besides NVIDIA GPUs, it also supports inference and deployment on Intel CPU, Kunlun chips, Ascend, and other new hardware.
* **High-Performance Inference Plugin**: Recommended to combine with high-performance inference plugins to further improve inference speed. See [High-Performance Inference Guide](../../deployment/high_performance_inference.md) for details.
* **Service Deployment**: Supports highly stable service deployment solutions. See [Service Deployment Guide](../../deployment/serving.md) for details.
* **Secondary Development Capability**: Supports custom dataset training, dictionary extension, and model fine-tuning. Example: To add Korean recognition, you can extend the dictionary and fine-tune the model, seamlessly integrating into existing pipelines. See [Text Detection Module Usage Tutorial](../../module_usage/text_detection.en.md) and [Text Recognition Module Usage Tutorial](../../module_usage/text_recognition.en.md) for details.
