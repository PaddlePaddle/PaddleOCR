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
- `text_det_limit_type` is set to `"min"` and `text_det_limit_side_len` to `736`.

### 1. Comparison of Inference Performance Between PP-OCRv5 and PP-OCRv4

| Config | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| v5_mobile      | Uses PP-OCRv5_mobile_det and PP-OCRv5_mobile_rec models. |
| v4_mobile      | Uses PP-OCRv4_mobile_det and PP-OCRv4_mobile_rec models. |
| v5_server      | Uses PP-OCRv5_server_det and PP-OCRv5_server_rec models. |
| v4_server      | Uses PP-OCRv4_server_det and PP-OCRv4_server_rec models. |

**GPU**

| Configuration | Avg. Time per Image (s) | Avg. Characters Predicted per Second | Avg. CPU Utilization (%) | Peak RAM Usage (MB) | Avg. RAM Usage (MB) | Avg. GPU Utilization (%) | Peak VRAM Usage (MB) | Avg. VRAM Usage (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v5_mobile | 0.62 | 1054.23 | 106.35 | 1829.36 | 1521.92 | 17.42 | 4190.00  | 3114.02 |
| v4_mobile | 0.29 | 2062.53 | 112.21 | 1713.10 | 1456.14 | 26.53  | 1304.00 | 1166.68 |
| v5_server | 0.74 | 878.84 | 105.68 | 1899.80 | 1569.46 | 34.39 | 5402.00 | 4683.93 |
| v4_server | 0.47 | 1322.06 | 108.06 | 1773.10 | 1518.94 | 55.25 | 6760.67 | 5788.02 |


**CPU**

| Configuration | Avg. Time per Image (s) | Avg. Characters Predicted per Second | Avg. CPU Utilization (%) | Peak RAM Usage (MB) | Avg. RAM Usage (MB) |
| ------------- | ----------------------- | ------------------------------------ | ------------------------ | ------------------- | ------------------- |
| v5_mobile | 1.75 | 371.82 | 965.89 | 2219.98 | 1830.97 |
| v4_mobile | 1.37 | 444.27 | 1007.33 | 2090.53 | 1797.76 |
| v5_server | 4.34 | 149.98 | 990.24 | 4020.85 | 3137.20 |
| v4_server | 5.42 | 115.20 | 999.03 | 4018.35 | 3105.29 |

> Note: PP-OCRv5 uses a larger dictionary in the recognition model, which increases inference time and causes slower performance compared to PP-OCRv4.

### 2. Impact of Auxiliary Features on PP-OCRv5 Inference Performance

| Config | Description                                                                                               |
| --------------- | --------------------------------------------------------------------------------------------------------- |
| base            | No document orientation classification, no image correction, no text line orientation classification.     |
| with_textline  | Includes text line orientation classification only.                                                       |
| with_all       | Includes document orientation classification, image correction, and text line orientation classification. |

**GPU**

| Configuration  | Avg. Time per Image (s) | Avg. Characters Predicted per Second | Avg. CPU Utilization (%) | Peak RAM Usage (MB) | Avg. RAM Usage (MB) | Avg. GPU Utilization (%) | Peak VRAM Usage (MB) | Avg. VRAM Usage (MB) |
| -------------- | ----------------------- | ------------------------------------ | ------------------------ | ------------------- | ------------------- | ------------------------ | -------------------- | -------------------- |
| base | 0.62 | 1054.23 | 106.35 | 1829.36 | 1521.92 | 17.42 | 4190.00 | 3114.02 |
| with_textline | 0.64 | 1012.32 | 106.37 | 1867.69 | 1527.42 | 19.16 | 4198.00 | 3115.05 |
| with_all | 1.09 | 562.99 | 105.67 | 2381.53 | 1792.48 | 10.77 | 2480.00 | 2065.54 |

**CPU**

| Configuration  | Avg. Time per Image (s) | Avg. Characters Predicted per Second | Avg. CPU Utilization (%) | Peak RAM Usage (MB) | Avg. RAM Usage (MB) |
| -------------- | ----------------------- | ------------------------------------ | ------------------------ | ------------------- | ------------------- |
| base | 1.75 | 371.82 | 965.89 | 2219.98 | 1830.97 |
| with_textline | 1.87 | 347.61 | 972.08 | 2232.38 | 1822.13 |
| with_all | 3.13 | 195.25 | 828.37 | 2751.47 | 2179.70 |

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

**GPU**

| Configuration     | Avg. Time per Image (s) | Avg. Characters Predicted per Second | Avg. CPU Utilization (%) | Peak RAM Usage (MB) | Avg. RAM Usage (MB) | Avg. GPU Utilization (%) | Peak VRAM Usage (MB) | Avg. VRAM Usage (MB) |
| ----------------- | ----------------------- | ------------------------------------ | ------------------------ | ------------------- | ------------------- | ------------------------ | -------------------- | -------------------- |
| mobile_min_1280 | 0.66 | 985.77 | 109.52 | 1878.74 | 1536.43 | 18.01 | 4050.00 | 3407.33 |
| mobile_min_736 | 0.62 | 1054.23 | 106.35 | 1829.36 | 1521.92 | 17.42 | 4190.00 | 3114.02 |
| mobile_max_960 | 0.52 | 1206.68 | 104.01 | 1795.27 | 1484.73 | 18.66 | 2490.00 | 2173.91 |
| mobile_max_640 | 0.45 | 1353.49 | 103.32 | 1728.91 | 1470.64 | 18.55 | 2378.00 | 1998.62 |
| server_min_1280 | 0.86 | 759.10 | 107.81 | 1876.31 | 1572.20 | 37.33 | 10368.00 | 8287.41 |
| server_min_736 | 0.74 | 878.84 | 105.68 | 1899.80 | 1569.46 | 34.39 | 5402.00 | 4683.93 |
| server_max_960 | 0.64 | 988.85 | 103.61 | 1831.31 | 1544.26 | 30.29 | 2929.33 | 2079.90 |
| server_max_640 | 0.57 | 1036.90 | 102.89 | 1838.36 | 1532.50 | 28.91 | 3153.33 | 2743.40 |


**CPU**

| Configuration     | Avg. Time per Image (s) | Avg. Characters Predicted per Second | Avg. CPU Utilization (%) | Peak RAM Usage (MB) | Avg. RAM Usage (MB) |
| ----------------- | ----------------------- | ------------------------------------ | ------------------------ | ------------------- | ------------------- |
| mobile_min_1280 | 2.00 | 326.44 | 976.83 | 2233.16 | 1867.94 |
| mobile_min_736 | 1.75 | 371.82 | 965.89 | 2219.98 | 1830.97 |
| mobile_max_960 | 1.49 | 422.62 | 969.11 | 2048.67 | 1677.82 |
| mobile_max_640 | 1.31 | 459.11 | 978.41 | 2023.25 | 1616.42 |
| server_min_1280 | 5.57 | 117.08 | 991.34 | 4452.39 | 3286.19 |
| server_min_736 | 4.34 | 149.98 | 990.24 | 4020.85 | 3137.20 |
| server_max_960 | 3.39 | 186.59 | 984.67 | 3492.62 | 2977.13 |
| server_max_640 | 2.95 | 201.00 | 980.59 | 3342.38 | 2935.24 |

# Deployment and Secondary Development
* **Multiple System Support**: Compatible with mainstream operating systems including Windows, Linux, and Mac.
* **Multiple Hardware Support**: Besides NVIDIA GPUs, it also supports inference and deployment on Intel CPU, Kunlun chips, Ascend, and other new hardware.
* **High-Performance Inference Plugin**: Recommended to combine with high-performance inference plugins to further improve inference speed. See [High-Performance Inference Guide](../../deployment/high_performance_inference.md) for details.
* **Service Deployment**: Supports highly stable service deployment solutions. See [Service Deployment Guide](../../deployment/serving.md) for details.
* **Secondary Development Capability**: Supports custom dataset training, dictionary extension, and model fine-tuning. Example: To add Korean recognition, you can extend the dictionary and fine-tune the model, seamlessly integrating into existing pipelines. See [Text Detection Module Usage Tutorial](../../module_usage/text_detection.en.md) and [Text Recognition Module Usage Tutorial](../../module_usage/text_recognition.en.md) for details.
