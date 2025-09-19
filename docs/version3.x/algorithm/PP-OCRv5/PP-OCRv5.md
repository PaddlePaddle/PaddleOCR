# 一、PP-OCRv5简介
**PP-OCRv5** 是PP-OCR新一代文字识别解决方案，该方案聚焦于多场景、多文字类型的文字识别。在文字类型方面，PP-OCRv5支持简体中文、中文拼音、繁体中文、英文、日文5大主流文字类型，在场景方面，PP-OCRv5升级了中英复杂手写体、竖排文本、生僻字等多种挑战性场景的识别能力。在内部多场景复杂评估集上，PP-OCRv5较PP-OCRv4端到端提升13个百分点。

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/algorithm_ppocrv5.png" width="600"/>
</div>


# 二、关键指标
### 1. 文本检测指标
<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>手写中文</th>
      <th>手写英文</th>
      <th>印刷中文</th>
      <th>印刷英文</th>
      <th>繁体中文</th>
      <th>古籍文本</th>
      <th>日文</th>
      <th>通用场景</th>
      <th>拼音</th>
      <th>旋转</th>
      <th>扭曲</th>
      <th>艺术字</th>
      <th>平均</th>
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

对比PP-OCRv4，PP-OCRv5在所有检测场景下均有明显提升，尤其在手写、古籍、日文检测能力上表现更优。

### 2. 文本识别指标


<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/ocrv5_rec_acc.png" width="600"/>
</div>


<table>
  <thead>
    <tr>
      <th>评估集类别</th>
      <th>手写中文</th>
      <th>手写英文</th>
      <th>印刷中文</th>
      <th>印刷英文</th>
      <th>繁体中文</th>
      <th>古籍文本</th>
      <th>日文</th>
      <th>易混淆字符</th>
      <th>通用场景</th>
      <th>拼音</th>
      <th>竖直文本</th>
      <th>艺术字</th>
      <th>加权平均</th>
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

单模型即可覆盖多语言和多类型文本，识别精度大幅领先前代产品和主流开源方案。


# 三、PP-OCRv5 Demo示例

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/algorithm_ppocrv5_demo1.png" width="600"/>
</div>

<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/PP-OCRv5/algorithm_ppocrv5_demo.pdf">更多示例</a>

## 四、推理性能参考数据

测试环境：

- NVIDIA Tesla V100
- Intel Xeon Gold 6271C
- PaddlePaddle 3.0.0

在 200 张图像（包括通用图像与文档图像）上测试。测试时从磁盘读取图像，因此读图时间及其他额外开销也被包含在总耗时内。如果将图像提前载入到内存，可进一步减少平均每图约 25 ms 的时间开销。

如果不特别说明，则：

- 使用 PP-OCRv4_mobile_det 和 PP-OCRv4_mobile_rec 模型。
- 不使用文档图像方向分类、文本图像矫正、文本行方向分类。
- 将 `text_det_limit_type` 设置为 `"min"`、`text_det_limit_side_len` 设置为 `736`。

### 1. PP-OCRv5 与 PP-OCRv4 推理性能对比

| 配置 | 说明 |
| --- | --- |
| v5_mobile | 使用 PP-OCRv5_mobile_det 和 PP-OCRv5_mobile_rec 模型。 |
| v4_mobile | 使用 PP-OCRv4_mobile_det 和 PP-OCRv4_mobile_rec 模型。 |
| v5_server | 使用 PP-OCRv5_server_det 和 PP-OCRv5_server_rec 模型。 |
| v4_server | 使用 PP-OCRv4_server_det 和 PP-OCRv4_server_rec 模型。 |

**GPU：**

| 配置      | 平均每图耗时（s） | 平均每秒预测字符数量 | 平均 CPU 利用率（%） | 峰值 RAM 用量（MB） | 平均 RAM 用量（MB） | 平均 GPU 利用率（%） | 峰值 VRAM 用量（MB） | 平均 VRAM 用量（MB） |
|-----------|------------------|--------------------|--------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| v5_mobile | 0.62 | 1054.23 | 106.35 | 1829.36 | 1521.92 | 17.42 | 4190.00  | 3114.02 |
| v4_mobile | 0.29 | 2062.53 | 112.21 | 1713.10 | 1456.14 | 26.53  | 1304.00 | 1166.68 |
| v5_server | 0.74 | 878.84 | 105.68 | 1899.80 | 1569.46 | 34.39 | 5402.00 | 4683.93 |
| v4_server | 0.47 | 1322.06 | 108.06 | 1773.10 | 1518.94 | 55.25 | 6760.67 | 5788.02 |

**CPU：**

| 配置 | 平均每图耗时（s） | 平均每秒预测字符数量 | 平均 CPU 利用率（%） | 峰值 RAM 用量（MB） | 平均 RAM 用量（MB） |
| --------- | ---- | ------- | ------ | -------- | -------- |
| v5_mobile | 1.75 | 371.82 | 965.89 | 2219.98 | 1830.97 |
| v4_mobile | 1.37 | 444.27 | 1007.33 | 2090.53 | 1797.76 |
| v5_server | 4.34 | 149.98 | 990.24 | 4020.85 | 3137.20 |
| v4_server | 5.42 | 115.20 | 999.03 | 4018.35 | 3105.29 |

> 说明：PP-OCRv5 的识别模型使用了更大的字典，需要更长的推理时间，导致 PP-OCRv5 的推理速度慢于 PP-OCRv4。

### 2. 使用辅助功能对 PP-OCRv5 推理性能的影响

| 配置 | 说明 |
| --- | --- |
| base | 不使用文档图像方向分类、文本图像矫正、文本行方向分类。 |
| with_textline | 使用文本行方向分类，不使用文档图像方向分类、文本图像矫正。 |
| with_all | 使用文档图像方向分类、文本图像矫正、文本行方向分类。 |

**GPU：**

| 配置 | 平均每图耗时（s） | 平均每秒预测字符数量 | 平均 CPU 利用率（%） | 峰值 RAM 用量（MB） | 平均 RAM 用量（MB） | 平均GPU利用率（%） | 峰值 VRAM 用量（MB） | 平均 VRAM 用量（MB） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.62 | 1054.23 | 106.35 | 1829.36 | 1521.92 | 17.42 | 4190.00 | 3114.02 |
| with_textline | 0.64 | 1012.32 | 106.37 | 1867.69 | 1527.42 | 19.16 | 4198.00 | 3115.05 |
| with_all | 1.09 | 562.99 | 105.67 | 2381.53 | 1792.48 | 10.77 | 2480.00 | 2065.54 |

**CPU：**

| 配置 | 平均每图耗时（s） | 平均每秒预测字符数量 | 平均 CPU 利用率（%） | 峰值 RAM 用量（MB） | 平均 RAM 用量（MB） |
| --- | --- | --- | --- | --- | --- |
| base | 1.75 | 371.82 | 965.89 | 2219.98 | 1830.97 |
| with_textline | 1.87 | 347.61 | 972.08 | 2232.38 | 1822.13 |
| with_all | 3.13 | 195.25 | 828.37 | 2751.47 | 2179.70 |

> 说明：文本图像矫正等辅助功能会对端到端推理精度造成影响，因此并不一定使用的辅助功能越多、资源用量越大。

### 3. 文本检测模块输入缩放尺寸策略对 PP-OCRv5 推理性能的影响

| 配置 | 说明 |
| --- | --- |
| mobile_min_1280 | 使用 PP-OCRv5_mobile_det 和 PP-OCRv5_mobile_rec 模型，将 `text_det_limit_type` 设置为 `"min"`、`text_det_limit_side_len` 设置为 `1280`。 |
| mobile_min_736 | 使用 PP-OCRv5_mobile_det 和 PP-OCRv5_mobile_rec 模型，将 `text_det_limit_type` 设置为 `"min"`、`text_det_limit_side_len` 设置为 `736`。 |
| mobile_max_960 | 使用 PP-OCRv5_mobile_det 和 PP-OCRv5_mobile_rec 模型，将 `text_det_limit_type` 设置为 `"max"`、`text_det_limit_side_len` 设置为 `960`。 |
| mobile_max_640 | 使用 PP-OCRv5_mobile_det 和 PP-OCRv5_mobile_rec 模型，将 `text_det_limit_type` 设置为 `"max"`、`text_det_limit_side_len` 设置为 `640`。 |
| server_min_1280 | 使用 PP-OCRv5_server_det 和 PP-OCRv5_server_rec 模型，将 `text_det_limit_type` 设置为 `"min"`、`text_det_limit_side_len` 设置为 `1280`。 |
| server_min_736 | 使用 PP-OCRv5_server_det 和 PP-OCRv5_server_rec 模型，将 `text_det_limit_type` 设置为 `"min"`、`text_det_limit_side_len` 设置为 `736`。 |
| server_max_960 | 使用 PP-OCRv5_server_det 和 PP-OCRv5_server_rec 模型，将 `text_det_limit_type` 设置为 `"max"`、`text_det_limit_side_len` 设置为 `960`。 |
| server_max_640 | 使用 PP-OCRv5_server_det 和 PP-OCRv5_server_rec 模型，将 `text_det_limit_type` 设置为 `"max"`、`text_det_limit_side_len` 设置为 `640`。 |

**GPU：**

| 配置 | 平均每图耗时（s） | 平均每秒预测字符数量 | 平均 CPU 利用率（%） | 峰值 RAM 用量（MB） | 平均 RAM 用量（MB） | 平均GPU利用率（%） | 峰值 VRAM 用量（MB） | 平均 VRAM 用量（MB） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mobile_min_1280 | 0.66 | 985.77 | 109.52 | 1878.74 | 1536.43 | 18.01 | 4050.00 | 3407.33 |
| mobile_min_736 | 0.62 | 1054.23 | 106.35 | 1829.36 | 1521.92 | 17.42 | 4190.00 | 3114.02 |
| mobile_max_960 | 0.52 | 1206.68 | 104.01 | 1795.27 | 1484.73 | 18.66 | 2490.00 | 2173.91 |
| mobile_max_640 | 0.45 | 1353.49 | 103.32 | 1728.91 | 1470.64 | 18.55 | 2378.00 | 1998.62 |
| server_min_1280 | 0.86 | 759.10 | 107.81 | 1876.31 | 1572.20 | 37.33 | 10368.00 | 8287.41 |
| server_min_736 | 0.74 | 878.84 | 105.68 | 1899.80 | 1569.46 | 34.39 | 5402.00 | 4683.93 |
| server_max_960 | 0.64 | 988.85 | 103.61 | 1831.31 | 1544.26 | 30.29 | 2929.33 | 2079.90 |
| server_max_640 | 0.57 | 1036.90 | 102.89 | 1838.36 | 1532.50 | 28.91 | 3153.33 | 2743.40 |

**CPU：**

| 配置 | 平均每图耗时（s） | 平均每秒预测字符数量 | 平均 CPU 利用率（%） | 峰值 RAM 用量（MB） | 平均 RAM 用量（MB） |
| --- | --- | --- | --- | --- | --- |
| mobile_min_1280 | 2.00 | 326.44 | 976.83 | 2233.16 | 1867.94 |
| mobile_min_736 | 1.75 | 371.82 | 965.89 | 2219.98 | 1830.97 |
| mobile_max_960 | 1.49 | 422.62 | 969.11 | 2048.67 | 1677.82 |
| mobile_max_640 | 1.31 | 459.11 | 978.41 | 2023.25 | 1616.42 |
| server_min_1280 | 5.57 | 117.08 | 991.34 | 4452.39 | 3286.19 |
| server_min_736 | 4.34 | 149.98 | 990.24 | 4020.85 | 3137.20 |
| server_max_960 | 3.39 | 186.59 | 984.67 | 3492.62 | 2977.13 |
| server_max_640 | 2.95 | 201.00 | 980.59 | 3342.38 | 2935.24 |

### 4. 产线基准测试数据

<details>
<summary>点击展开/折叠表格</summary>

<table border="1">
<tr><th>流水线配置</th><th>硬件</th><th>平均推理时间 (s)</th><th>峰值CPU利用率 (%)</th><th>平均CPU利用率 (%)</th><th>峰值主机内存 (MB)</th><th>平均主机内存 (MB)</th><th>峰值GPU利用率 (%)</th><th>平均GPU利用率 (%)</th><th>峰值设备内存 (MB)</th><th>平均设备内存 (MB)</th></tr>
<tr>
<td rowspan="7">OCR-default</td>
<td>Intel 6271C</td>
<td>3.97</td>
<td>1015.40</td>
<td>917.61</td>
<td>4381.22</td>
<td>3457.78</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>3.79</td>
<td>1022.50</td>
<td>921.68</td>
<td>4675.46</td>
<td>3585.96</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.65</td>
<td>113.50</td>
<td>102.48</td>
<td>2240.15</td>
<td>1868.44</td>
<td>47</td>
<td>19.60</td>
<td>7612.00</td>
<td>6634.15</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>1.06</td>
<td>114.90</td>
<td>103.05</td>
<td>2142.66</td>
<td>1791.43</td>
<td>72</td>
<td>20.01</td>
<td>5516.00</td>
<td>4812.81</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.65</td>
<td>108.90</td>
<td>101.95</td>
<td>2456.05</td>
<td>2080.26</td>
<td>100</td>
<td>36.52</td>
<td>6736.00</td>
<td>6017.05</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.74</td>
<td>115.90</td>
<td>102.22</td>
<td>2352.88</td>
<td>1993.39</td>
<td>100</td>
<td>25.56</td>
<td>6762.00</td>
<td>6039.93</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>1.17</td>
<td>107.10</td>
<td>101.78</td>
<td>2361.88</td>
<td>1986.61</td>
<td>100</td>
<td>51.11</td>
<td>5282.00</td>
<td>4585.10</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-mobile</td>
<td>Intel 6271C</td>
<td>1.39</td>
<td>1019.60</td>
<td>1007.69</td>
<td>2178.12</td>
<td>1873.73</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.15</td>
<td>1015.70</td>
<td>1006.87</td>
<td>2184.91</td>
<td>1916.85</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.35</td>
<td>110.80</td>
<td>103.77</td>
<td>2022.49</td>
<td>1808.11</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.27</td>
<td>110.90</td>
<td>103.80</td>
<td>1762.36</td>
<td>1525.04</td>
<td>31</td>
<td>19.30</td>
<td>4328.00</td>
<td>3356.30</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.55</td>
<td>113.80</td>
<td>103.68</td>
<td>1728.02</td>
<td>1470.52</td>
<td>38</td>
<td>18.59</td>
<td>4198.00</td>
<td>3199.12</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.22</td>
<td>111.90</td>
<td>103.99</td>
<td>2073.88</td>
<td>1876.14</td>
<td>32</td>
<td>20.25</td>
<td>4386.00</td>
<td>3435.86</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.31</td>
<td>119.90</td>
<td>104.24</td>
<td>2037.38</td>
<td>1771.06</td>
<td>52</td>
<td>32.74</td>
<td>3446.00</td>
<td>2733.21</td>
</tr>
<tr>
<td>M4</td>
<td>6.51</td>
<td>147.30</td>
<td>106.24</td>
<td>3550.58</td>
<td>3236.75</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.46</td>
<td>111.90</td>
<td>103.11</td>
<td>2035.38</td>
<td>1742.39</td>
<td>65</td>
<td>46.77</td>
<td>3968.00</td>
<td>2991.91</td>
</tr>
<tr>
<td rowspan="7">OCR-nopp-server</td>
<td>Intel 6271C</td>
<td>3.00</td>
<td>1016.00</td>
<td>1004.87</td>
<td>4445.46</td>
<td>3179.86</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>3.23</td>
<td>1010.70</td>
<td>1002.63</td>
<td>4175.39</td>
<td>3137.58</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.34</td>
<td>110.90</td>
<td>103.30</td>
<td>1904.99</td>
<td>1591.10</td>
<td>57</td>
<td>32.29</td>
<td>7494.00</td>
<td>6551.47</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.69</td>
<td>108.90</td>
<td>102.95</td>
<td>1808.30</td>
<td>1568.64</td>
<td>72</td>
<td>35.30</td>
<td>5410.00</td>
<td>4741.18</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.38</td>
<td>109.40</td>
<td>102.34</td>
<td>2100.00</td>
<td>1863.73</td>
<td>100</td>
<td>50.18</td>
<td>6614.00</td>
<td>5926.51</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.41</td>
<td>109.00</td>
<td>103.18</td>
<td>2055.21</td>
<td>1845.14</td>
<td>100</td>
<td>47.15</td>
<td>6654.00</td>
<td>5951.22</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.82</td>
<td>104.40</td>
<td>101.73</td>
<td>1906.88</td>
<td>1689.69</td>
<td>100</td>
<td>76.41</td>
<td>5178.00</td>
<td>4502.64</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-min736-mobile</td>
<td>Intel 6271C</td>
<td>1.41</td>
<td>1020.10</td>
<td>1008.14</td>
<td>2184.16</td>
<td>1911.86</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.20</td>
<td>1015.70</td>
<td>1007.08</td>
<td>2254.04</td>
<td>1935.18</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.36</td>
<td>112.90</td>
<td>104.29</td>
<td>2174.58</td>
<td>1827.67</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.27</td>
<td>113.90</td>
<td>104.48</td>
<td>1717.55</td>
<td>1529.77</td>
<td>30</td>
<td>19.54</td>
<td>4328.00</td>
<td>3388.44</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.57</td>
<td>118.80</td>
<td>104.45</td>
<td>1693.10</td>
<td>1470.74</td>
<td>40</td>
<td>19.83</td>
<td>4198.00</td>
<td>3206.91</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.22</td>
<td>113.40</td>
<td>104.66</td>
<td>2037.13</td>
<td>1797.10</td>
<td>31</td>
<td>20.64</td>
<td>4384.00</td>
<td>3427.91</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.31</td>
<td>119.30</td>
<td>106.05</td>
<td>1879.15</td>
<td>1732.39</td>
<td>49</td>
<td>30.40</td>
<td>3446.00</td>
<td>2751.08</td>
</tr>
<tr>
<td>M4</td>
<td>6.39</td>
<td>124.90</td>
<td>107.16</td>
<td>3578.98</td>
<td>3209.90</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.47</td>
<td>109.60</td>
<td>103.26</td>
<td>1961.40</td>
<td>1742.95</td>
<td>60</td>
<td>44.26</td>
<td>3968.00</td>
<td>3002.81</td>
</tr>
<tr>
<td rowspan="7">OCR-nopp-min736-server</td>
<td>Intel 6271C</td>
<td>3.26</td>
<td>1068.50</td>
<td>1004.96</td>
<td>4582.52</td>
<td>3135.68</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>3.52</td>
<td>1010.70</td>
<td>1002.33</td>
<td>4723.23</td>
<td>3209.27</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.35</td>
<td>108.90</td>
<td>103.94</td>
<td>1703.65</td>
<td>1485.50</td>
<td>60</td>
<td>35.54</td>
<td>7492.00</td>
<td>6576.97</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.71</td>
<td>110.80</td>
<td>103.54</td>
<td>1800.06</td>
<td>1559.28</td>
<td>78</td>
<td>36.65</td>
<td>5410.00</td>
<td>4741.55</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.40</td>
<td>110.20</td>
<td>102.75</td>
<td>2012.64</td>
<td>1843.45</td>
<td>100</td>
<td>55.74</td>
<td>6614.00</td>
<td>5940.44</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.44</td>
<td>114.90</td>
<td>103.87</td>
<td>2002.72</td>
<td>1773.17</td>
<td>100</td>
<td>49.28</td>
<td>6654.00</td>
<td>5980.68</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.89</td>
<td>105.00</td>
<td>101.91</td>
<td>2149.31</td>
<td>1795.35</td>
<td>100</td>
<td>76.39</td>
<td>5176.00</td>
<td>4528.77</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-max640-mobile</td>
<td>Intel 6271C</td>
<td>1.00</td>
<td>1033.70</td>
<td>1005.95</td>
<td>2021.88</td>
<td>1743.27</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>0.88</td>
<td>1043.60</td>
<td>1006.77</td>
<td>1980.82</td>
<td>1724.51</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.28</td>
<td>125.70</td>
<td>101.56</td>
<td>1962.27</td>
<td>1782.68</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.21</td>
<td>122.50</td>
<td>101.87</td>
<td>1772.39</td>
<td>1569.55</td>
<td>29</td>
<td>18.74</td>
<td>2360.00</td>
<td>2039.07</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.43</td>
<td>133.80</td>
<td>101.82</td>
<td>1636.93</td>
<td>1464.10</td>
<td>37</td>
<td>20.94</td>
<td>2386.00</td>
<td>2055.30</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.18</td>
<td>119.90</td>
<td>102.12</td>
<td>2119.93</td>
<td>1889.49</td>
<td>29</td>
<td>20.92</td>
<td>2636.00</td>
<td>2321.11</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.24</td>
<td>126.80</td>
<td>101.78</td>
<td>1905.14</td>
<td>1739.93</td>
<td>48</td>
<td>30.71</td>
<td>2232.00</td>
<td>1911.18</td>
</tr>
<tr>
<td>M4</td>
<td>7.08</td>
<td>137.80</td>
<td>104.83</td>
<td>2931.08</td>
<td>2658.25</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.36</td>
<td>124.80</td>
<td>101.70</td>
<td>1983.21</td>
<td>1729.43</td>
<td>61</td>
<td>46.10</td>
<td>2162.00</td>
<td>1836.63</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-max960-mobile</td>
<td>Intel 6271C</td>
<td>1.21</td>
<td>1020.00</td>
<td>1008.49</td>
<td>2200.30</td>
<td>1800.74</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.01</td>
<td>1024.10</td>
<td>1007.32</td>
<td>2038.80</td>
<td>1800.05</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.32</td>
<td>107.50</td>
<td>102.00</td>
<td>2001.21</td>
<td>1799.01</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.23</td>
<td>107.70</td>
<td>102.33</td>
<td>1727.89</td>
<td>1490.18</td>
<td>30</td>
<td>20.19</td>
<td>2646.00</td>
<td>2385.40</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.49</td>
<td>109.90</td>
<td>102.26</td>
<td>1726.01</td>
<td>1504.90</td>
<td>38</td>
<td>20.11</td>
<td>2498.00</td>
<td>2227.73</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.20</td>
<td>109.90</td>
<td>102.52</td>
<td>1959.46</td>
<td>1798.35</td>
<td>28</td>
<td>19.38</td>
<td>2712.00</td>
<td>2450.10</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.27</td>
<td>102.90</td>
<td>101.19</td>
<td>1938.48</td>
<td>1741.19</td>
<td>47</td>
<td>29.27</td>
<td>3344.00</td>
<td>2585.02</td>
</tr>
<tr>
<td>M4</td>
<td>5.44</td>
<td>122.10</td>
<td>105.91</td>
<td>3094.72</td>
<td>2686.52</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.41</td>
<td>106.00</td>
<td>101.81</td>
<td>1859.88</td>
<td>1722.62</td>
<td>68</td>
<td>47.05</td>
<td>2264.00</td>
<td>2001.07</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-max640-server</td>
<td>Intel 6271C</td>
<td>2.16</td>
<td>1026.30</td>
<td>1005.10</td>
<td>3467.93</td>
<td>3074.06</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>2.30</td>
<td>1008.70</td>
<td>1003.32</td>
<td>3435.54</td>
<td>3042.62</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.35</td>
<td>104.70</td>
<td>101.27</td>
<td>1948.85</td>
<td>1779.77</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.25</td>
<td>104.90</td>
<td>101.42</td>
<td>1833.93</td>
<td>1560.71</td>
<td>41</td>
<td>27.61</td>
<td>4480.00</td>
<td>3955.14</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.56</td>
<td>106.20</td>
<td>101.47</td>
<td>1669.73</td>
<td>1500.87</td>
<td>58</td>
<td>31.78</td>
<td>3160.00</td>
<td>2838.78</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.23</td>
<td>109.40</td>
<td>101.45</td>
<td>1968.77</td>
<td>1800.81</td>
<td>58</td>
<td>30.81</td>
<td>2602.00</td>
<td>2588.77</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.30</td>
<td>106.10</td>
<td>101.55</td>
<td>2027.13</td>
<td>1749.07</td>
<td>69</td>
<td>39.10</td>
<td>3318.00</td>
<td>2795.54</td>
</tr>
<tr>
<td>M4</td>
<td>7.26</td>
<td>133.90</td>
<td>104.48</td>
<td>5473.38</td>
<td>3472.28</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.58</td>
<td>103.90</td>
<td>100.86</td>
<td>1884.23</td>
<td>1714.48</td>
<td>84</td>
<td>63.50</td>
<td>2852.00</td>
<td>2540.37</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-max960-server</td>
<td>Intel 6271C</td>
<td>2.53</td>
<td>1014.50</td>
<td>1005.22</td>
<td>3625.57</td>
<td>3151.73</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>2.66</td>
<td>1010.60</td>
<td>1003.39</td>
<td>3580.64</td>
<td>3197.09</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.40</td>
<td>105.90</td>
<td>101.76</td>
<td>2040.65</td>
<td>1810.97</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.29</td>
<td>108.90</td>
<td>102.12</td>
<td>1821.03</td>
<td>1620.02</td>
<td>44</td>
<td>30.38</td>
<td>4290.00</td>
<td>2928.79</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.60</td>
<td>109.90</td>
<td>101.98</td>
<td>1797.75</td>
<td>1544.96</td>
<td>61</td>
<td>32.48</td>
<td>2936.00</td>
<td>2117.71</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.28</td>
<td>108.80</td>
<td>101.92</td>
<td>2016.22</td>
<td>1811.74</td>
<td>73</td>
<td>41.82</td>
<td>2636.00</td>
<td>2241.23</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.34</td>
<td>111.00</td>
<td>102.75</td>
<td>1964.21</td>
<td>1750.21</td>
<td>68</td>
<td>41.25</td>
<td>2722.00</td>
<td>2293.74</td>
</tr>
<tr>
<td>M4</td>
<td>6.28</td>
<td>129.10</td>
<td>103.74</td>
<td>7780.70</td>
<td>3571.92</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.67</td>
<td>116.90</td>
<td>101.33</td>
<td>1941.09</td>
<td>1693.39</td>
<td>88</td>
<td>65.48</td>
<td>2714.00</td>
<td>1923.06</td>
</tr>
<tr>
<td rowspan="7">OCR-nopp-min1280-server</td>
<td>Intel 6271C</td>
<td>4.13</td>
<td>1043.40</td>
<td>1005.45</td>
<td>5993.70</td>
<td>3454.00</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>4.46</td>
<td>1011.70</td>
<td>996.72</td>
<td>5633.51</td>
<td>3489.79</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.42</td>
<td>113.90</td>
<td>106.08</td>
<td>1747.88</td>
<td>1546.18</td>
<td>85</td>
<td>43.73</td>
<td>13558.00</td>
<td>11297.98</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.82</td>
<td>116.80</td>
<td>105.18</td>
<td>1873.38</td>
<td>1609.55</td>
<td>100</td>
<td>39.57</td>
<td>10376.00</td>
<td>8427.30</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.55</td>
<td>114.80</td>
<td>103.14</td>
<td>2036.36</td>
<td>1864.45</td>
<td>100</td>
<td>69.67</td>
<td>13224.00</td>
<td>11411.31</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.55</td>
<td>105.90</td>
<td>101.86</td>
<td>1931.35</td>
<td>1764.44</td>
<td>100</td>
<td>56.16</td>
<td>12418.00</td>
<td>10510.77</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>1.13</td>
<td>105.90</td>
<td>102.35</td>
<td>2066.73</td>
<td>1787.78</td>
<td>100</td>
<td>83.50</td>
<td>10142.00</td>
<td>8338.80</td>
</tr>
<tr>
<td rowspan="9">OCR-nopp-min1280-mobile</td>
<td>Intel 6271C</td>
<td>1.59</td>
<td>1019.90</td>
<td>1008.39</td>
<td>2366.86</td>
<td>1992.03</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>1.29</td>
<td>1017.70</td>
<td>1007.28</td>
<td>2501.24</td>
<td>2059.99</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.43</td>
<td>120.90</td>
<td>107.02</td>
<td>2108.87</td>
<td>1821.91</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.29</td>
<td>117.90</td>
<td>107.19</td>
<td>1847.97</td>
<td>1570.89</td>
<td>31</td>
<td>18.98</td>
<td>3746.00</td>
<td>3321.86</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>0.61</td>
<td>122.80</td>
<td>107.07</td>
<td>1789.25</td>
<td>1542.56</td>
<td>39</td>
<td>20.52</td>
<td>4058.00</td>
<td>3487.46</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.24</td>
<td>116.80</td>
<td>106.80</td>
<td>2092.63</td>
<td>1882.77</td>
<td>28</td>
<td>18.67</td>
<td>3902.00</td>
<td>3444.00</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.34</td>
<td>125.80</td>
<td>106.79</td>
<td>1959.45</td>
<td>1783.97</td>
<td>49</td>
<td>32.66</td>
<td>3532.00</td>
<td>3094.29</td>
</tr>
<tr>
<td>M4</td>
<td>6.64</td>
<td>139.40</td>
<td>107.63</td>
<td>4283.97</td>
<td>3112.59</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>0.51</td>
<td>116.90</td>
<td>105.06</td>
<td>1927.22</td>
<td>1675.34</td>
<td>68</td>
<td>45.78</td>
<td>3828.00</td>
<td>3283.78</td>
</tr>
</table>


<table border="1">
<tr><th>Pipeline configuration</th><th>description</th></tr>
<tr>
<td>OCR-default</td>
<td>默认配置</td>
</tr>
<tr>
<td>OCR-nopp-mobile</td>
<td>默认配置基础上，关闭文档图像预处理，使用mobile的det和rec模型</td>
</tr>
<tr>
<td>OCR-nopp-server</td>
<td>默认配置基础上，关闭文档图像预处理</td>
</tr>
<tr>
<td>OCR-nopp-min736-mobile</td>
<td>默认配置基础上，关闭文档图像预处理，det模型输入缩放策略为min+736，使用mobile的det和rec模型</td>
</tr>
<tr>
<td>OCR-nopp-min736-server</td>
<td>默认配置基础上，关闭文档图像预处理，det模型输入缩放策略为min+736</td>
</tr>
<tr>
<td>OCR-nopp-max640-mobile</td>
<td>默认配置基础上，关闭文档图像预处理，det模型输入缩放策略为max+640，使用mobile的det和rec模型</td>
</tr>
<tr>
<td>OCR-nopp-max960-mobile</td>
<td>默认配置基础上，关闭文档图像预处理，det模型输入缩放策略为max+960，使用mobile的det和rec模型</td>
</tr>
<tr>
<td>OCR-nopp-max640-server</td>
<td>默认配置基础上，关闭文档图像预处理，det模型输入缩放策略为max+640</td>
</tr>
<tr>
<td>OCR-nopp-max960-server</td>
<td>默认配置基础上，关闭文档图像预处理，det模型输入缩放策略为max+960</td>
</tr>
<tr>
<td>OCR-nopp-min1280-server</td>
<td>默认配置基础上，关闭文档图像预处理，det模型输入缩放策略为min+1280</td>
</tr>
<tr>
<td>OCR-nopp-min1280-mobile</td>
<td>默认配置基础上，关闭文档图像预处理，det模型输入缩放策略为min+1280，使用mobile的det和rec模型</td>
</tr>
</table>
</details>


* 测试环境：
    * PaddlePaddle 3.1.0、CUDA 11.8、cuDNN 8.9
    * PaddleX @ develop (f1eb28e23cfa54ce3e9234d2e61fcb87c93cf407)
    * Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda11.8-cudnn8.9
* 测试数据：
    * 测试数据包含文档场景和通用场景的200张图像。
* 测试策略：
    * 使用 20 个样本进行预热，然后对整个数据集重复 1 次以进行速度性能测试。
* 备注：
    * 由于我们没有收集NPU和XPU的设备内存数据，因此表中相应位置的数据标记为N/A。

# 五、部署与二次开发
* **多系统支持**：兼容Windows、Linux、Mac等主流操作系统。
* **多硬件支持**：除了英伟达GPU外，还支持Intel CPU、昆仑芯、昇腾等新硬件推理和部署。
* **高性能推理插件**：推荐结合高性能推理插件进一步提升推理速度，详见[高性能推理指南](../../deployment/high_performance_inference.md)。
* **服务化部署**：支持高稳定性服务化部署方案，详见[服务化部署指南](../../deployment/serving.md)。
* **二次开发能力**：支持自定义数据集训练、字典扩展、模型微调。举例：如需增加韩文识别，可扩展字典并微调模型，无缝集成到现有产线，详见[文本检测模块使用教程](../../module_usage/text_detection.md)及[文本识别模块使用教程](../../module_usage/text_recognition.md)
