# 一、PP-StructureV3 简介
PP-StructureV3 能够将文档图像和 PDF 文件高效转换为结构化内容（如 Markdown 格式），并具备版面区域检测、表格识别、公式识别、图表理解以及多栏阅读顺序恢复等强大功能。该工具在多种文档类型下均表现优异，能够处理复杂的文档数据。PP-StructureV3 支持灵活的服务化部署，兼容多种硬件环境，并可通过多种编程语言进行调用。同时，支持二次开发，用户可以基于自有数据集进行模型训练和优化，训练后的模型可实现无缝集成。

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-StructureV3/algorithm_ppstructurev3.png" width="800"/>
</div>

# 二、关键指标

<div align="center">
<table>
  <thead>
    <tr>
      <th rowspan="2">Method Type</th>
      <th rowspan="2">Methods</th>
      <th colspan="2">Overall<sup>Edit</sup>↓</th>
      <th colspan="2">Text<sup>Edit</sup>↓</th>
      <th colspan="2">Formula<sup>Edit</sup>↓</th>
      <th colspan="2">Table<sup>Edit</sup>↓</th>
      <th colspan="2">Read Order<sup>Edit</sup>↓</th>
    </tr>
    <tr>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
    </tr>
  </thead>
 <tbody>
  <tr> 
   <td rowspan="9">Pipeline Tools</td> 
   <td><b>PP-structureV3</b></td> 
   <td><b>0.145</b></td> 
   <td><b>0.206</b></td> 
   <td>0.058</td> 
   <td><b>0.088</b></td> 
   <td>0.295</td> 
   <td>0.535</td> 
   <td>0.159</td> 
   <td><b>0.109</b></td> 
   <td>0.069</td> 
   <td><b>0.091</b></td> 
  </tr> 
  <tr> 
   <td>MinerU-0.9.3</td> 
   <td>0.15</td> 
   <td>0.357</td> 
   <td>0.061</td> 
   <td>0.215</td> 
   <td>0.278</td> 
   <td>0.577</td> 
   <td>0.18</td> 
   <td>0.344</td> 
   <td>0.079</td> 
   <td>0.292</td> 
  </tr> 
  <tr> 
   <td>MinerU-1.3.11</td> 
   <td>0.166</td> 
   <td>0.310</td> 
   <td>0.0826</td> 
   <td>0.2000</td> 
   <td>0.3368</td> 
   <td>0.6236</td> 
   <td>0.1613</td> 
   <td>0.1833</td> 
   <td>0.0834</td> 
   <td>0.2316</td> 
  </tr> 
  <tr> 
   <td>Marker-1.2.3</td> 
   <td>0.336</td> 
   <td>0.556</td> 
   <td>0.08</td> 
   <td>0.315</td> 
   <td>0.53</td> 
   <td>0.883</td> 
   <td>0.619</td> 
   <td>0.685</td> 
   <td>0.114</td> 
   <td>0.34</td> 
  </tr> 
  <tr> 
   <td>Mathpix</td> 
   <td>0.191</td> 
   <td>0.365</td> 
   <td>0.105</td> 
   <td>0.384</td> 
   <td>0.306</td> 
   <td>0.454</td> 
   <td>0.243</td> 
   <td>0.32</td> 
   <td>0.108</td> 
   <td>0.304</td> 
  </tr> 
  <tr> 
   <td>Docling-2.14.0</td> 
   <td>0.589</td> 
   <td>0.909</td> 
   <td>0.416</td> 
   <td>0.987</td> 
   <td>0.999</td> 
   <td>1</td> 
   <td>0.627</td> 
   <td>0.81</td> 
   <td>0.313</td> 
   <td>0.837</td> 
  </tr> 
  <tr> 
   <td>Pix2Text-1.1.2.3</td> 
   <td>0.32</td> 
   <td>0.528</td> 
   <td>0.138</td> 
   <td>0.356</td> 
   <td><b>0.276</b></td> 
   <td>0.611</td> 
   <td>0.584</td> 
   <td>0.645</td> 
   <td>0.281</td> 
   <td>0.499</td> 
  </tr> 
  <tr> 
   <td>Unstructured-0.17.2</td> 
   <td>0.586</td> 
   <td>0.716</td> 
   <td>0.198</td> 
   <td>0.481</td> 
   <td>0.999</td> 
   <td>1</td> 
   <td>1</td> 
   <td>0.998</td> 
   <td>0.145</td> 
   <td>0.387</td> 
  </tr> 
  <tr> 
   <td>OpenParse-0.7.0</td> 
   <td>0.646</td> 
   <td>0.814</td> 
   <td>0.681</td> 
   <td>0.974</td> 
   <td>0.996</td> 
   <td>1</td> 
   <td>0.284</td> 
   <td>0.639</td> 
   <td>0.595</td> 
   <td>0.641</td> 
  </tr> 
  <tr> 
   <td rowspan="5">Expert VLMs</td> 
   <td>GOT-OCR</td> 
   <td>0.287</td> 
   <td>0.411</td> 
   <td>0.189</td> 
   <td>0.315</td> 
   <td>0.36</td> 
   <td>0.528</td> 
   <td>0.459</td> 
   <td>0.52</td> 
   <td>0.141</td> 
   <td>0.28</td> 
  </tr> 
  <tr> 
   <td>Nougat</td> 
   <td>0.452</td> 
   <td>0.973</td> 
   <td>0.365</td> 
   <td>0.998</td> 
   <td>0.488</td> 
   <td>0.941</td> 
   <td>0.572</td> 
   <td>1</td> 
   <td>0.382</td> 
   <td>0.954</td> 
  </tr> 
  <tr> 
   <td>Mistral OCR</td> 
   <td>0.268</td> 
   <td>0.439</td> 
   <td>0.072</td> 
   <td>0.325</td> 
   <td>0.318</td> 
   <td>0.495</td> 
   <td>0.6</td> 
   <td>0.65</td> 
   <td>0.083</td> 
   <td>0.284</td> 
  </tr> 
  <tr> 
   <td>OLMOCR-sglang</td> 
   <td>0.326</td> 
   <td>0.469</td> 
   <td>0.097</td> 
   <td>0.293</td> 
   <td>0.455</td> 
   <td>0.655</td> 
   <td>0.608</td> 
   <td>0.652</td> 
   <td>0.145</td> 
   <td>0.277</td> 
  </tr> 
  <tr> 
   <td>SmolDocling-256M_transformer</td> 
   <td>0.493</td> 
   <td>0.816</td> 
   <td>0.262</td> 
   <td>0.838</td> 
   <td>0.753</td> 
   <td>0.997</td> 
   <td>0.729</td> 
   <td>0.907</td> 
   <td>0.227</td> 
   <td>0.522</td> 
  </tr> 
  <tr> 
   <td rowspan="6">General VLMs</td> 
   <td>Gemini2.0-flash</td> 
   <td>0.191</td> 
   <td>0.264</td> 
   <td>0.091</td> 
   <td>0.139</td> 
   <td>0.389</td> 
   <td>0.584</td> 
   <td>0.193</td> 
   <td>0.206</td> 
   <td>0.092</td> 
   <td>0.128</td> 
  </tr> 
  <tr> 
   <td>Gemini2.5-Pro</td> 
   <td>0.148</td> 
   <td><b>0.212</b></td> 
   <td><b>0.055</b></td> 
   <td>0.168</td> 
   <td>0.356</td> 
   <td>0.439</td> 
   <td><b>0.13</b></td> 
   <td>0.119</td> 
   <td><b>0.049</b></td> 
   <td>0.121</td> 
  </tr> 
  <tr> 
   <td>GPT4o</td> 
   <td>0.233</td> 
   <td>0.399</td> 
   <td>0.144</td> 
   <td>0.409</td> 
   <td>0.425</td> 
   <td>0.606</td> 
   <td>0.234</td> 
   <td>0.329</td> 
   <td>0.128</td> 
   <td>0.251</td> 
  </tr> 
  <tr> 
   <td>Qwen2-VL-72B</td> 
   <td>0.252</td> 
   <td>0.327</td> 
   <td>0.096</td> 
   <td>0.218</td> 
   <td>0.404</td> 
   <td>0.487</td> 
   <td>0.387</td> 
   <td>0.408</td> 
   <td>0.119</td> 
   <td>0.193</td> 
  </tr> 
  <tr> 
   <td>Qwen2.5-VL-72B</td> 
   <td>0.214</td> 
   <td>0.261</td> 
   <td>0.092</td> 
   <td>0.18</td> 
   <td>0.315</td> 
   <td><b>0.434</b></td> 
   <td>0.341</td> 
   <td>0.262</td> 
   <td>0.106</td> 
   <td>0.168</td> 
  </tr> 
  <tr> 
   <td>InternVL2-76B</td> 
   <td>0.44</td> 
   <td>0.443</td> 
   <td>0.353</td> 
   <td>0.29</td> 
   <td>0.543</td> 
   <td>0.701</td> 
   <td>0.547</td> 
   <td>0.555</td> 
   <td>0.317</td> 
   <td>0.228</td> 
  </tr> 
 </tbody>
</table>
</div>

以上部分数据出自：
* <a href="https://github.com/opendatalab/OmniDocBench">OmniDocBench</a>
* <a href="https://arxiv.org/abs/2412.07626">OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations</a>


# 三、推理 Benchmark

在不同GPU环境下，不同配置的 PP-StructureV3 和 MinerU 对比的性能指标如下。

基本测试环境：
* Paddle 3.0正式版
* PaddleOCR 3.0.0正式版
* MinerU 1.3.10
* CUDA 11.8
* cuDNN 8.9

## 3.1 本地推理

本地推理分别在 V100 和 A100 两种 GPU机器上，测试了 6 种不同配置下 PP-StructureV3 的性能，测试数据为15个PDF文件，共925页，包含表格、公式、印章、图表等元素。

下述 PP-StructureV3 配置中，OCR 模型详情请见[PP-OCRv5](../PP-OCRv5/PP-OCRv5.md)，公式识别模型详情请见[公式识别](../../module_usage/formula_recognition.md)，文本检测模块 max_side_limit 设置请见[文本检测](../../module_usage/text_detection.md)。

### NVIDIA Tesla V100 + Intel Xeon Gold 6271C
<table border="1">
 <tr>
  <td>
   方案
  </td>
  <td colspan="4">
   配置
  </td>
  <td rowspan="2">
   平均每页耗时
    （s）
  </td>
  <td rowspan="2">
   平均CPU利用率
    （%）
  </td>
  <td rowspan="2">
   峰值RAM用量
    （GB）
  </td>
  <td rowspan="2">
   平均RAM用量
    （GB）
  </td>
  <td rowspan="2">
   平均GPU利用率
    （%）
  </td>
  <td rowspan="2">
   峰值VRAM用量
    （GB）
  </td>
  <td rowspan="2">
   平均VRAM用量
    （GB）
  </td>
 </tr>
 <tr>
  <td rowspan="7">
   PP-StructureV3
  </td>
  <td>
   OCR模型
  </td>
  <td>
   公式识别模型
  </td>
  <td>
   是否启用图表识别模块
  </td>
  <td>
   文本检测max_side_limit
  </td>
 </tr>
 <tr>
  <td>
   Server系列
  </td>
  <td>
   PP-FormulaNet-L
  </td>
  <td>
   ✗
  </td>
  <td>
   4096
  </td>
  <td>
   1.77
  </td>
  <td>
   111.4
  </td>
  <td>
   6.7
  </td>
  <td>
   5.2
  </td>
  <td>
   38.9
  </td>
  <td>
   17.0
  </td>
  <td>
   16.5
  </td>
 </tr>
 <tr>
  <td>
   Server系列
  </td>
  <td>
   PP-FormulaNet-L
  </td>
  <td>
   ✔
  </td>
  <td>
   4096
  </td>
  <td>
   4.09
  </td>
  <td>
   105.3
  </td>
  <td>
   5.5
  </td>
  <td>
   4.0
  </td>
  <td>
   24.7
  </td>
  <td>
   17.0
  </td>
  <td>
   16.6
  </td>
 </tr>
 <tr>
  <td>
   Mobile系列
  </td>
  <td>
   PP-FormulaNet-L
  </td>
  <td>
   ✗
  </td>
  <td>
   4096
  </td>
  <td>
   1.56
  </td>
  <td>
   113.7
  </td>
  <td>
   6.6
  </td>
  <td>
   4.9
  </td>
  <td>
   29.1
  </td>
  <td>
   10.7
  </td>
  <td>
   10.6
  </td>
 </tr>
 <tr>
  <td>
   Server系列
  </td>
  <td>
   PP-FormulaNet-M
  </td>
  <td>
   ✗
  </td>
  <td>
   4096
  </td>
  <td>
   1.42
  </td>
  <td>
   112.9
  </td>
  <td>
   6.8
  </td>
  <td>
   5.1
  </td>
  <td>
   38
  </td>
  <td>
   16.0
  </td>
  <td>
   15.5
  </td>
 </tr>
 <tr>
  <td>
   Mobile系列
  </td>
  <td>
   PP-FormulaNet-M
  </td>
  <td>
   ✗
  </td>
  <td>
   4096
  </td>
  <td>
   1.15
  </td>
  <td>
   114.8
  </td>
  <td>
   6.5
  </td>
  <td>
   5.0
  </td>
  <td>
   26.1
  </td>
  <td>
   8.4
  </td>
  <td>
   8.3
  </td>
 </tr>
 <tr>
  <td>
   Mobile系列
  </td>
  <td>
   PP-FormulaNet-M
  </td>
  <td>
   ✗
  </td>
  <td>
   1200
  </td>
  <td>
   0.99
  </td>
  <td>
   113
  </td>
  <td>
   7.0
  </td>
  <td>
   5.6
  </td>
  <td>
   29.2
  </td>
  <td>
   8.6
  </td>
  <td>
   8.5
  </td>
 </tr>
 <tr>
  <td>
   MinerU
  </td>
  <td colspan="4">
   -
  </td>
  <td>
   1.57
  </td>
  <td>
   142.9
  </td>
  <td>
   13.3
  </td>
  <td>
   11.8
  </td>
  <td>
   43.3
  </td>
  <td>
   31.6
  </td>
  <td>
   9.7
  </td>
 </tr>
</table>

### NVIDIA A100 + Intel Xeon Platinum 8350C
<table border="1">
 <tr>
  <td>
   方案
  </td>
  <td colspan="4">
   配置
  </td>
  <td rowspan="2">
   平均每页耗时
    （s）
  </td>
  <td rowspan="2">
   平均CPU利用率
    （%）
  </td>
  <td rowspan="2">
   峰值RAM用量
    （GB）
  </td>
  <td rowspan="2">
   平均RAM用量
    （GB）
  </td>
  <td rowspan="2">
   平均GPU利用率
    （%）
  </td>
  <td rowspan="2">
   峰值VRAM用量
    （GB）
  </td>
  <td rowspan="2">
   平均VRAM用量
    （GB）
  </td>
 </tr>
 <tr>
  <td rowspan="7">
   PP-StructureV3
  </td>
  <td>
   OCR模型
  </td>
  <td>
   公式识别模型
  </td>
  <td>
   是否启用图表识别模块
  </td>
  <td>
   文本检测max_side_limit
  </td>
 </tr>
 <tr>
  <td>
   Server系列
  </td>
  <td>
   PP-FormulaNet-L
  </td>
  <td>
   ✗
  </td>
  <td>
   4096
  </td>
  <td>
   1.12
  </td>
  <td>
   109.8
  </td>
  <td>
   9.2
  </td>
  <td>
   7.8
  </td>
  <td>
   29.8
  </td>
  <td>
   21.8
  </td>
  <td>
   21.1
  </td>
 </tr>
 <tr>
  <td>
   Server系列
  </td>
  <td>
   PP-FormulaNet-L
  </td>
  <td>
   ✔
  </td>
  <td>
   4096
  </td>
  <td>
   2.76
  </td>
  <td>
   103.7
  </td>
  <td>
   9.0
  </td>
  <td>
   7.7
  </td>
  <td>
   24
  </td>
  <td>
   21.8
  </td>
  <td>
   21.1
  </td>
 </tr>
 <tr>
  <td>
   Mobile系列
  </td>
  <td>
   PP-FormulaNet-L
  </td>
  <td>
   ✗
  </td>
  <td>
   4096
  </td>
  <td>
   1.04
  </td>
  <td>
   110.7
  </td>
  <td>
   9.3
  </td>
  <td>
   7.8
  </td>
  <td>
   22
  </td>
  <td>
   12.2
  </td>
  <td>
   12.1
  </td>
 </tr>
 <tr>
  <td>
   Server系列
  </td>
  <td>
   PP-FormulaNet-M
  </td>
  <td>
   ✗
  </td>
  <td>
   4096
  </td>
  <td>
   0.95
  </td>
  <td>
   111.4
  </td>
  <td>
   9.1
  </td>
  <td>
   7.8
  </td>
  <td>
   28.1
  </td>
  <td>
   21.8
  </td>
  <td>
   21.0
  </td>
 </tr>
 <tr>
  <td>
   Mobile系列
  </td>
  <td>
   PP-FormulaNet-M
  </td>
  <td>
   ✗
  </td>
  <td>
   4096
  </td>
  <td>
   0.89
  </td>
  <td>
   112.1
  </td>
  <td>
   9.2
  </td>
  <td>
   7.8
  </td>
  <td>
   18.5
  </td>
  <td>
   11.4
  </td>
  <td>
   11.2
  </td>
 </tr>
 <tr>
  <td>
   Mobile系列
  </td>
  <td>
   PP-FormulaNet-M
  </td>
  <td>
   ✗
  </td>
  <td>
   1200
  </td>
  <td>
   0.64
  </td>
  <td>
   113.5
  </td>
  <td>
   10.2
  </td>
  <td>
   8.5
  </td>
  <td>
   23.7
  </td>
  <td>
   11.4
  </td>
  <td>
   11.2
  </td>
 </tr>
 <tr>
  <td>
   MinerU
  </td>
  <td colspan="4">
   -
  </td>
  <td>
   1.06
  </td>
  <td>
   168.3
  </td>
  <td>
   18.3
  </td>
  <td>
   16.8
  </td>
  <td>
   27.5
  </td>
  <td>
   76.9
  </td>
  <td>
   14.8
  </td>
 </tr>
</table>

## 3.2 服务化部署

服务化部署测试基于 NVIDIA A100 + Intel Xeon Platinum 8350C 环境，测试数据为 1500 张图像，包含表格、公式、印章、图表等元素。

<table>
 <tbody>
  <tr> 
   <td>实例数</td> 
   <td>并发请求数</td> 
   <td>吞吐</td> 
   <td>平均时延（s）</td> 
   <td>成功请求数/总请求数</td> 
  </tr> 
  <tr"> 
   <td>4卡 ✖️ 1实例/卡</td>
   <td>4</td> 
   <td>1.69</td> 
   <td>2.36</td> 
   <td>100%</td> 
  </tr> 
  <tr"> 
   <td>4卡 ✖️ 4实例/卡</td> 
   <td>16</td> 
   <td>4.05</td> 
   <td>3.87</td> 
   <td>100%</td> 
  </tr> 
 </tbody>
</table>

## 3.3 产线基准测试数据

<details>
<summary>点击展开/折叠表格</summary>

<table border="1">
<tr><th>流水线配置</th><th>硬件</th><th>平均推理时间 (s)</th><th>峰值CPU利用率 (%)</th><th>平均CPU利用率 (%)</th><th>峰值主机内存 (MB)</th><th>平均主机内存 (MB)</th><th>峰值GPU利用率 (%)</th><th>平均GPU利用率 (%)</th><th>峰值设备内存 (MB)</th><th>平均设备内存 (MB)</th></tr>
<tr>
<td rowspan="5">PP_StructureV3-default</td>
<td>Intel 8350C + A100</td>
<td>1.38</td>
<td>1384.60</td>
<td>113.26</td>
<td>5781.59</td>
<td>3431.21</td>
<td>100</td>
<td>32.79</td>
<td>37370.00</td>
<td>34165.68</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>2.38</td>
<td>608.70</td>
<td>109.96</td>
<td>6388.91</td>
<td>3737.19</td>
<td>100</td>
<td>39.08</td>
<td>26824.00</td>
<td>24581.61</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>1.36</td>
<td>744.30</td>
<td>112.82</td>
<td>6199.01</td>
<td>3865.78</td>
<td>100</td>
<td>43.81</td>
<td>35132.00</td>
<td>32077.12</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.74</td>
<td>418.50</td>
<td>105.96</td>
<td>6138.25</td>
<td>3503.41</td>
<td>100</td>
<td>48.54</td>
<td>18536.00</td>
<td>18353.93</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>3.70</td>
<td>434.40</td>
<td>105.45</td>
<td>6865.87</td>
<td>3595.68</td>
<td>100</td>
<td>71.92</td>
<td>13970.00</td>
<td>12668.58</td>
</tr>
<tr>
<td rowspan="3">PP_StructureV3-pp</td>
<td>Intel 8350C + A100</td>
<td>3.50</td>
<td>679.30</td>
<td>105.96</td>
<td>13850.20</td>
<td>5146.50</td>
<td>100</td>
<td>14.01</td>
<td>37656.00</td>
<td>34716.95</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>5.03</td>
<td>494.20</td>
<td>105.63</td>
<td>13542.94</td>
<td>4833.55</td>
<td>100</td>
<td>20.36</td>
<td>29402.00</td>
<td>26607.92</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>3.17</td>
<td>481.50</td>
<td>105.13</td>
<td>14179.97</td>
<td>5608.80</td>
<td>100</td>
<td>19.35</td>
<td>35454.00</td>
<td>32512.19</td>
</tr>
<tr>
<td rowspan="2">PP_StructureV3-full</td>
<td>Intel 8350C + A100</td>
<td>8.92</td>
<td>697.30</td>
<td>102.88</td>
<td>13777.07</td>
<td>4573.65</td>
<td>100</td>
<td>18.39</td>
<td>38776.00</td>
<td>37554.09</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>13.12</td>
<td>437.40</td>
<td>102.36</td>
<td>13974.00</td>
<td>4484.00</td>
<td>100</td>
<td>17.50</td>
<td>29878.00</td>
<td>28733.59</td>
</tr>
<tr>
<td rowspan="5">PP_StructureV3-seal</td>
<td>Intel 8350C + A100</td>
<td>1.39</td>
<td>747.50</td>
<td>112.55</td>
<td>5788.79</td>
<td>3742.03</td>
<td>100</td>
<td>33.81</td>
<td>38966.00</td>
<td>35832.44</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>2.44</td>
<td>630.10</td>
<td>110.18</td>
<td>6343.39</td>
<td>3725.98</td>
<td>100</td>
<td>42.23</td>
<td>28078.00</td>
<td>25834.70</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>1.40</td>
<td>792.20</td>
<td>113.63</td>
<td>6673.60</td>
<td>4417.34</td>
<td>100</td>
<td>46.33</td>
<td>35530.00</td>
<td>32516.87</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.75</td>
<td>422.40</td>
<td>106.08</td>
<td>6068.87</td>
<td>3973.49</td>
<td>100</td>
<td>50.12</td>
<td>19630.00</td>
<td>18374.37</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>3.76</td>
<td>400.30</td>
<td>105.10</td>
<td>6296.28</td>
<td>3651.42</td>
<td>100</td>
<td>72.57</td>
<td>14304.00</td>
<td>13268.36</td>
</tr>
<tr>
<td rowspan="4">PP_StructureV3-chart</td>
<td>Intel 8350C + A100</td>
<td>7.70</td>
<td>746.80</td>
<td>102.69</td>
<td>6355.58</td>
<td>4006.48</td>
<td>100</td>
<td>22.38</td>
<td>37380.00</td>
<td>36730.73</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>10.58</td>
<td>599.20</td>
<td>102.51</td>
<td>5754.14</td>
<td>3333.78</td>
<td>100</td>
<td>21.99</td>
<td>26820.00</td>
<td>26253.70</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>8.03</td>
<td>413.30</td>
<td>101.31</td>
<td>6473.29</td>
<td>3689.84</td>
<td>100</td>
<td>26.19</td>
<td>18540.00</td>
<td>18494.69</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>11.69</td>
<td>460.90</td>
<td>101.85</td>
<td>6503.12</td>
<td>3524.06</td>
<td>100</td>
<td>46.81</td>
<td>13966.00</td>
<td>12481.94</td>
</tr>
<tr>
<td rowspan="5">PP_StructureV3-notable</td>
<td>Intel 8350C + A100</td>
<td>1.24</td>
<td>738.30</td>
<td>110.45</td>
<td>5638.16</td>
<td>3278.30</td>
<td>100</td>
<td>35.32</td>
<td>30320.00</td>
<td>27026.17</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>2.24</td>
<td>452.40</td>
<td>107.79</td>
<td>5579.15</td>
<td>3635.95</td>
<td>100</td>
<td>43.00</td>
<td>23098.00</td>
<td>20684.43</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>1.18</td>
<td>989.00</td>
<td>107.71</td>
<td>6041.76</td>
<td>4024.76</td>
<td>100</td>
<td>50.67</td>
<td>33780.00</td>
<td>29733.15</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.58</td>
<td>225.00</td>
<td>102.56</td>
<td>5518.10</td>
<td>3333.08</td>
<td>100</td>
<td>49.90</td>
<td>21532.00</td>
<td>18567.99</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>3.40</td>
<td>413.30</td>
<td>103.58</td>
<td>5874.88</td>
<td>3662.49</td>
<td>100</td>
<td>76.82</td>
<td>13764.00</td>
<td>11890.62</td>
</tr>
<tr>
<td rowspan="7">PP_StructureV3-noformula</td>
<td>Intel 6271C</td>
<td>7.85</td>
<td>1172.50</td>
<td>964.70</td>
<td>17739.00</td>
<td>11101.02</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>8.83</td>
<td>1053.50</td>
<td>970.64</td>
<td>15463.48</td>
<td>9408.19</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.84</td>
<td>788.60</td>
<td>124.25</td>
<td>6246.39</td>
<td>3674.32</td>
<td>100</td>
<td>30.57</td>
<td>40084.00</td>
<td>37358.45</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>1.42</td>
<td>606.20</td>
<td>115.53</td>
<td>7015.57</td>
<td>3707.03</td>
<td>100</td>
<td>35.63</td>
<td>29540.00</td>
<td>27620.28</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.87</td>
<td>644.10</td>
<td>119.23</td>
<td>6895.76</td>
<td>4222.85</td>
<td>100</td>
<td>50.00</td>
<td>36878.00</td>
<td>34104.59</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>1.03</td>
<td>377.50</td>
<td>106.87</td>
<td>5819.88</td>
<td>3830.19</td>
<td>100</td>
<td>42.87</td>
<td>19340.00</td>
<td>17550.94</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>2.02</td>
<td>430.20</td>
<td>109.21</td>
<td>6600.62</td>
<td>3824.18</td>
<td>100</td>
<td>65.75</td>
<td>14332.00</td>
<td>12712.18</td>
</tr>
<tr>
<td rowspan="9">PP_StructureV3-lightweight</td>
<td>Intel 6271C</td>
<td>4.36</td>
<td>1189.70</td>
<td>995.78</td>
<td>14000.50</td>
<td>9374.97</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C</td>
<td>3.74</td>
<td>1049.60</td>
<td>967.77</td>
<td>12960.96</td>
<td>7644.25</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Hygon 7490 + P800</td>
<td>0.86</td>
<td>572.20</td>
<td>120.84</td>
<td>8290.49</td>
<td>3569.44</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 8350C + A100</td>
<td>0.61</td>
<td>823.40</td>
<td>126.25</td>
<td>9258.22</td>
<td>3776.63</td>
<td>52</td>
<td>18.95</td>
<td>7456.00</td>
<td>7131.95</td>
</tr>
<tr>
<td>Intel 6271C + V100</td>
<td>1.07</td>
<td>686.80</td>
<td>116.70</td>
<td>9381.75</td>
<td>4126.28</td>
<td>58</td>
<td>22.92</td>
<td>8450.00</td>
<td>8083.30</td>
</tr>
<tr>
<td>Intel 8563C + H20</td>
<td>0.46</td>
<td>999.00</td>
<td>122.21</td>
<td>9734.78</td>
<td>4516.40</td>
<td>61</td>
<td>24.41</td>
<td>7524.00</td>
<td>7167.52</td>
</tr>
<tr>
<td>Intel 8350C + A10</td>
<td>0.70</td>
<td>355.40</td>
<td>111.51</td>
<td>9415.45</td>
<td>4094.06</td>
<td>89</td>
<td>30.85</td>
<td>7248.00</td>
<td>6927.58</td>
</tr>
<tr>
<td>M4</td>
<td>12.22</td>
<td>223.60</td>
<td>107.35</td>
<td>9531.22</td>
<td>7884.61</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
<td>N/A</td>
</tr>
<tr>
<td>Intel 6271C + T4</td>
<td>1.13</td>
<td>461.40</td>
<td>112.16</td>
<td>7923.09</td>
<td>3837.31</td>
<td>85</td>
<td>41.67</td>
<td>8218.00</td>
<td>7902.04</td>
</tr>
</table>


<table border="1">
<tr><th>Pipeline configuration</th><th>description</th></tr>
<tr>
<td>PP_StructureV3-default</td>
<td>默认配置</td>
</tr>
<tr>
<td>PP_StructureV3-pp</td>
<td>默认配置基础上，开启文档图像预处理</td>
</tr>
<tr>
<td>PP_StructureV3-full</td>
<td>默认配置基础上，开启文档图像预处理和图表解析</td>
</tr>
<tr>
<td>PP_StructureV3-seal</td>
<td>默认配置基础上，开启印章文本识别</td>
</tr>
<tr>
<td>PP_StructureV3-chart</td>
<td>默认配置基础上，开启文档图表解析</td>
</tr>
<tr>
<td>PP_StructureV3-notable</td>
<td>默认配置基础上，关闭表格识别</td>
</tr>
<tr>
<td>PP_StructureV3-noformula</td>
<td>默认配置基础上，关闭公式识别</td>
</tr>
<tr>
<td>PP_StructureV3-lightweight</td>
<td>默认配置基础上，将所有任务模型都换成最轻量版本</td>
</tr>
</table>
</details>


* 测试环境：
    * PaddlePaddle 3.1.0、CUDA 11.8、cuDNN 8.9
    * PaddleX @ develop (f1eb28e23cfa54ce3e9234d2e61fcb87c93cf407)
    * Docker image: ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.1.0-gpu-cuda11.8-cudnn8.9
* 测试数据：
    * 测试数据包含表格、印章、公式、图表的280张图像。
* 测试策略：
    * 使用 20 个样本进行预热，然后对整个数据集重复 1 次以进行速度性能测试。
* 备注：
    * 由于我们没有收集NPU和XPU的设备内存数据，因此表中相应位置的数据标记为N/A。

# 四、PP-StructureV3 Demo示例

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-StructureV3/algorithm_ppstructurev3_demo.png" width="600"/>
</div> 

<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex%2FPaddleX3.0%2Fdoc_images%2FPP-StructureV3%2Falgorithm_ppstructurev3_demo.pdf">更多示例</a>

# 五、使用方法和常见问题

**Q:默认模型是什么配置，如果需要更高精度、更快速度、或者更小显存，应该调哪些参数或者更换哪些模型，对结果影响大概有多大？**

**A:** 默认模型均采用了了各个模块参数量最大的模型，3.3 章节中展示了不同的模型选择对于显存和推理速度的影响。可以根据设备情况和样本难易程度选择合适的模型。另外，在 Python API 或 CLI 设置 device 为<设备类型>:<设备编号1>,<设备编号2>...（例如gpu:0,1,2,3）可实现多卡并行推理。如果内置的多卡并行推理功能提速效果仍不满足预期，可参考多进程并行推理示例代码，结合具体场景进行进一步优化：[多进程并行推理](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/instructions/parallel_inference.html)。

---

**Q: PP-StructureV3 是否可以在 CPU 上运行？**

**A:** PP-StructureV3 虽然更推荐在 GPU 环境下进行推理，但也支持在 CPU 上运行。得益于多种配置选项及对轻量级模型的充分优化，在仅有 CPU 环境时，用户可以参考 3.3 节选择轻量化配置进行推理。例如，在 Intel 8350C CPU 上，每张图片的推理时间约为 3.74 秒。

---

**Q: 如何将 PP-StructureV3 集成到自己的项目中？**

**A:**  
- 对于 Python 项目，可以直接使用 PaddleOCR 的 Python API 完成集成。  
- 对于其他编程语言，建议通过服务化部署方式集成。PaddleOCR 支持包括 C++、C#、Java、Go、PHP 等多种语言的客户端调用方式，具体集成方法可参考 [官方文档](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PP-StructureV3.html#3)。  
- 如果需要与大模型进行交互，PaddleOCR 还提供了 MCP 服务，详细说明可参考 [MCP 服务器](https://www.paddleocr.ai/latest/version3.x/deployment/mcp_server.html)。

---

**Q:服务化部署可以并发处理请求吗？**

**A:** 对于基础服务化部署方案，服务同一时间只处理一个请求，该方案主要用于快速验证、打通开发链路，或者用在不需要并发请求的场景；对于高稳定性服务化部署方案，服务默认在同一时间只处理一个请求，但用户可以参考服务化部署指南，通过调整配置实现水平扩展，以使服务同时处理多个请求。

---

**Q: 服务化部署如何降低时延、提升吞吐？**

**A:** PaddleOCR 提供的2种服务化部署方案，无论使用哪一种方案，都可以通过启用高性能推理插件提升模型推理速度，从而降低处理时延。此外，对于高稳定性服务化部署方案，通过调整服务配置，设置多个实例，也可以充分利用部署机器的资源，有效提升吞吐。高稳定性服务化部署方案调整配置可以参考[文档](https://paddlepaddle.github.io/PaddleX/latest/pipeline_deploy/serving.html#22)。