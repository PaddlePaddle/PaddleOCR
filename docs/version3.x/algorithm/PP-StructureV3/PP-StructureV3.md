# 一、PP-StructureV3 简介
**PP-StructureV3** 产线在通用版面解析v1产线的基础上，强化了版面区域检测、表格识别、公式识别的能力，增加了图表理解和多栏阅读顺序的恢复能力、结果转换 Markdown 文件的能力，在多种文档数据中，表现优异，可以处理较复杂的文档数据。本产线同时提供了灵活的服务化部署方式，支持在多种硬件上使用多种编程语言调用。不仅如此，本产线也提供了二次开发的能力，您可以基于本产线在您自己的数据集上训练调优，训练后的模型也可以无缝集成。

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-StructureV3/algorithm_ppstructurev3.png" width="600"/>
</div>

# 二、关键指标
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
   <td><b>0.147</b></td> 
   <td><b>0.212</b></td> 
   <td>0.059</td> 
   <td><b>0.09</b></td> 
   <td>0.295</td> 
   <td>0.535</td> 
   <td>0.159</td> 
   <td><b>0.109</b></td> 
   <td>0.075</td> 
   <td><b>0.114</b></td> 
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

以上部分数据出自：
* <a href="https://github.com/opendatalab/OmniDocBench">OmniDocBench</a>
* <a href="https://arxiv.org/abs/2412.07626">OmniDocBench: Benchmarking Diverse PDF Document Parsing with Comprehensive Annotations</a>


# 三、端到端推理benchmark
## 3.1 测试条件

* Paddle 3.0正式版
* PaddleOCR 3.0.0正式版
* MinerU 1.3.10
* CUDA 11.8
* cuDNN 8.9

## 3.2 测试数据
1. 本地推理

测试硬件：NVIDIA Tesla V100 + Intel Xeon Gold 6271C

<table style="width:585pt">
 <colgroup>
  <col width="87" span="9" style="width:65pt"> 
 </colgroup>
 <tbody>
  <tr> 
   <td></td> 
   <td>产线配置</td> 
   <td>平均每页耗时（s）</td> 
   <td>平均CPU利用率（%）</td> 
   <td>峰值RAM用量（MB）</td> 
   <td>平均RAM用量（MB）</td> 
   <td>平均GPU利用率（%）</td> 
   <td>峰值VRAM用量（MB）</td> 
   <td>平均VRAM用量（MB）</td> 
  </tr> 
  <tr> 
   <td rowspan="6">PP-StructureV3</td> 
   <td>基础配置</td> 
   <td>1.77</td> 
   <td>111.4</td> 
   <td>6822.4</td> 
   <td>5278.2</td> 
   <td>38.9</td> 
   <td>17403</td> 
   <td>16909.3</td> 
  </tr> 
  <tr> 
   <td>使用图表识别</td> 
   <td>4.09</td> 
   <td>105.3</td> 
   <td>5628</td> 
   <td>4085.1</td> 
   <td>24.7</td> 
   <td>17403</td> 
   <td>17030.9</td> 
  </tr> 
  <tr> 
   <td>使用 PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec 轻量模型</td> 
   <td>1.56</td> 
   <td>113.7</td> 
   <td>6712.9</td> 
   <td>5052</td> 
   <td>29.1</td> 
   <td>10929</td> 
   <td>10840.7</td> 
  </tr> 
  <tr> 
   <td>使用 PP-FormulaNet-M 轻量公式识别模型</td> 
   <td>1.42</td> 
   <td>112.9</td> 
   <td>6944.1</td> 
   <td>5193.6</td> 
   <td>38</td> 
   <td>16390</td> 
   <td>15840</td> 
  </tr> 
  <tr> 
   <td>使用 PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec + PP-FormulaNet-M</td> 
   <td>1.15</td> 
   <td>114.8</td> 
   <td>6666.5</td> 
   <td>5105.4</td> 
   <td>26.1</td> 
   <td>8606</td> 
   <td>8517.2</td> 
  </tr> 
  <tr> 
   <td>使用 PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec + PP-FormulaNet-M,文本检测模型输入最大尺寸设置为 1200</td> 
   <td>0.99</td> 
   <td>113</td> 
   <td>7172.9</td> 
   <td>5686.4</td> 
   <td>29.2</td> 
   <td>8776</td> 
   <td>8680.8</td> 
  </tr> 
  <tr data-slate-node="element"> 
   <td data-gramm="false" data-slate-node="element" data-slate-type="table-cell" data-slate-render-key="PpiUjdl1" cellindex="0">MinerU</td> 
   <td>-</td> 
   <td>1.57</td> 
   <td>142.9</td> 
   <td>13655.8</td> 
   <td>12083</td> 
   <td>43.3</td> 
   <td>32406</td> 
   <td>9915.4</td> 
  </tr> 
 </tbody>
</table>

测试硬件：NVIDIA A100 + Intel Xeon Platinum 8350C

<table> 
 <colgroup>
  <col> 
 </colgroup>
 <tbody>
  <tr> 
   <td></td> 
   <td>Pipeline Configurations</td> 
   <td>Average time per page (s)</td> 
   <td>Average CPU (%)</td> 
   <td>Peak RAM Usage (MB)</td> 
   <td>Average RAM Usage (MB)</td> 
   <td>Average GPU (%)</td> 
   <td>Peak VRAM Usage (MB)</td> 
   <td>Average VRAM Usage (MB)</td> 
  </tr> 
  <tr> 
   <td rowspan="6">PP-StructureV3</td> 
   <td>Basic</td> 
   <td>1.12</td> 
   <td>109.8</td> 
   <td>9418.3</td> 
   <td>7977.9</td> 
   <td>29.8</td> 
   <td>22294</td> 
   <td>21638.4</td> 
  </tr> 
  <tr> 
   <td>Use chart recognition pipeline</td> 
   <td>2.76</td> 
   <td>103.7</td> 
   <td>9253.6</td> 
   <td>7840.6</td> 
   <td>24</td> 
   <td>22298</td> 
   <td>21555.3</td> 
  </tr> 
  <tr> 
   <td>Use PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec</td> 
   <td>1.04</td> 
   <td>110.7</td> 
   <td>9520.8</td> 
   <td>8034.3</td> 
   <td>22</td> 
   <td>12490</td> 
   <td>12383.1</td> 
  </tr> 
  <tr> 
   <td>Use PP-FormulaNet_plus-M</td> 
   <td>0.95</td> 
   <td>111.4</td> 
   <td>9272.9</td> 
   <td>7939.9</td> 
   <td>28.1</td> 
   <td>22350</td> 
   <td>21498.4</td> 
  </tr> 
  <tr> 
   <td>Use PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec + PP-FormulaNet_plus-M</td> 
   <td>0.89</td> 
   <td>112.1</td> 
   <td>9457.2</td> 
   <td>8031.5</td> 
   <td>18.5</td> 
   <td>11642</td> 
   <td>11433.6</td> 
  </tr> 
  <tr> 
   <td>Use PP-OCRv5_mobile_det + PP-OCRv5_mobile_rec + PP-FormulaNet_plus-M, and max length of text detection set to 1200</td> 
   <td>0.64</td> 
   <td>113.5</td> 
   <td>10401.1</td> 
   <td>8688.8</td> 
   <td>23.7</td> 
   <td>11716</td> 
   <td>11453.9</td> 
  </tr> 
  <tr"> 
   <td>MinerU</td> 
   <td>-</td> 
   <td>1.06</td> 
   <td>168.3</td> 
   <td>18690.4</td> 
   <td>17213.8</td> 
   <td>27.5</td> 
   <td>78760</td> 
   <td>15119</td> 
  </tr> 
 </tbody>
</table>

2. 服务化部署

输入数据：1500张图像，包含表格、公式、印章、图表等元素。
使用基础配置。

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
   <td>4卡*1</td>
   <td>4</td> 
   <td>1.69</td> 
   <td>2.36</td> 
   <td>1</td> 
  </tr> 
  <tr"> 
   <td>4卡*4</td> 
   <td>16</td> 
   <td>4.05</td> 
   <td>3.87</td> 
   <td>1</td> 
  </tr> 
 </tbody>
</table>

# 四、PP-StructureV3 Demo示例

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-StructureV3/algorithm_ppstructurev3_demo.png" width="600"/>
</div>

<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex%2FPaddleX3.0%2Fdoc_images%2FPP-StructureV3%2Falgorithm_ppstructurev3_demo.pdf">更多示例</a>

# 五、使用方法和常见问题

1. 默认模型是什么配置，如果需要更高精度、更快速度、或者更小显存，应该调哪些参数或者更换哪些模型，对结果影响大概有多大？

在“使用轻量OCR模型+轻量公式模型，文本检测max 1200”的基础上，将产线配置文件中的use_chart_recognition设置为False，不加载图表识别模型，可以进一步减少显存用量。在V100测试环境中，峰值和平均显存用量分别从8776.0 MB和8680.8 MB降低到6118.0 MB和6016.7 MB；在A100测试环境中，峰值和平均显存用量分别从11716.0 MB和11453.9 MB降低到9850.0 MB和9593.5 MB。
在Python API或CLI设置device为<设备类型>:<设备编号1>,<设备编号2>...（例如gpu:0,1,2,3）可实现多卡并行推理。如果内置的多卡并行推理功能提速效果仍不满足预期，可参考多进程并行推理示例代码，结合具体场景进行进一步优化：[多进程并行推理](https://github.com/PaddlePaddle/PaddleX/blob/develop/docs/pipeline_usage/instructions/parallel_inference.md#%E5%A4%9A%E8%BF%9B%E7%A8%8B%E5%B9%B6%E8%A1%8C%E6%8E%A8%E7%90%86%E7%A4%BA%E4%BE%8B)。

2. 服务化部署的常见问题

（1）服务可以并发处理请求吗？

对于基础服务化部署方案，服务同一时间只处理一个请求，该方案主要用于快速验证、打通开发链路，或者用在不需要并发请求的场景；

对于高稳定性服务化部署方案，服务默认在同一时间只处理一个请求，但用户可以参考服务化部署指南，通过调整配置实现水平扩展，以使服务同时处理多个请求。

（2）如何降低时延、提升吞吐？

无论使用哪一种服务化部署方案，都可以通过启用高性能推理插件提升模型推理速度，从而降低处理时延。

此外，对于高稳定性服务化部署方案，通过调整服务配置，设置多个实例，也可以充分利用部署机器的资源，有效提升吞吐。
