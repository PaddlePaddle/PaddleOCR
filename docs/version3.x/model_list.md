---
comments: true
---

# PaddleOCR模型列表（CPU/GPU）

PaddleOCR 内置了多条产线，每条产线都包含了若干模块，每个模块包含若干模型，具体使用哪些模型，您可以根据下边的 benchmark 数据来选择。如您更考虑模型精度，请选择精度较高的模型，如您更考虑模型推理速度，请选择推理速度较快的模型，如您更考虑模型存储大小，请选择存储大小较小的模型。

## [文本检测模块](./module_usage/text_detection.md)

<table>
<thead>
<tr>
<th>模型</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv5_server_det</td>
<td>83.8</td>
<td>89.55 / 70.19</td>
<td>383.15 / 383.15</td>
<td>101</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv5_server_det.yaml">PP-OCRv5_server_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv5_mobile_det</td>
<td>79.0</td>
<td>10.67 / 6.36</td>
<td>57.77 / 28.15</td>
<td>4.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv5_mobile_det.yaml">PP-OCRv5_mobile_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv4_server_det</td>
<td>82.56</td>
<td>127.82 / 98.87</td>
<td>585.95 / 489.77</td>
<td>109</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv4_server_det.yaml">PP-OCRv4_server_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv4_mobile_det</td>
<td>63.8</td>
<td>9.87 / 4.17</td>
<td>56.60 / 20.79</td>
<td>4.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv4_mobile_det.yaml">PP-OCRv4_mobile_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv3_mobile_det</td>
<td>78.68</td>
<td>9.90 / 3.60</td>
<td>41.93 / 20.76</td>
<td>2.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv3_mobile_det.yaml">PP-OCRv3_mobile_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_mobile_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_mobile_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv3_server_det</td>
<td>80.11</td>
<td>119.50 / 75.00</td>
<td>379.35 / 318.35</td>
<td>102.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_detection/PP-OCRv3_server_det.yaml">PP-OCRv3_server_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv3_server_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv3_server_det_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的中英文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 593 张图片。</b>

## [印章文本检测模块](./module_usage/seal_text_detection.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>检测Hmean（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>PP-OCRv4_mobile_seal_det</td>
<td>96.36</td>
<td>9.70 / 3.56</td>
<td>50.38 / 19.64</td>
<td>4.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/seal_text_detection/PP-OCRv4_mobile_seal_det.yaml">PP-OCRv4_mobile_seal_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_seal_det_pretrained.pdparams">训练模型</a></td></tr>
<tr>
<td>PP-OCRv4_server_seal_det</td>
<td>98.40</td>
<td>124.64 / 91.57</td>
<td>545.68 / 439.86</td>
<td>109</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/seal_text_detection/PP-OCRv4_server_seal_det.yaml">PP-OCRv4_server_seal_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_seal_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_seal_det_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的印章数据集，包含500印章图像。</b>

## [文本识别模块](./module_usage/text_recognition.md)

* <b>中文识别模型</b>
<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>PP-OCRv5_server_rec</td>
<td>86.38</td>
<td>8.46 / 2.36</td>
<td>31.21 / 31.21</td>
<td>81</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv5_server_rec.yaml">PP-OCRv5_server_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv5_mobile_rec</td>
<td>81.29</td>
<td>5.43 / 1.46</td>
<td>21.20 / 5.32</td>
<td>16</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv5_mobile_rec.yaml">PP-OCRv5_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv4_server_rec_doc</td>
<td>86.58</td>
<td>8.69 / 2.78</td>
<td>37.93 / 37.93</td>
<td>182</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_server_rec_doc.yaml">PP-OCRv4_server_rec_doc.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv4_server_rec_doc_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv4_mobile_rec</td>
<td>78.74</td>
<td>5.26 / 1.12</td>
<td>17.48 / 3.61</td>
<td>10.5</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_mobile_rec.yaml">PP-OCRv4_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_mobile_rec_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv4_server_rec</td>
<td>85.19</td>
<td>8.75 / 2.49</td>
<td>36.93 / 36.93</td>
<td>173</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv4_server_rec.yaml">PP-OCRv4_server_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv4_server_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv4_server_rec_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-OCRv3_mobile_rec</td>
<td>72.96</td>
<td>3.89 / 1.16</td>
<td>8.72 / 3.56</td>
<td>10.3</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/PP-OCRv3_mobile_rec.yaml">PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的中文数据集，覆盖街景、网图、文档、手写多个场景，其中文本识别包含 8367 张图片。</b></p>
<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>ch_SVTRv2_rec</td>
<td>68.81</td>
<td>10.38 / 8.31</td>
<td>66.52 / 30.83</td>
<td>80.5</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ch_SVTRv2_rec.yaml">ch_SVTRv2_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_SVTRv2_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_SVTRv2_rec_pretrained.pdparams">训练模型</a></td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>A榜。 </b></p>
<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>ch_RepSVTR_rec</td>
<td>65.07</td>
<td>6.29 / 1.57</td>
<td>20.64 / 5.40</td>
<td>48.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ch_RepSVTR_rec.yaml">ch_RepSVTR_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ch_RepSVTR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ch_RepSVTR_rec_pretrained.pdparams">训练模型</a></td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 <a href="https://aistudio.baidu.com/competition/detail/1131/0/introduction">PaddleOCR算法模型挑战赛 - 赛题一：OCR端到端识别任务</a>B榜。 </b></p>

* <b>英文识别模型</b>
<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>en_PP-OCRv4_mobile_rec</td>
<td> 70.39</td>
<td>4.81 / 1.23</td>
<td>17.20 / 4.18</td>
<td>7.5</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/en_PP-OCRv4_mobile_rec.yaml">en_PP-OCRv4_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv4_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>en_PP-OCRv3_mobile_rec</td>
<td>70.69</td>
<td>3.56 / 0.78</td>
<td>8.44 / 5.78</td>
<td>17.3</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/en_PP-OCRv3_mobile_rec.yaml">en_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
en_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
</table>

<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的英文数据集。 </b></p>

* <b>多语言识别模型</b>
<table>
<tr>
<th>模型</th>
<th>识别 Avg Accuracy(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>korean_PP-OCRv3_mobile_rec</td>
<td>60.21</td>
<td>3.73 / 0.98</td>
<td>8.76 / 2.91</td>
<td>9.6</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/korean_PP-OCRv3_mobile_rec.yaml">korean_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
korean_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>japan_PP-OCRv3_mobile_rec</td>
<td>45.69</td>
<td>3.86 / 1.01</td>
<td>8.62 / 2.92</td>
<td>9.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/japan_PP-OCRv3_mobile_rec.yaml">japan_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
japan_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>chinese_cht_PP-OCRv3_mobile_rec</td>
<td>82.06</td>
<td>3.90 / 1.16</td>
<td>9.24 / 3.18</td>
<td>10.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/chinese_cht_PP-OCRv3_mobile_rec.yaml">chinese_cht_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
chinese_cht_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>te_PP-OCRv3_mobile_rec</td>
<td>95.88</td>
<td>3.59 / 0.81</td>
<td>8.28 / 6.21</td>
<td>8.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/te_PP-OCRv3_mobile_rec.yaml">te_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
te_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>ka_PP-OCRv3_mobile_rec</td>
<td>96.96</td>
<td>3.49 / 0.89</td>
<td>8.63 / 2.77</td>
<td>17.4</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ka_PP-OCRv3_mobile_rec.yaml">ka_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ka_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>ta_PP-OCRv3_mobile_rec</td>
<td>76.83</td>
<td>3.49 / 0.86</td>
<td>8.35 / 3.41</td>
<td>8.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/ta_PP-OCRv3_mobile_rec.yaml">ta_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
ta_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>latin_PP-OCRv3_mobile_rec</td>
<td>76.93</td>
<td>3.53 / 0.78</td>
<td>8.50 / 6.83</td>
<td>8.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/latin_PP-OCRv3_mobile_rec.yaml">latin_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
latin_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>arabic_PP-OCRv3_mobile_rec</td>
<td>73.55</td>
<td>3.60 / 0.83</td>
<td>8.44 / 4.69</td>
<td>17.3</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/arabic_PP-OCRv3_mobile_rec.yaml">arabic_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
arabic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>cyrillic_PP-OCRv3_mobile_rec</td>
<td>94.28</td>
<td>3.56 / 0.79</td>
<td>8.22 / 2.76</td>
<td>8.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/cyrillic_PP-OCRv3_mobile_rec.yaml">cyrillic_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
cyrillic_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
<tr>
<td>devanagari_PP-OCRv3_mobile_rec</td>
<td>96.44</td>
<td>3.60 / 0.78</td>
<td>6.95 / 2.87</td>
<td>8.7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/text_recognition/devanagari_PP-OCRv3_mobile_rec.yaml">devanagari_PP-OCRv3_mobile_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/\
devanagari_PP-OCRv3_mobile_rec_infer.tar">推理模型</a>/<a href="">训练模型</a></td>
</tr>
</table>
<p><b>注：以上精度指标的评估集是 PaddleOCR 自建的多语种数据集。</b></p>

## [公式识别模块](./module_usage/formula_recognition.md)

<table>
<tr>
<th>模型</th>
<th>En-BLEU(%)</th>
<th>Zh-BLEU(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>UniMERNet</td>
<td>85.91</td>
<td>43.50</td>
<td>1311.84 / 1311.84</td>
<td>- / 8288.07</td>
<td>1530</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/UniMERNet.yaml">UniMERNet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UniMERNet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UniMERNet_pretrained.pdparams">训练模型</a></td>
</tr>
<td>PP-FormulaNet-S</td>
<td>87.00</td>
<td>45.71</td>
<td>182.25 / 182.25</td>
<td>- / 254.39</td>
<td>224</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/PP-FormulaNet-S.yaml">PP-FormulaNet-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-S_pretrained.pdparams">训练模型</a></td>
</tr>
<td>PP-FormulaNet-L</td>
<td>90.36</td>
<td>45.78</td>
<td>1482.03 / 1482.03</td>
<td>- / 3131.54</td>
<td>695</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/PP-FormulaNet-L.yaml">PP-FormulaNet-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet-L_pretrained.pdparams">训练模型</a></td>
</tr>
<td>PP-FormulaNet_plus-S</td>
<td>88.71</td>
<td>53.32</td>
<td>179.20 / 179.20</td>
<td>- / 260.99</td>
<td>248</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/PP-FormulaNet_plus-S.yaml">PP-FormulaNet_plus-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-S_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-FormulaNet_plus-M</td>
<td>91.45</td>
<td>89.76</td>
<td>1040.27 / 1040.27</td>
<td>- / 1615.80</td>
<td>592</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/PP-FormulaNet_plus-M.yaml">PP-FormulaNet_plus-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-M_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-FormulaNet_plus-L</td>
<td>92.22</td>
<td>90.64</td>
<td>1476.07 / 1476.07</td>
<td>- / 3125.58</td>
<td>698</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/PP-FormulaNet_plus-L.yaml">PP-FormulaNet_plus-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-FormulaNet_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-FormulaNet_plus-L_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>LaTeX_OCR_rec</td>
<td>74.55</td>
<td>39.96</td>
<td>1088.89 / 1088.89</td>
<td>- / -</td>
<td>99</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/formula_recognition/LaTeX_OCR_rec.yaml">LaTeX_OCR_rec.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/LaTeX_OCR_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/LaTeX_OCR_rec_pretrained.pdparams">训练模型</a></td>
</tr>
</table>
<b>注：以上精度指标测量自 PaddleX 内部自建公式识别测试集。LaTeX_OCR_rec在LaTeX-OCR公式识别测试集的BLEU score为 0.8821。</b>

## [表格结构识别模块](./module_usage/table_structure_recognition.md)

<table>
<tr>
<th>模型</th>
<th>精度（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>SLANet</td>
<td>59.52</td>
<td>23.96 / 21.75</td>
<td>- / 43.12</td>
<td>6.9</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANet.yaml">SLANet.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>SLANet_plus</td>
<td>63.69</td>
<td>23.43 / 22.16</td>
<td>- / 41.80</td>
<td>6.9</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANet_plus.yaml">SLANet_plus.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANet_plus_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>SLANeXt_wired</td>
<td rowspan="2">69.65</td>
<td rowspan="2">85.92 / 85.92</td>
<td rowspan="2">- / 501.66</td>
<td rowspan="2">351</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANeXt_wired.yaml">SLANeXt_wired.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wired_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wired_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>SLANeXt_wireless</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_structure_recognition/SLANeXt_wireless.yaml">SLANeXt_wireless.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/SLANeXt_wireless_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANeXt_wireless_pretrained.pdparams">训练模型</a></td>
</tr>
</table>
<b>注：以上精度指标测量自 PaddleX 内部自建高难度中文表格识别数据集。</b>


## [表格单元格检测模块](./module_usage/table_cells_detection.md)

<table>
<tr>
<th>模型</th>
<th>mAP(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>RT-DETR-L_wired_table_cell_det</td>
<td rowspan="2">82.7</td>
<td rowspan="2">33.47 / 27.02</td>
<td rowspan="2">402.55 / 256.56</td>
<td rowspan="2">124</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_cells_detection/RT-DETR-L_wired_table_cell_det.yaml">RT-DETR-L_wired_table_cell_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wired_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wired_table_cell_det_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>RT-DETR-L_wireless_table_cell_det</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_cells_detection/RT-DETR-L_wireless_table_cell_det.yaml">RT-DETR-L_wireless_table_cell_det.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-L_wireless_table_cell_det_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-L_wireless_table_cell_det_pretrained.pdparams">训练模型</a></td>
</tr>
</table>
<p><b>注：以上精度指标测量自 PaddleX 内部自建表格单元格检测数据集。</b></p>

## [表格分类模块](./module_usage/table_classification.md)

<table>
<tr>
<th>模型</th>
<th>Top1 Acc(%)</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>PP-LCNet_x1_0_table_cls</td>
<td>94.2</td>
<td>2.62 / 0.60</td>
<td>3.17 / 1.14</td>
<td>6.6</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/table_classification/PP-LCNet_x1_0_table_cls.yaml">PP-LCNet_x1_0_table_cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_table_cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_table_cls_pretrained.pdparams">训练模型</a></td>
</tr>
</table>
<p><b>注：以上精度指标测量自 PaddleX 内部自建表格分类数据集。</b></p>

## [文本图像矫正模块](./module_usage/text_image_unwarping.md)
<table>
<thead>
<tr>
<th>模型名称</th>
<th>CER</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml 文件</th>
<th>模型下载链接</th></tr>
</thead>
<tbody>
<tr>
<td>UVDoc</td>
<td>0.179</td>
<td>19.05 / 19.05</td>
<td>- / 869.82</td>
<td>30.3</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/image_unwarping/UVDoc.yaml">UVDoc.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/UVDoc_pretrained.pdparams">训练模型</a></td></tr>
</tbody>
</table>
<b>注：以上精度指标测量自 </b><b>PaddleX自建的图像矫正数据集</b><b>。</b>

## [版面区域检测模块](./module_usage/layout_detection.md)

* <b>版面检测模型，包含20个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、表格、图和表标题（图标题、表格标题和图表标题）、印章、图表、侧栏文本和参考文献内容</b>
<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout_plus-L</td>
<td>83.2</td>
<td>53.03 / 17.23</td>
<td>634.62 / 378.32</td>
<td>126.01 </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PP-DocLayout_plus-L.yaml">PP-DocLayout_plus-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout_plus-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout_plus-L_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>

<b>注：以上精度指标的评估集是自建的版面区域检测数据集，包含中英文论文、杂志、报纸、研报、PPT、试卷、课本等 1300 张文档类型图片。</b>

* <b>文档图像版面子模块检测，包含1个 版面区域 类别，能检测多栏的报纸、杂志的每个子文章的文本区域:</b>
<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocBlockLayout</td>
<td>95.9</td>
<td>34.60 / 28.54</td>
<td>506.43 / 256.83</td>
<td>123.92</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PP-DocBlockLayout.yaml">PP-DocBlockLayout.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBlockLayout_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocBlockLayout_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>

<b>注：以上精度指标的评估集是自建的版面子区域检测数据集，包含中英文论文、杂志、报纸、研报、PPT、试卷、课本等 1000 张文档类型图片。</b>


* <b>版面检测模型，包含23个常见的类别：文档标题、段落标题、文本、页码、摘要、目录、参考文献、脚注、页眉、页脚、算法、公式、公式编号、图像、图表标题、表格、表格标题、印章、图表标题、图表、页眉图像、页脚图像、侧栏文本</b>
<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-DocLayout-L</td>
<td>90.4</td>
<td>33.59 / 33.59</td>
<td>503.01 / 251.08</td>
<td>123.76 </td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PP-DocLayout-L.yaml">PP-DocLayout-L.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-L_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-L_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-DocLayout-M</td>
<td>75.2</td>
<td>13.03 / 4.72</td>
<td>43.39 / 24.44</td>
<td>22.578</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PP-DocLayout-M.yaml">PP-DocLayout-M.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-M_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-M_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PP-DocLayout-S</td>
<td>70.9</td>
<td>11.54 / 3.86</td>
<td>18.53 / 6.29</td>
<td>4.834</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PP-DocLayout-S.yaml">PP-DocLayout-S.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayout-S_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-DocLayout-S_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>

<b>注：以上精度指标的评估集是自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 500 张文档类型图片。</b>

* <b>表格版面检测模型</b>
<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x_table</td>
<td>97.5</td>
<td>9.57 / 6.63</td>
<td>27.66 / 16.75</td>
<td>7.4</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet_layout_1x_table.yaml">PicoDet_layout_1x_table.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_table_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_table_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody></table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面表格区域检测数据集，包含中英文 7835 张带有表格的论文文档类型图片。</b>

* <b>3类版面检测模型，包含表格、图像、印章</b>
<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_3cls</td>
<td>88.2</td>
<td>8.43 / 3.44</td>
<td>17.60 / 6.51</td>
<td>4.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-S_layout_3cls.yaml">PicoDet-S_layout_3cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_3cls_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PicoDet-L_layout_3cls</td>
<td>89.0</td>
<td>12.80 / 9.57</td>
<td>45.04 / 23.86</td>
<td>22.6</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-L_layout_3cls.yaml">PicoDet-L_layout_3cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_3cls_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>RT-DETR-H_layout_3cls</td>
<td>95.8</td>
<td>114.80 / 25.65</td>
<td>924.38 / 924.38</td>
<td>470.1</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/RT-DETR-H_layout_3cls.yaml">RT-DETR-H_layout_3cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_3cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_3cls_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody></table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 1154 张文档类型图片。</b>

* <b>5类英文文档区域检测模型，包含文字、标题、表格、图片以及列表</b>
<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet_layout_1x</td>
<td>97.8</td>
<td>9.62 / 6.75</td>
<td>26.96 / 12.77</td>
<td>7.4</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet_layout_1x.yaml">PicoDet_layout_1x.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet_layout_1x_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet_layout_1x_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody></table>
<b>注：以上精度指标的评估集是 [PubLayNet](https://developer.ibm.com/exchanges/data/all/publaynet/) 的评估数据集，包含英文文档的 11245 张图片。</b>

* <b>17类区域检测模型，包含17个版面常见类别，分别是：段落标题、图片、文本、数字、摘要、内容、图表标题、公式、表格、表格标题、参考文献、文档标题、脚注、页眉、算法、页脚、印章</b>
<table>
<thead>
<tr>
<th>模型</th>
<th>mAP(0.5)（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PicoDet-S_layout_17cls</td>
<td>87.4</td>
<td>8.80 / 3.62</td>
<td>17.51 / 6.35</td>
<td>4.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-S_layout_17cls.yaml">PicoDet-S_layout_17cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-S_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-S_layout_17cls_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>PicoDet-L_layout_17cls</td>
<td>89.0</td>
<td>12.60 / 10.27</td>
<td>43.70 / 24.42</td>
<td>22.6</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/PicoDet-L_layout_17cls.yaml">PicoDet-L_layout_17cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PicoDet-L_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PicoDet-L_layout_17cls_pretrained.pdparams">训练模型</a></td>
</tr>
<tr>
<td>RT-DETR-H_layout_17cls</td>
<td>98.3</td>
<td>115.29 / 101.18</td>
<td>964.75 / 964.75</td>
<td>470.2</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/layout_detection/RT-DETR-H_layout_17cls.yaml">RT-DETR-H_layout_17cls.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/RT-DETR-H_layout_17cls_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/RT-DETR-H_layout_17cls_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是 PaddleOCR 自建的版面区域检测数据集，包含中英文论文、杂志和研报等常见的 892 张文档类型图片。</b>

## [文档图像方向分类模块](./module_usage/doc_img_orientation_classification.md)

<table>
<thead>
<tr>
<th>模型</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br/>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/doc_text_orientation/PP-LCNet_x1_0_doc_ori.yaml">PP-LCNet_x1_0_doc_ori.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>
<b>注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</b>


## [文本行方向分类模块](./module_usage/doc_img_orientation_classification.md)

<table>
<thead>
<tr>
<th>模型</th>
<th>Top-1 Acc（%）</th>
<th>GPU推理耗时（ms）<br>[常规模式 / 高性能模式]</th>
<th>CPU推理耗时（ms）<br>[常规模式 / 高性能模式]</th>
<th>模型存储大小（MB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
</thead>
<tbody>
<tr>
<td>PP-LCNet_x1_0_doc_ori</td>
<td>99.06</td>
<td>2.62 / 0.59</td>
<td>3.24 / 1.19</td>
<td>7</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/textline_orientation/PP-LCNet_x0_25_textline_ori.yaml">PP-LCNet_x0_25_textline_ori.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-LCNet_x1_0_doc_ori_pretrained.pdparams">训练模型</a></td>
</tr>
</tbody>
</table>

<b>注：以上精度指标的评估集是自建的数据集，覆盖证件和文档等多个场景，包含 1000 张图片。</b>

## [文档类视觉语言模型模块](./module_usage/doc_vlm.md)

<table>
<tr>
<th>模型</th>
<th>模型参数尺寸（B）</th>
<th>模型存储大小（GB）</th>
<th>yaml文件</th>
<th>模型下载链接</th>
</tr>
<tr>
<td>PP-DocBee-2B</td>
<td>2</td>
<td>4.2</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/doc_vlm/PP-DocBee-2B.yaml">PP-DocBee-2B.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee-2B_infer.tar">推理模型</a></td>
</tr>
<tr>
<td>PP-DocBee-7B</td>
<td>7</td>
<td>15.8</td>
<td><a href="https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/configs/modules/doc_vlm/PP-DocBee-7B.yaml">PP-DocBee-7B.yaml</a></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee-7B_infer.tar">推理模型</a></td>
</tr>
<tr>
<td>PP-DocBee2-3B</td>
<td>3</td>
<td>7.6</td>
<td></td>
<td><a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocBee2-3B_infer.tar">推理模型</a></td>
</tr>
</table>
