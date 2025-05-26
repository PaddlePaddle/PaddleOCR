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

# 四、部署与二次开发
* **多系统支持**：兼容Windows、Linux、Mac等主流操作系统。
* **多硬件支持**：除了英伟达GPU外，还支持Intel CPU、昆仑芯、昇腾等新硬件推理和部署。
* **高性能推理插件**：推荐结合高性能推理插件进一步提升推理速度，详见[高性能推理指南](../../deployment/high_performance_inference.md)。
* **服务化部署**：支持高稳定性服务化部署方案，详见[服务化部署指南](../../deployment/serving.md)。
* **二次开发能力**：支持自定义数据集训练、字典扩展、模型微调。举例：如需增加韩文识别，可扩展字典并微调模型，无缝集成到现有产线，详见[文本检测模块使用教程](../../module_usage/text_detection.md)及[文本识别模块使用教程](../../module_usage/text_recognition.md)
