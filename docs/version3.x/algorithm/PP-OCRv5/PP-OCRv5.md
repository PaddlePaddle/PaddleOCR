# 一、PP-OCRv5简介
**PP-OCRv5** 是PP-OCR新一代文字识别解决方案，该方案聚焦于多场景、多文字类型的文字识别。在文字类型方面，PP-OCRv5支持简体中文、中文拼音、繁体中文、英文、日文5大主流文字类型，在场景方面，PP-OCRv5升级了中英复杂手写体、竖排文本、生僻字等多种挑战性场景的识别能力。在内部多场景复杂评估集上，PP-OCRv5较PP-OCRv4端到端提升13个百分点。

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/algorithm_ppocrv5.png" width="400"/>



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

对比PP-OCRv4，PP-OCRv5在所有检测场景下均有明显提升，尤其在手写、古籍、日文检测能力上表现更优。

### 2. 文本识别指标

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/ocrv5_rec_acc.png" width="400"/>

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

单模型即可覆盖多语言和多类型文本，识别精度大幅领先前代产品和主流开源方案。


# 四、端到端推理benchmark
依赖版本：Paddle框架 3.0正式版，CUDA 11.8，cuDNN 8.9。

输入数据：1185张图像，涵盖通用场景以及文档场景。

测试时加载文档图像预处理模型，但不使用（通过API传参关闭）。

本地推理：

1. 测试硬件：NVIDIA Tesla V100 + Intel Xeon Gold 6271C

<table>
  <thead>
    <tr>
      <th>产线配置</th>
      <th>平均每图耗时（s）</th>
      <th>平均CPU利用率（%）</th>
      <th>峰值RAM用量（MB）</th>
      <th>平均RAM用量（MB）</th>
      <th>平均GPU利用率（%）</th>
      <th>峰值VRAM用量（MB）</th>
      <th>平均VRAM用量（MB）</th>
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

2. 测试硬件：NVIDIA A100 + Intel Xeon Platinum 8350C

<table>
  <thead>
    <tr>
      <th>产线配置</th>
      <th>平均每图耗时（s）</th>
      <th>平均CPU利用率（%）</th>
      <th>峰值RAM用量（MB）</th>
      <th>平均RAM用量（MB）</th>
      <th>平均GPU利用率（%）</th>
      <th>峰值VRAM用量（MB）</th>
      <th>平均VRAM用量（MB）</th>
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

服务化部署：

测试硬件：NVIDIA A100 + Intel Xeon Platinum 8350C

<table>
  <thead>
    <tr>
      <th>产线配置</th>
      <th>实例数</th>
      <th>并发请求数</th>
      <th>吞吐</th>
      <th>平均时延（s）</th>
      <th>成功请求数/总请求数</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">PP-OCRv5_server</td>
      <td>4卡*1</td>
      <td>4</td>
      <td>7.20</td>
      <td>0.55</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>4卡*4</td>
      <td>16</td>
      <td>21.78</td>
      <td>0.73</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>4卡*8</td>
      <td>32</td>
      <td>28.57</td>
      <td>1.11</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td rowspan="3">PP-OCRv5_mobile</td>
      <td>4卡*1</td>
      <td>4</td>
      <td>7.95</td>
      <td>0.50</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>4卡*4</td>
      <td>16</td>
      <td>24.94</td>
      <td>0.64</td>
      <td>1.000</td>
    </tr>
    <tr>
      <td>4卡*8</td>
      <td>32</td>
      <td>29.92</td>
      <td>1.05</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>

# 五、PP-OCRv5关键提升Demo示例

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-OCRv5/ocrv5_demo.gif" width="400"/>

# 六、部署与二次开发
* **多系统支持**：兼容Windows、Linux、Mac等主流操作系统。
* **多硬件支持**：除了英伟达GPU外，还支持Intel CPU、昆仑芯、shengteng等新硬件推理和部署。
* **高性能推理插件**：推荐结合高性能推理插件进一步提升推理速度，详见[高性能推理指南](../../deployment/high_performance_inference.md)。
* **服务化部署**：支持高稳定性服务化部署方案，详见[服务化部署指南](../../deployment/serving.md)。
* **二次开发能力**：支持自定义数据集训练、字典扩展、模型微调。举例：如需增加韩文识别，可扩展字典并微调模型，无缝集成到现有产线，详见[文本识别模块使用教程](../../module_usage/text_recognition.md)
