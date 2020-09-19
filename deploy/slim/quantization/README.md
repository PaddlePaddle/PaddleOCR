> 运行示例前请先安装1.2.0或更高版本PaddleSlim


# 模型量化压缩教程

压缩结果：
<table>
<thead>
  <tr>
    <th>序号</th>
    <th>任务</th>
    <th>模型</th>
    <th>压缩策略</th>
    <th>精度(自建中文数据集)</th>
    <th>耗时(ms)</th>
    <th>整体耗时(ms)</th>
    <th>加速比</th>
    <th>整体模型大小(M)</th>
    <th>压缩比例</th>
    <th>下载链接</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">0</td>
    <td>检测</td>
    <td>MobileNetV3_DB</td>
    <td>无</td>
    <td>61.7</td>
    <td>224</td>
    <td rowspan="2">375</td>
    <td rowspan="2">-</td>
    <td rowspan="2">8.6</td>
    <td rowspan="2">-</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>MobileNetV3_CRNN</td>
    <td>无</td>
    <td>62.0</td>
    <td>9.52</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">1</td>
    <td>检测</td>
    <td>SlimTextDet</td>
    <td>PACT量化训练</td>
    <td>62.1</td>
    <td>195</td>
    <td rowspan="2">348</td>
    <td rowspan="2">8%</td>
    <td rowspan="2">2.8</td>
    <td rowspan="2">67.82%</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>SlimTextRec</td>
    <td>PACT量化训练</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">2</td>
    <td>检测</td>
    <td>SlimTextDet_quat_pruning</td>
    <td>剪裁+PACT量化训练</td>
    <td>60.86</td>
    <td>142</td>
    <td rowspan="2">288</td>
    <td rowspan="2">30%</td>
    <td rowspan="2">2.8</td>
    <td rowspan="2">67.82%</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>SlimTextRec</td>
    <td>PACT量化训练</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">3</td>
    <td>检测</td>
    <td>SlimTextDet_pruning</td>
    <td>剪裁</td>
    <td>61.57</td>
    <td>138</td>
    <td rowspan="2">295</td>
    <td rowspan="2">27%</td>
    <td rowspan="2">2.9</td>
    <td rowspan="2">66.28%</td>
    <td></td>
  </tr>
  <tr>
    <td>识别</td>
    <td>SlimTextRec</td>
    <td>PACT量化训练</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
</tbody>
</table>



## 概述

复杂的模型有利于提高模型的性能，但也导致模型中存在一定冗余，模型量化将全精度缩减到定点数减少这种冗余，达到减少模型计算复杂度，提高模型推理性能的目的。

该示例使用PaddleSlim提供的[量化压缩API](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)对OCR模型进行压缩。
在阅读该示例前，建议您先了解以下内容：

- [OCR模型的常规训练方法](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/detection.md)
- [PaddleSlim使用文档](https://paddleslim.readthedocs.io/zh_CN/latest/index.html)



## 安装PaddleSlim

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git

cd Paddleslim

python setup.py install
```



## 获取预训练模型

[识别预训练模型下载地址]()

[检测预训练模型下载地址]()


## 量化训练
加载预训练模型后，在定义好量化策略后即可对模型进行量化。量化相关功能的使用具体细节见：[模型量化](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/quantization_api.html)

进入PaddleOCR根目录，通过以下命令对模型进行量化：

```bash
python deploy/slim/quantization/quant.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=det_mv3_db/best_accuracy Global.save_model_dir=./output/quant_model
```




## 导出模型

在得到量化训练保存的模型后，我们可以将其导出为inference_model，用于预测部署：

```bash
python deploy/slim/quantization/export_model.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=output/quant_model/best_accuracy Global.save_model_dir=./output/quant_inference_model
```
