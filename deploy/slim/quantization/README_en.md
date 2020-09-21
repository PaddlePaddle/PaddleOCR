\> PaddleSlim 1.2.0 or higher version should be installed before runing this example.



# Model compress tutorial (Quantization)

Compress results：
<table>
<thead>
  <tr>
    <th>ID</th>
    <th>Task</th>
    <th>Model</th>
    <th>Compress Strategy</th>
    <th>Criterion(Chinese dataset)</th>
    <th>Inference Time(ms)</th>
    <th>Inference Time(Total model)(ms)</th>
    <th>Acceleration Ratio</th>
    <th>Model Size(MB)</th>
    <th>Commpress Ratio</th>
    <th>Download Link</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">0</td>
    <td>Detection</td>
    <td>MobileNetV3_DB</td>
    <td>None</td>
    <td>61.7</td>
    <td>224</td>
    <td rowspan="2">375</td>
    <td rowspan="2">-</td>
    <td rowspan="2">8.6</td>
    <td rowspan="2">-</td>
    <td></td>
  </tr>
  <tr>
    <td>Recognition</td>
    <td>MobileNetV3_CRNN</td>
    <td>None</td>
    <td>62.0</td>
    <td>9.52</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">1</td>
    <td>Detection</td>
    <td>SlimTextDet</td>
    <td>PACT Quant Aware Training</td>
    <td>62.1</td>
    <td>195</td>
    <td rowspan="2">348</td>
    <td rowspan="2">8%</td>
    <td rowspan="2">2.8</td>
    <td rowspan="2">67.82%</td>
    <td></td>
  </tr>
  <tr>
    <td>Recognition</td>
    <td>SlimTextRec</td>
    <td>PACT Quant Aware Training</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">2</td>
    <td>Detection</td>
    <td>SlimTextDet_quat_pruning</td>
    <td>Pruning+PACT Quant Aware Training</td>
    <td>60.86</td>
    <td>142</td>
    <td rowspan="2">288</td>
    <td rowspan="2">30%</td>
    <td rowspan="2">2.8</td>
    <td rowspan="2">67.82%</td>
    <td></td>
  </tr>
  <tr>
    <td>Recognition</td>
    <td>SlimTextRec</td>
    <td>PPACT Quant Aware Training</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">3</td>
    <td>Detection</td>
    <td>SlimTextDet_pruning</td>
    <td>Pruning</td>
    <td>61.57</td>
    <td>138</td>
    <td rowspan="2">295</td>
    <td rowspan="2">27%</td>
    <td rowspan="2">2.9</td>
    <td rowspan="2">66.28%</td>
    <td></td>
  </tr>
  <tr>
    <td>Recognition</td>
    <td>SlimTextRec</td>
    <td>PACT Quant Aware Training</td>
    <td>61.48</td>
    <td>8.6</td>
    <td></td>
  </tr>
</tbody>
</table>



## Overview

Generally, a more complex model would achive better performance in the task, but it also leads to some redundancy in the model. Quantization is a technique that reduces this redundancyby reducing the full precision data to a fixed number, so as to reduce model calculation complexity and improve model inference performance.

This example uses PaddleSlim provided [APIs of Quantization](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/) to compress the OCR model.
PaddleSlim (GitHub: https://github.com/PaddlePaddle/PaddleSlim), an open source library which integrates model pruning, quantization (including quantization training and offline quantization), distillation, neural network architecture search, and many other commonly used and leading model compression technique in the industry.

It is recommended that you could understand following pages before reading this example,：



- [The training strategy of OCR model](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/detection.md)

- [PaddleSlim Document](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)



## Install PaddleSlim

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git

cd Paddleslim

python setup.py install

```


## Download Pretrain Model

[Download link of Detection pretrain model]()

[Download link of recognization pretrain model]()


## Quan-Aware Training

After loading the pre training model, the model can be quantified after defining the quantization strategy. For specific details of quantization method, see：[Model Quantization](https://paddleslim.readthedocs.io/zh_CN/latest/api_cn/quantization_api.html)

Enter the PaddleOCR root directory，perform model quantization with the following command：

```bash
python deploy/slim/prune/sensitivity_anal.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1
```



## Export inference model

After getting the model after pruning and finetuning we, can export it as inference_model for predictive deployment:

```bash
python deploy/slim/quantization/export_model.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=output/quant_model/best_accuracy Global.save_model_dir=./output/quant_inference_model
```
