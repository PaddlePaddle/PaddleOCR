\> PaddleSlim develop version should be installed before runing this example.



# Model compress tutorial (Pruning)

Compress results：
<table>
<thead>
  <tr>
    <th>ID</th>
    <th>Task</th>
    <th>Model</th>
    <th>Compress Strategy<sup><a href="#quant">[3]</a><a href="#prune">[4]</a><sup></th>
    <th>Criterion(Chinese dataset)</th>
    <th>Inference Time<sup><a href="#latency">[1]</a></sup>(ms)</th>
    <th>Inference Time(Total model)<sup><a href="#rec">[2]</a></sup>(ms)</th>
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

Generally, a more complex model would achive better performance in the task, but it also leads to some redundancy in the model. Model Pruning is a technique that reduces this redundancy by removing the sub-models in the neural network model, so as to reduce model calculation complexity and improve model inference performance.

This example uses PaddleSlim provided[APIs of Pruning](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/) to compress the OCR model.

It is recommended that you could understand following pages before reading this example,：



\- [The training strategy of OCR model](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/detection.md)

\- [PaddleSlim Document](https://paddlepaddle.github.io/PaddleSlim/)



## Install PaddleSlim

```bash

git clone https://github.com/PaddlePaddle/PaddleSlim.git

cd Paddleslim

python setup.py install

```


## Download Pretrain Model

[Download link of Detection pretrain model]()


## Pruning sensitivity analysis

  After the pre-training model is loaded, sensitivity analysis is performed on each network layer of the model to understand the redundancy of each network layer, thereby determining the pruning ratio of each network layer. For specific details of sensitivity analysis, see：[Sensitivity analysis](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/image_classification_sensitivity_analysis_tutorial.md)

Enter the PaddleOCR root directory，perform sensitivity analysis on the model with the following command：

```bash

python deploy/slim/prune/sensitivity_anal.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1

```



## Model pruning and Fine-tune

  When pruning, the previous sensitivity analysis file would determines the pruning ratio of each network layer. In the specific implementation, in order to retain as many low-level features extracted from the image as possible, we skipped the 4 convolutional layers close to the input in the backbone. Similarly, in order to reduce the model performance loss caused by pruning, we selected some of the less redundant and more sensitive [network layer](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/slim/prune/pruning_and_finetune.py#L41) through the sensitivity table obtained from the previous sensitivity analysis.And choose to skip these network layers in the subsequent pruning process. After pruning, the model need a finetune process to recover the performance and the training strategy of finetune is similar to the strategy of training original OCR detection model.

```bash

python deploy/slim/prune/pruning_and_finetune.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1

```





## Export inference model

After getting the model after pruning and finetuning we, can export it as inference_model for predictive deployment:

```bash

python deploy/slim/prune/export_prune_model.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./output/det_db/best_accuracy Global.test_batch_size_per_card=1 Global.save_inference_dir=inference_model

```
