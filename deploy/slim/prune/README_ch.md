\> 运行示例前请先安装develop版本PaddleSlim



# 模型裁剪压缩教程

压缩结果：
<table>
<thead>
  <tr>
    <th>序号</th>
    <th>任务</th>
    <th>模型</th>
    <th>压缩策略<sup><a href="#quant">[3]</a><a href="#prune">[4]</a><sup></th>
    <th>精度(自建中文数据集)</th>
    <th>耗时<sup><a href="#latency">[1]</a></sup>(ms)</th>
    <th>整体耗时<sup><a href="#rec">[2]</a></sup>(ms)</th>
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

复杂的模型有利于提高模型的性能，但也导致模型中存在一定冗余，模型裁剪通过移出网络模型中的子模型来减少这种冗余，达到减少模型计算复杂度，提高模型推理性能的目的。

该示例使用PaddleSlim提供的[裁剪压缩API](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/)对OCR模型进行压缩。

在阅读该示例前，建议您先了解以下内容：



\- [OCR模型的常规训练方法](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/detection.md)

\- [PaddleSlim使用文档](https://paddlepaddle.github.io/PaddleSlim/)



## 安装PaddleSlim

\```bash

git clone https://github.com/PaddlePaddle/PaddleSlim.git

cd Paddleslim

python setup.py install

\```


## 获取预训练模型
[检测预训练模型下载地址]()


## 敏感度分析训练
  加载预训练模型后，通过对现有模型的每个网络层进行敏感度分析，了解各网络层冗余度，从而决定每个网络层的裁剪比例。敏感度分析的具体细节见：[敏感度分析](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/image_classification_sensitivity_analysis_tutorial.md)

进入PaddleOCR根目录，通过以下命令对模型进行敏感度分析：

\```bash

python deploy/slim/prune/sensitivity_anal.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1

\```



## 裁剪模型与fine-tune
  裁剪时通过之前的敏感度分析文件决定每个网络层的裁剪比例。在具体实现时，为了尽可能多的保留从图像中提取的低阶特征，我们跳过了backbone中靠近输入的4个卷积层。同样，为了减少由于裁剪导致的模型性能损失，我们通过之前敏感度分析所获得的敏感度表，挑选出了一些冗余较少，对裁剪较为敏感的[网络层](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/slim/prune/pruning_and_finetune.py#L41)，并在之后的裁剪过程中选择避开这些网络层。裁剪过后finetune的过程沿用OCR检测模型原始的训练策略。

\```bash

python deploy/slim/prune/pruning_and_finetune.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1

\```





## 导出模型

在得到裁剪训练保存的模型后，我们可以将其导出为inference_model，用于预测部署：

\```bash

python deploy/slim/prune/export_prune_model.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./output/det_db/best_accuracy Global.test_batch_size_per_card=1 Global.save_inference_dir=inference_model

\```
