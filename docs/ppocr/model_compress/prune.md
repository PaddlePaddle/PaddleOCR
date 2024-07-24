---
comments: true
---

# PP-OCR模型裁剪

复杂的模型有利于提高模型的性能，但也导致模型中存在一定冗余，模型裁剪通过移出网络模型中的子模型来减少这种冗余，达到减少模型计算复杂度，提高模型推理性能的目的。
本教程将介绍如何使用飞桨模型压缩库PaddleSlim做PaddleOCR模型的压缩。
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim)集成了模型剪枝、量化（包括量化训练和离线量化）、蒸馏和神经网络搜索等多种业界常用且领先的模型压缩功能，如果您感兴趣，可以关注并了解。

在开始本教程之前，建议先了解：

1. [PaddleOCR模型的训练方法](../model_train/training.md)
2. [模型裁剪教程](https://github.com/PaddlePaddle/PaddleSlim/blob/release%2F2.0.0/docs/zh_cn/tutorials/pruning/dygraph/filter_pruning.md)

## 快速开始

### 1.安装PaddleSlim

```bash linenums="1"
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
git checkout develop
python3 setup.py install
```

### 2.获取预训练模型

模型裁剪需要加载事先训练好的模型，PaddleOCR也提供了一系列[模型](../model_list.md)，开发者可根据需要自行选择模型或使用自己的模型。

### 3.敏感度分析训练

加载预训练模型后，通过对现有模型的每个网络层进行敏感度分析，得到敏感度文件：sen.pickle，可以通过PaddleSlim提供的[接口](https://github.com/PaddlePaddle/PaddleSlim/blob/9b01b195f0c4bc34a1ab434751cb260e13d64d9e/paddleslim/dygraph/prune/filter_pruner.py#L75)加载文件，获得各网络层在不同裁剪比例下的精度损失。从而了解各网络层冗余度，决定每个网络层的裁剪比例。
敏感度文件内容格式：

```python linenums="1"
sen.pickle(Dict){
            'layer_weight_name_0': sens_of_each_ratio(Dict){'pruning_ratio_0': acc_loss, 'pruning_ratio_1': acc_loss}
            'layer_weight_name_1': sens_of_each_ratio(Dict){'pruning_ratio_0': acc_loss, 'pruning_ratio_1': acc_loss}
        }
```

例子：

```python linenums="1"
{
    'conv10_expand_weights': {0.1: 0.006509952684312718, 0.2: 0.01827734339798862, 0.3: 0.014528405644659832, 0.6: 0.06536008804270439, 0.8: 0.11798612250664964, 0.7: 0.12391408417493704, 0.4: 0.030615754498018757, 0.5: 0.047105205602406594}
    'conv10_linear_weights': {0.1: 0.05113190831455035, 0.2: 0.07705573833558801, 0.3: 0.12096721757739311, 0.6: 0.5135061352930738, 0.8: 0.7908166677143281, 0.7: 0.7272187676899062, 0.4: 0.1819252083008504, 0.5: 0.3728054727792405}
}
```

加载敏感度文件后会返回一个字典，字典中的keys为网络模型参数模型的名字，values为一个字典，里面保存了相应网络层的裁剪敏感度信息。例如在例子中，conv10_expand_weights所对应的网络层在裁掉10%的卷积核后模型性能相较原模型会下降0.65%，详细信息可见[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/algo/algo.md#2-%E5%8D%B7%E7%A7%AF%E6%A0%B8%E5%89%AA%E8%A3%81%E5%8E%9F%E7%90%86)

进入PaddleOCR根目录，通过以下命令对模型进行敏感度分析训练：

```bash linenums="1"
python3.7 deploy/slim/prune/sensitivity_anal.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.pretrained_model="your trained model" Global.save_model_dir=./output/prune_model/
```

### 4.导出模型、预测部署

在得到裁剪训练保存的模型后，我们可以将其导出为inference_model：

```bash linenums="1"
pytho3.7 deploy/slim/prune/export_prune_model.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.pretrained_model=./output/det_db/best_accuracy  Global.save_inference_dir=./prune/prune_inference_model
```

inference model的预测和部署参考：

1. [inference model python端预测](../infer_deploy/python_infer.md)
2. [inference model C++预测](../infer_deploy/cpp_infer.md)
