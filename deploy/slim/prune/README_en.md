
## Introduction

Generally, a more complex model would achive better performance in the task, but it also leads to some redundancy in the model. Model Pruning is a technique that reduces this redundancy by removing the sub-models in the neural network model, so as to reduce model calculation complexity and improve model inference performance.

This example uses PaddleSlim provided[APIs of Pruning](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/docs/zh_cn/api_cn/dygraph/pruners) to compress the OCR model.
[PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim), an open source library which integrates model pruning, quantization (including quantization training and offline quantization), distillation, neural network architecture search, and many other commonly used and leading model compression technique in the industry.

It is recommended that you could understand following pages before reading this example：
1. [PaddleOCR training methods](../../../doc/doc_ch/quickstart.md)
2. [The demo of prune](https://github.com/PaddlePaddle/PaddleSlim/blob/release%2F2.0.0/docs/zh_cn/tutorials/pruning/dygraph/filter_pruning.md)

## Quick start

Five steps for OCR model prune:
1. Install PaddleSlim
2. Prepare the trained model
3. Sensitivity analysis and tailoring training
4. Export model, predict deployment

### 1. Install PaddleSlim

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd PaddleSlim
git checkout develop
python3 setup.py install
```


### 2. Download Pre-trained Model
Model prune needs to load pre-trained models.
PaddleOCR also provides a series of [models](../../../doc/doc_en/models_list_en.md). Developers can choose their own models or use their own models according to their needs.


### 3. Pruning sensitivity analysis

  After the pre-trained model is loaded, sensitivity analysis is performed on each network layer of the model to understand the redundancy of each network layer, and save a sensitivity file which named: sen.pickle.  After that, user could load the sensitivity file via the [methods provided by PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/paddleslim/prune/sensitive.py#L221) and determining the pruning ratio of each network layer automatically. For specific details of sensitivity analysis, see：[Sensitivity analysis](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/en/tutorials/image_classification_sensitivity_analysis_tutorial_en.md)
  The data format of sensitivity file：
      sen.pickle(Dict){
              'layer_weight_name_0': sens_of_each_ratio(Dict){'pruning_ratio_0': acc_loss, 'pruning_ratio_1': acc_loss}
              'layer_weight_name_1': sens_of_each_ratio(Dict){'pruning_ratio_0': acc_loss, 'pruning_ratio_1': acc_loss}
          }

      example：
          {
              'conv10_expand_weights': {0.1: 0.006509952684312718, 0.2: 0.01827734339798862, 0.3: 0.014528405644659832, 0.6: 0.06536008804270439, 0.8: 0.11798612250664964, 0.7: 0.12391408417493704, 0.4: 0.030615754498018757, 0.5: 0.047105205602406594}
              'conv10_linear_weights': {0.1: 0.05113190831455035, 0.2: 0.07705573833558801, 0.3: 0.12096721757739311, 0.6: 0.5135061352930738, 0.8: 0.7908166677143281, 0.7: 0.7272187676899062, 0.4: 0.1819252083008504, 0.5: 0.3728054727792405}
          }
  The function would return a dict after loading the sensitivity file. The keys of the dict are name of parameters in each layer. And the value of key is the information about pruning sensitivity of corresponding layer. In example, pruning 10% filter of the layer corresponding to conv10_expand_weights would lead to 0.65% degradation of model performance. The details could be seen at: [Sensitivity analysis](https://github.com/PaddlePaddle/PaddleSlim/blob/release/2.0-alpha/docs/zh_cn/algo/algo.md)


Enter the PaddleOCR root directory，perform sensitivity analysis on the model with the following command：

```bash

python3.7 deploy/slim/prune/sensitivity_anal.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.pretrained_model="your trained model"  Global.save_model_dir=./output/prune_model/

```


### 5.  Export inference model and deploy it

We can export the pruned model as inference_model for deployment:
```bash
python deploy/slim/prune/export_prune_model.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml  -o Global.pretrained_model=./output/det_db/best_accuracy  Global.save_inference_dir=./prune/prune_inference_model
```

Reference for prediction and deployment of inference model:
1. [inference model python prediction](../../../doc/doc_en/inference_en.md)
2. [inference model C++ prediction](../../cpp_infer/readme_en.md)
