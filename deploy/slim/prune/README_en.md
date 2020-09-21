

## Introduction

Complicated models help to improve the performance of the model, but it also leads to some redundancy in the model. Model tailoring reduces this redundancy by removing the sub-models in the network model, so as to reduce model calculation complexity and improve model inference performance. .

This tutorial will introduce how to use PaddleSlim to crop PaddleOCR model.

It is recommended that you could understand following pages before reading this example：
1. [PaddleOCR training methods](../../../doc/doc_ch/quickstart.md)
2. [The demo of prune](https://paddlepaddle.github.io/PaddleSlim/tutorials/pruning_tutorial/)
3. [PaddleSlim prune API](https://paddlepaddle.github.io/PaddleSlim/api/prune_api/)

## Quick start

Five steps for OCR model prune:
1. Install PaddleSlim
2. Prepare the trained model
3. Sensitivity analysis and training
4. Model tailoring training
5. Export model, predict deployment

### 1. Install PaddleSlim

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python setup.py install
```


### 2. Download Pretrain Model
Model prune needs to load pre-trained models.
PaddleOCR also provides a series of models [../../../doc/doc_en/models_list_en.md]. Developers can choose their own models or use their own models according to their needs.


### 3. Pruning sensitivity analysis

  After the pre-training model is loaded, sensitivity analysis is performed on each network layer of the model to understand the redundancy of each network layer, thereby determining the pruning ratio of each network layer. For specific details of sensitivity analysis, see：[Sensitivity analysis](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/tutorials/image_classification_sensitivity_analysis_tutorial.md)

Enter the PaddleOCR root directory，perform sensitivity analysis on the model with the following command：

```bash

python deploy/slim/prune/sensitivity_anal.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1

```



### 4. Model pruning and Fine-tune

  When pruning, the previous sensitivity analysis file would determines the pruning ratio of each network layer. In the specific implementation, in order to retain as many low-level features extracted from the image as possible, we skipped the 4 convolutional layers close to the input in the backbone. Similarly, in order to reduce the model performance loss caused by pruning, we selected some of the less redundant and more sensitive [network layer](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/deploy/slim/prune/pruning_and_finetune.py#L41) through the sensitivity table obtained from the previous sensitivity analysis.And choose to skip these network layers in the subsequent pruning process. After pruning, the model need a finetune process to recover the performance and the training strategy of finetune is similar to the strategy of training original OCR detection model.

```bash

python deploy/slim/prune/pruning_and_finetune.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./deploy/slim/prune/pretrain_models/det_mv3_db/best_accuracy Global.test_batch_size_per_card=1

```


### 5.  Export inference model and deploy it

We can export the pruned model as inference_model for deployment:
```bash
python deploy/slim/prune/export_prune_model.py -c configs/det/det_mv3_db.yml -o Global.pretrain_weights=./output/det_db/best_accuracy Global.test_batch_size_per_card=1 Global.save_inference_dir=inference_model
```

Reference for prediction and deployment of inference model:
1. [inference model python prediction](../../../doc/doc_en/inference_en.md)
2. [inference model C++ prediction](../../cpp_infer/readme_en.md)
3. [Deployment of inference model on mobile](../../lite/readme_en.md)
