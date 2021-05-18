
## Introduction

Generally, a more complex model would achive better performance in the task, but it also leads to some redundancy in the model.
Quantization is a technique that reduces this redundancy by reducing the full precision data to a fixed number,
so as to reduce model calculation complexity and improve model inference performance.

This example uses PaddleSlim provided [APIs of Quantization](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/) to compress the OCR model.

It is recommended that you could understand following pages before reading this example：
- [The training strategy of OCR model](../../../doc/doc_en/quickstart_en.md)
- [PaddleSlim Document](https://paddlepaddle.github.io/PaddleSlim/api/quantization_api/)

## Quick Start
Quantization is mostly suitable for the deployment of lightweight models on mobile terminals.
After training, if you want to further compress the model size and accelerate the prediction, you can use quantization methods to compress the model according to the following steps.

1. Install PaddleSlim
2. Prepare trained model
3. Quantization-Aware Training
4. Export inference model
5. Deploy quantization inference model


### 1. Install PaddleSlim

```bash
git clone https://github.com/PaddlePaddle/PaddleSlim.git
cd Paddleslim
python setup.py install
```


### 2. Download Pretrain Model
PaddleOCR provides a series of trained [models](../../../doc/doc_en/models_list_en.md).
If the model to be quantified is not in the list, you need to follow the [Regular Training](../../../doc/doc_en/quickstart_en.md) method to get the trained model.


### 3. Quant-Aware Training
Quantization training includes offline quantization training and online quantization training.
Online quantization training is more effective. It is necessary to load the pre-training model.
After the quantization strategy is defined, the model can be quantified.

The code for quantization training is located in `slim/quantization/quant.py`. For example, to train a detection model, the training instructions are as follows:
```bash
python deploy/slim/quantization/quant.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model='your trained model'   Global.save_model_dir=./output/quant_model

# download provided model
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
tar -xf ch_ppocr_mobile_v2.0_det_train.tar
python deploy/slim/quantization/quant.py -c configs/det/det_mv3_db.yml -o Global.pretrained_model=./ch_ppocr_mobile_v2.0_det_train/best_accuracy   Global.save_model_dir=./output/quant_model
```


### 4. Export inference model

After getting the model after pruning and finetuning we, can export it as inference_model for predictive deployment:

```bash
python deploy/slim/quantization/export_model.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=output/quant_model/best_accuracy Global.save_inference_dir=./output/quant_inference_model
```

### 5. Deploy
The numerical range of the quantized model parameters derived from the above steps is still FP32, but the numerical range of the parameters is int8.
The derived model can be converted through the `opt tool` of PaddleLite.

For quantitative model deployment, please refer to [Mobile terminal model deployment](../../lite/readme_en.md)
