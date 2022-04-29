
# PP-OCR Models Quantization

Generally, a more complex model would achieve better performance in the task, but it also leads to some redundancy in the model.
Quantization is a technique that reduces this redundancy by reducing the full precision data to a fixed number,
so as to reduce model calculation complexity and improve model inference performance.

This example uses PaddleSlim provided [APIs of Quantization](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/dygraph/quanter/qat.rst) to compress the OCR model.

It is recommended that you could understand following pages before reading this example：
- [The training strategy of OCR model](../../../doc/doc_en/quickstart_en.md)
- [PaddleSlim Document](https://github.com/PaddlePaddle/PaddleSlim/blob/develop/docs/zh_cn/api_cn/dygraph/quanter/qat.rst)

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
pip3 install paddleslim==2.2.2
```


### 2. Download Pre-trained Model
PaddleOCR provides a series of pre-trained [models](../../../doc/doc_en/models_list_en.md).
If the model to be quantified is not in the list, you need to follow the [Regular Training](../../../doc/doc_en/quickstart_en.md) method to get the trained model.


### 3. Quant-Aware Training
Quantization training includes offline quantization training and online quantization training.
Online quantization training is more effective. It is necessary to load the pre-trained model.
After the quantization strategy is defined, the model can be quantified.

The code for quantization training is located in `slim/quantization/quant.py`. For example, to train a detection model, the training instructions are as follows:
```bash
python deploy/slim/quantization/quant.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.pretrained_model='your trained model'   Global.save_model_dir=./output/quant_model

# download provided model
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar
tar -xf ch_ppocr_mobile_v2.0_det_train.tar
python deploy/slim/quantization/quant.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.pretrained_model=./ch_ppocr_mobile_v2.0_det_train/best_accuracy   Global.save_model_dir=./output/quant_model
```


Model distillation and model quantization can be used at the same time, taking the PPOCRv3 detection model as an example:
```
# download provided model
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar
tar xf https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar

python deploy/slim/quantization/quant.py -c configs/det/ch_PP-OCRv3_det/ch_PP-OCRv3_det_cml.yml -o Global.pretrained_model='./ch_PP-OCRv3_det_distill_train/best_accuracy'   Global.save_model_dir=./output/quant_model_distill/
```

If you want to quantify the text recognition model, you can modify the configuration file and loaded model parameters.

### 4. Export inference model

Once we got the model after pruning and fine-tuning, we can export it as an inference model for the deployment of predictive tasks:

```bash
python deploy/slim/quantization/export_model.py -c configs/det/ch_ppocr_v2.0/ch_det_mv3_db_v2.0.yml -o Global.checkpoints=output/quant_model/best_accuracy Global.save_inference_dir=./output/quant_inference_model
```

### 5. Deploy
The numerical range of the quantized model parameters derived from the above steps is still FP32, but the numerical range of the parameters is int8.
The derived model can be converted through the `opt tool` of PaddleLite.

For quantitative model deployment, please refer to [Mobile terminal model deployment](../../lite/readme_en.md)
