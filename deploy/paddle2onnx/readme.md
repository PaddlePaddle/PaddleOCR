# Paddle2ONNX model transformation and prediction

This chapter describes how the PaddleOCR model is converted into an ONNX model and predicted based on the ONNXRuntime engine.

## 1. Environment preparation

Need to prepare PaddleOCR, Paddle2ONNX model conversion environment, and ONNXRuntime prediction environment

###  PaddleOCR

Clone the PaddleOCR repository, use the release/2.6 branch, and install it.

```
git clone  -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR && python3.7 setup.py install
```

###  Paddle2ONNX

Paddle2ONNX supports converting the PaddlePaddle model format to the ONNX model format. The operator currently supports exporting ONNX Opset 9~11 stably, and some Paddle operators support lower ONNX Opset conversion.
For more details, please refer to [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX/blob/develop/README_en.md)


- install Paddle2ONNX
```
python3.7 -m pip install paddle2onnx
```

- install ONNXRuntime
```
# It is recommended to install version 1.9.0, and the version number can be changed according to the environment
python3.7 -m pip install onnxruntime==1.9.0
```

## 2. Model conversion


- Paddle model download

There are two ways to obtain the Paddle model: Download the prediction model provided by PaddleOCR in [model_list](../../doc/doc_en/models_list_en.md);
Refer to [Model Export Instructions](../../doc/doc_en/inference_en.md#1-convert-training-model-to-inference-model) to convert the trained weights to inference_model.

Take the PP-OCRv3 detection, recognition, and classification model as an example:

```
wget -nc -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar
cd ./inference && tar xf en_PP-OCRv3_det_infer.tar && cd ..

wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar
cd ./inference && tar xf en_PP-OCRv3_rec_infer.tar && cd ..

wget -nc  -P ./inference https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
cd ./inference && tar xf ch_ppocr_mobile_v2.0_cls_infer.tar && cd ..
```

- convert model

Convert Paddle inference model to ONNX model format using Paddle2ONNX:

```
paddle2onnx --model_dir ./inference/en_PP-OCRv3_det_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/det_onnx/model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/en_PP-OCRv3_rec_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/rec_onnx/model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True

paddle2onnx --model_dir ./inference/ch_ppocr_mobile_v2.0_cls_infer \
--model_filename inference.pdmodel \
--params_filename inference.pdiparams \
--save_file ./inference/cls_onnx/model.onnx \
--opset_version 10 \
--input_shape_dict="{'x':[-1,3,-1,-1]}" \
--enable_onnx_checker True
```
After execution, the ONNX model will be saved in `./inference/det_onnx/`, `./inference/rec_onnx/`, `./inference/cls_onnx/` paths respectively

* Note: For the OCR model, the conversion process must be in the form of dynamic shape, that is, add the option --input_shape_dict="{'x': [-1, 3, -1, -1]}", otherwise the prediction result may be the same as Predicting directly with Paddle is slightly different.
  In addition, the following models do not currently support conversion to ONNX models:
  NRTR, SAR, RARE, SRN

## 3. prediction

Take the English OCR model as an example, use **ONNXRuntime** to predict and execute the following commands:

```
python3.7 tools/infer/predict_system.py --use_gpu=False --use_onnx=True \
--det_model_dir=./inference/det_onnx/model.onnx  \
--rec_model_dir=./inference/rec_onnx/model.onnx  \
--cls_model_dir=./inference/cls_onnx/model.onnx  \
--image_dir=doc/imgs_en/img_12.jpg \
--rec_char_dict_path=ppocr/utils/en_dict.txt
```

Taking the English OCR model as an example, use **Paddle Inference** to predict and execute the following commands:

```
python3.7 tools/infer/predict_system.py --use_gpu=False \
--cls_model_dir=./inference/ch_ppocr_mobile_v2.0_cls_infer \
--rec_model_dir=./inference/en_PP-OCRv3_rec_infer \
--det_model_dir=./inference/en_PP-OCRv3_det_infer \
--image_dir=doc/imgs_en/img_12.jpg \
--rec_char_dict_path=ppocr/utils/en_dict.txt
```


After executing the command, the predicted identification information will be printed out in the terminal, and the visualization results will be saved under `./inference_results/`.

ONNXRuntime result：

<div align="center">
    <img src="../../doc/imgs_results/multi_lang/img_12.jpg" width=800">
</div>

Paddle Inference result：

<div align="center">
    <img src="../../doc/imgs_results/multi_lang/img_12.jpg" width=800">
</div>


Using ONNXRuntime to predict, terminal output:
```
[2022/10/10 12:06:28] ppocr DEBUG: dt_boxes num : 11, elapse : 0.3568880558013916
[2022/10/10 12:06:31] ppocr DEBUG: rec_res num  : 11, elapse : 2.6445000171661377
[2022/10/10 12:06:31] ppocr DEBUG: 0  Predict time of doc/imgs_en/img_12.jpg: 3.021s
[2022/10/10 12:06:31] ppocr DEBUG: ACKNOWLEDGEMENTS, 0.997
[2022/10/10 12:06:31] ppocr DEBUG: We would like to thank all the designers and, 0.976
[2022/10/10 12:06:31] ppocr DEBUG: contributors who have been involved in the, 0.979
[2022/10/10 12:06:31] ppocr DEBUG: production of this book; their contributions, 0.989
[2022/10/10 12:06:31] ppocr DEBUG: have been indispensable to its creation. We, 0.956
[2022/10/10 12:06:31] ppocr DEBUG: would also like to express our gratitude to all, 0.991
[2022/10/10 12:06:31] ppocr DEBUG: the producers for their invaluable opinions, 0.978
[2022/10/10 12:06:31] ppocr DEBUG: and assistance throughout this project. And to, 0.988
[2022/10/10 12:06:31] ppocr DEBUG: the many others whose names are not credited, 0.958
[2022/10/10 12:06:31] ppocr DEBUG: but have made specific input in this book, we, 0.970
[2022/10/10 12:06:31] ppocr DEBUG: thank you for your continuous support., 0.998
[2022/10/10 12:06:31] ppocr DEBUG: The visualized image saved in ./inference_results/img_12.jpg
[2022/10/10 12:06:31] ppocr INFO: The predict total time is 3.2482550144195557
```

Using Paddle Inference to predict, terminal output:

```
[2022/10/10 12:06:28] ppocr DEBUG: dt_boxes num : 11, elapse : 0.3568880558013916
[2022/10/10 12:06:31] ppocr DEBUG: rec_res num  : 11, elapse : 2.6445000171661377
[2022/10/10 12:06:31] ppocr DEBUG: 0  Predict time of doc/imgs_en/img_12.jpg: 3.021s
[2022/10/10 12:06:31] ppocr DEBUG: ACKNOWLEDGEMENTS, 0.997
[2022/10/10 12:06:31] ppocr DEBUG: We would like to thank all the designers and, 0.976
[2022/10/10 12:06:31] ppocr DEBUG: contributors who have been involved in the, 0.979
[2022/10/10 12:06:31] ppocr DEBUG: production of this book; their contributions, 0.989
[2022/10/10 12:06:31] ppocr DEBUG: have been indispensable to its creation. We, 0.956
[2022/10/10 12:06:31] ppocr DEBUG: would also like to express our gratitude to all, 0.991
[2022/10/10 12:06:31] ppocr DEBUG: the producers for their invaluable opinions, 0.978
[2022/10/10 12:06:31] ppocr DEBUG: and assistance throughout this project. And to, 0.988
[2022/10/10 12:06:31] ppocr DEBUG: the many others whose names are not credited, 0.958
[2022/10/10 12:06:31] ppocr DEBUG: but have made specific input in this book, we, 0.970
[2022/10/10 12:06:31] ppocr DEBUG: thank you for your continuous support., 0.998
[2022/10/10 12:06:31] ppocr DEBUG: The visualized image saved in ./inference_results/img_12.jpg
[2022/10/10 12:06:31] ppocr INFO: The predict total time is 3.2482550144195557
```
