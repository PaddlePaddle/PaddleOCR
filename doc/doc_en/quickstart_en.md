
# Quick start of Chinese OCR model

## 1. Prepare for the environment

Please refer to [quick installation](./installation_en.md) to configure the PaddleOCR operating environment.

* Note: Support the use of PaddleOCR through whl package installation，pelease refer  [PaddleOCR Package](./whl_en.md).

## 2.inference models

The detection and recognition models on the mobile and server sides are as follows. For more models  (including multiple languages), please refer to [PP-OCR v2.0 series model list](../doc_ch/models_list.md)

| Model introduction     | Model name      | Recommended scene          | Detection model | Direction Classifier | Recognition model |
| ------------ | --------------- | ----------------|---- | ---------- | -------- |
| Ultra-lightweight Chinese OCR model（8.6M） | ch_ppocr_mobile_v2.0_xx |Mobile-side/Server-side|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar)      |
| Universal Chinese OCR model（146.4M）   | ch_ppocr_server_v2.0_xx |Server-side |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)          |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar)    |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar)  |


* If `wget` is not installed in the windows environment, you can copy the link to the browser to download when downloading the model, then uncompress it and place it in the corresponding directory.

Copy the download address of the `inference model` for detection and recognition in the table above, and uncompress them.

```
mkdir inference && cd inference
# Download the detection model and unzip
wget {url/of/detection/inference_model} && tar xf {name/of/detection/inference_model/package}
# Download the recognition model and unzip
wget {url/of/recognition/inference_model} && tar xf {name/of/recognition/inference_model/package}
# Download the direction classifier model and unzip
wget {url/of/classification/inference_model} && tar xf {name/of/classification/inference_model/package}
cd ..
```

Take the ultra-lightweight model as an example:

```
mkdir inference && cd inference
# Download the detection model of the ultra-lightweight Chinese OCR model and uncompress it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar && tar xf ch_ppocr_mobile_v2.0_det_infer.tar
# Download the recognition model of the ultra-lightweight Chinese OCR model and uncompress it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar && tar xf ch_ppocr_mobile_v2.0_rec_infer.tar
# Download the angle classifier model of the ultra-lightweight Chinese OCR model and uncompress it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar && tar xf ch_ppocr_mobile_v2.0_cls_infer.tar
cd ..
```

After decompression, the file structure should be as follows:

```
├── ch_ppocr_mobile_v2.0_cls_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── ch_ppocr_mobile_v2.0_det_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── ch_ppocr_mobile_v2.0_rec_infer
    ├── inference.pdiparams
    ├── inference.pdiparams.info
    └── inference.pdmodel
```

## 3. Single image or image set prediction

* The following code implements text detection、angle class and recognition process. When performing prediction, you need to specify the path of a single image or image set through the parameter `image_dir`, the parameter `det_model_dir` specifies the path to detect the inference model, the parameter `rec_model_dir` specifies the path to identify the inference model, the parameter `use_angle_cls` specifies whether to use the direction classifier, the parameter `cls_model_dir` specifies the path to identify the direction classifier model, the parameter `use_space_char` specifies whether to predict the space char. The visual results are saved to the `./inference_results` folder by default.



```bash

# Predict a single image specified by image_dir
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True

# Predict imageset specified by image_dir
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/" --det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True

# If you want to use the CPU for prediction, you need to set the use_gpu parameter to False
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_ppocr_mobile_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_mobile_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True --use_gpu=False
```

- Universal Chinese OCR model

Please follow the above steps to download the corresponding models and update the relevant parameters, The example is as follows.

```
# Predict a single image specified by image_dir
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_ppocr_server_v2.0_det_infer/"  --rec_model_dir="./inference/ch_ppocr_server_v2.0_rec_infer/" --cls_model_dir="./inference/ch_ppocr_mobile_v2.0_cls_infer/" --use_angle_cls=True --use_space_char=True
```

* Note
    - If you want to use the recognition model which does not support space char recognition, please update the source code to the latest version and add parameters `--use_space_char=False`.
    - If you do not want to use direction classifier, please update the source code to the latest version and add parameters `--use_angle_cls=False`.


For more text detection and recognition tandem reasoning, please refer to the document tutorial
: [Inference with Python inference engine](./inference_en.md)。

In addition, the tutorial also provides other deployment methods for the Chinese OCR model:
- [Server-side C++ inference](../../deploy/cpp_infer/readme_en.md)
- [Service deployment](../../deploy/pdserving/readme_en.md)
- [End-to-end deployment](../../deploy/lite/readme_en.md)
