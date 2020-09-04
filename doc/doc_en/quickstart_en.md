
# Quick start of Chinese OCR model

## 1. Prepare for the environment

Please refer to [quick installation](./installation_en.md) to configure the PaddleOCR operating environment.

*Note: Support the use of PaddleOCR through whl package installation，pelease refer  [PaddleOCR Package](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/whl_en.md)。*

## 2.inference models

| Name | Introduction | Detection model | Recognition model | Recognition model with space support |
|-|-|-|-|-|
|chinese_db_crnn_mobile| Ultra-lightweight Chinese OCR model |[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_enhance.tar)
|chinese_db_crnn_server| Universal Chinese OCR model |[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/ch_models/ch_det_r50_vd_db.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn.tar)|[inference model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance_infer.tar) / [pretrained model](https://paddleocr.bj.bcebos.com/ch_models/ch_rec_r34_vd_crnn_enhance.tar)

* If wget is not installed in the windows environment, you can copy the link to the browser to download when downloading the model, and uncompress it and place it in the corresponding directory.

Copy the download address of the `inference model` for detection and recognition in the table above, and uncompress them.

```
mkdir inference && cd inference
# Download the detection model and unzip
wget {url/of/detection/inference_model} && tar xf {name/of/detection/inference_model/package}
# Download the recognition model and unzip
wget {url/of/recognition/inference_model} && tar xf {name/of/recognition/inference_model/package}
cd ..
```

Take the ultra-lightweight model as an example:

```
mkdir inference && cd inference
# Download the detection model of the ultra-lightweight Chinese OCR model and uncompress it
wget https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar && tar xf ch_det_mv3_db_infer.tar
# Download the recognition model of the ultra-lightweight Chinese OCR model and uncompress it
wget https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar && tar xf ch_rec_mv3_crnn_infer.tar
cd ..
```

After decompression, the file structure should be as follows:

```
|-inference
    |-ch_rec_mv3_crnn
        |- model
        |- params
    |-ch_det_mv3_db
        |- model
        |- params
    ...
```

## 3. Single image or image set prediction

* The following code implements text detection and recognition process. When performing prediction, you need to specify the path of a single image or image set through the parameter `image_dir`, the parameter `det_model_dir` specifies the path to detect the inference model, and the parameter `rec_model_dir` specifies the path to identify the inference model. The visual results are saved to the `./inference_results` folder by default.



```bash

# Predict a single image specified by image_dir
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_det_mv3_db/"  --rec_model_dir="./inference/ch_rec_mv3_crnn/"

# Predict imageset specified by image_dir
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/" --det_model_dir="./inference/ch_det_mv3_db/"  --rec_model_dir="./inference/ch_rec_mv3_crnn/"

# If you want to use the CPU for prediction, you need to set the use_gpu parameter to False
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_det_mv3_db/"  --rec_model_dir="./inference/ch_rec_mv3_crnn/" --use_gpu=False
```

- Universal Chinese OCR model

Please follow the above steps to download the corresponding models and update the relevant parameters, The example is as follows.

```
# Predict a single image specified by image_dir
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/11.jpg" --det_model_dir="./inference/ch_det_r50_vd_db/"  --rec_model_dir="./inference/ch_rec_r34_vd_crnn/"
```

- Universal Chinese OCR model with the support of space

Please follow the above steps to download the corresponding models and update the relevant parameters, The example is as follows.


* Note: Please update the source code to the latest version and add parameters `--use_space_char=True`

```
# Predict a single image specified by image_dir
python3 tools/infer/predict_system.py --image_dir="./doc/imgs_en/img_12.jpg" --det_model_dir="./inference/ch_det_r50_vd_db/"  --rec_model_dir="./inference/ch_rec_r34_vd_crnn_enhance/" --use_space_char=True
```

For more text detection and recognition tandem reasoning, please refer to the document tutorial
: [Inference with Python inference engine](./inference_en.md)。

In addition, the tutorial also provides other deployment methods for the Chinese OCR model:
- [Server-side C++ inference](../../deploy/cpp_infer/readme_en.md)
- [Service deployment](./serving_en.md)
- [End-to-end deployment](../../deploy/lite/readme_en.md)
