
# Reasoning based on Python prediction engine

The inference model (the model saved by `fluid.io.save_inference_model`) is generally a solidified model saved after the model training is completed, and is mostly used to give prediction in deployment.

The model saved during the training process is the checkpoints model, which saves the parameters of the model and is mostly used to resume training.

Compared with the checkpoints model, the inference model will additionally save the structural information of the model. It has superior performance in predicting in deployment and accelerating inferencing, is flexible and convenient, and is suitable for integration with actual systems. For more details, please refer to the document [Classification Framework](https://github.com/PaddlePaddle/PaddleClas/blob/master/docs/zh_CN/extension/paddle_inference.md).

Next, we first introduce how to convert a trained model into an inference model, and then we will introduce text detection, text recognition, and the concatenation of them based on inference model.

- [CONVERT TRAINING MODEL TO INFERENCE MODEL](#CONVERT)
    - [Convert detection model to inference model](#Convert_detection_model)
    - [Convert recognition model to inference model](#Convert_recognition_model)
    - [Convert angle classification model to inference model](#Convert_angle_class_model)


- [TEXT DETECTION MODEL INFERENCE](#DETECTION_MODEL_INFERENCE)
    - [1. LIGHTWEIGHT CHINESE DETECTION MODEL INFERENCE](#LIGHTWEIGHT_DETECTION)
    - [2. DB TEXT DETECTION MODEL INFERENCE](#DB_DETECTION)
    - [3. EAST TEXT DETECTION MODEL INFERENCE](#EAST_DETECTION)
    - [4. SAST TEXT DETECTION MODEL INFERENCE](#SAST_DETECTION)
    - [5. Multilingual model inference](#Multilingual model inference)

- [TEXT RECOGNITION MODEL INFERENCE](#RECOGNITION_MODEL_INFERENCE)
    - [1. LIGHTWEIGHT CHINESE MODEL](#LIGHTWEIGHT_RECOGNITION)
    - [2. CTC-BASED TEXT RECOGNITION MODEL INFERENCE](#CTC-BASED_RECOGNITION)
    - [3. ATTENTION-BASED TEXT RECOGNITION MODEL INFERENCE](#ATTENTION-BASED_RECOGNITION)
    - [4. SRN-BASED TEXT RECOGNITION MODEL INFERENCE](#SRN-BASED_RECOGNITION)
    - [5. TEXT RECOGNITION MODEL INFERENCE USING CUSTOM CHARACTERS DICTIONARY](#USING_CUSTOM_CHARACTERS)
    - [6. MULTILINGUAL MODEL INFERENCE](MULTILINGUAL_MODEL_INFERENCE)

- [ANGLE CLASSIFICATION MODEL INFERENCE](#ANGLE_CLASS_MODEL_INFERENCE)
    - [1. ANGLE CLASSIFICATION MODEL INFERENCE](#ANGLE_CLASS_MODEL_INFERENCE)

- [TEXT DETECTION ANGLE CLASSIFICATION AND RECOGNITION INFERENCE CONCATENATION](#CONCATENATION)
    - [1. LIGHTWEIGHT CHINESE MODEL](#LIGHTWEIGHT_CHINESE_MODEL)
    - [2. OTHER MODELS](#OTHER_MODELS)

<a name="CONVERT"></a>
## CONVERT TRAINING MODEL TO INFERENCE MODEL
<a name="Convert_detection_model"></a>
### Convert detection model to inference model

Download the lightweight Chinese detection model:
```
wget -P ./ch_lite/ https://paddleocr.bj.bcebos.com/20-09-22/mobile/det/ch_ppocr_mobile_v1.1_det_train.tar && tar xf ./ch_lite/ch_ppocr_mobile_v1.1_det_train.tar -C ./ch_lite/
```

The above model is a DB algorithm trained with MobileNetV3 as the backbone. To convert the trained model into an inference model, just run the following command:
```
# -c Set the training algorithm yml configuration file
# -o Set optional parameters
#  Global.checkpoints parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
#  Global.save_inference_dir Set the address where the converted model will be saved.

python3 tools/export_model.py -c configs/det/det_mv3_db_v1.1.yml -o Global.checkpoints=./ch_lite/ch_ppocr_mobile_v1.1_det_train/best_accuracy Global.save_inference_dir=./inference/det_db/
```

When converting to an inference model, the configuration file used is the same as the configuration file used during training. In addition, you also need to set the `Global.checkpoints` and `Global.save_inference_dir` parameters in the configuration file.
`Global.checkpoints` points to the model parameter file saved during training, and `Global.save_inference_dir` is the directory where the generated inference model is saved.
After the conversion is successful, there are two files in the `save_inference_dir` directory:
```
inference/det_db/
  └─  model     Check the program file of inference model
  └─  params    Check the parameter file of the inference model
```

<a name="Convert_recognition_model"></a>
### Convert recognition model to inference model

Download the lightweight Chinese recognition model:
```
wget -P ./ch_lite/ https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_train.tar && tar xf ch_ppocr_mobile_v1.1_rec_train.tar -C ./ch_lite/
```

The recognition model is converted to the inference model in the same way as the detection, as follows:
```
# -c Set the training algorithm yml configuration file
# -o Set optional parameters
#  Global.checkpoints parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
#  Global.save_inference_dir Set the address where the converted model will be saved.

python3 tools/export_model.py -c configs/rec/ch_ppocr_v1.1/rec_chinese_lite_train_v1.1.yml -o Global.checkpoints=./ch_lite/ch_ppocr_mobile_v1.1_rec_train/best_accuracy \
```

If you have a model trained on your own dataset with a different dictionary file, please make sure that you modify the `character_dict_path` in the configuration file to your dictionary file path.

After the conversion is successful, there are two files in the directory:
```
/inference/rec_crnn/
  └─  model     Identify the saved model files
  └─  params    Identify the parameter files of the inference model
```

<a name="Convert_angle_class_model"></a>
### Convert angle classification model to inference model

Download the angle classification model:
```
wget -P ./ch_lite/ https://paddleocr.bj.bcebos.com/20-09-22/cls/ch_ppocr_mobile_v1.1_cls_train.tar && tar xf ./ch_lite/ch_ppocr_mobile_v1.1_cls_train.tar -C ./ch_lite/
```

The angle classification model is converted to the inference model in the same way as the detection, as follows:
```
# -c Set the training algorithm yml configuration file
# -o Set optional parameters
#  Global.checkpoints parameter Set the training model address to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
#  Global.save_inference_dir Set the address where the converted model will be saved.

python3 tools/export_model.py -c configs/cls/cls_mv3.yml -o Global.checkpoints=./ch_lite/ch_ppocr_mobile_v1.1_cls_train/best_accuracy \
        Global.save_inference_dir=./inference/cls/
```

After the conversion is successful, there are two files in the directory:
```
/inference/cls/
  └─  model     Identify the saved model files
  └─  params    Identify the parameter files of the inference model
```


<a name="DETECTION_MODEL_INFERENCE"></a>
## TEXT DETECTION MODEL INFERENCE

The following will introduce the lightweight Chinese detection model inference, DB text detection model inference and EAST text detection model inference. The default configuration is based on the inference setting of the DB text detection model.
Because EAST and DB algorithms are very different, when inference, it is necessary to **adapt the EAST text detection algorithm by passing in corresponding parameters**.

<a name="LIGHTWEIGHT_DETECTION"></a>
### 1. LIGHTWEIGHT CHINESE DETECTION MODEL INFERENCE

For lightweight Chinese detection model inference, you can execute the following commands:

```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/"
```

The visual text detection results are saved to the ./inference_results folder by default, and the name of the result file is prefixed with'det_res'. Examples of results are as follows:

![](../imgs_results/det_res_2.jpg)

By setting the size of the parameter `det_max_side_len`, the maximum value of picture normalization in the detection algorithm is changed. When the length and width of the picture are less than det_max_side_len, the original picture is used for prediction, otherwise the picture is scaled to the maximum value for prediction. This parameter is set to det_max_side_len=960 by default. If the resolution of the input picture is relatively large and you want to use a larger resolution for prediction, you can execute the following command:

```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/" --det_max_side_len=1200
```

If you want to use the CPU for prediction, execute the command as follows
```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/" --use_gpu=False
```

<a name="DB_DETECTION"></a>
### 2. DB TEXT DETECTION MODEL INFERENCE

First, convert the model saved in the DB text detection training process into an inference model. Taking the model based on the Resnet50_vd backbone network and trained on the ICDAR2015 English dataset as an example ([model download link](https://paddleocr.bj.bcebos.com/det_r50_vd_db.tar)), you can use the following command to convert:

```
# Set the yml configuration file of the training algorithm after -c
# The Global.checkpoints parameter sets the address of the training model to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
# The Global.save_inference_dir parameter sets the address where the converted model will be saved.

python3 tools/export_model.py -c configs/det/det_r50_vd_db.yml -o Global.checkpoints="./models/det_r50_vd_db/best_accuracy" Global.save_inference_dir="./inference/det_db"
```

DB text detection model inference, you can execute the following command:

```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_db/"
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![](../imgs_results/det_res_img_10_db.jpg)

**Note**: Since the ICDAR2015 dataset has only 1,000 training images, mainly for English scenes, the above model has very poor detection result on Chinese text images.

<a name="EAST_DETECTION"></a>
### 3. EAST TEXT DETECTION MODEL INFERENCE

First, convert the model saved in the EAST text detection training process into an inference model. Taking the model based on the Resnet50_vd backbone network and trained on the ICDAR2015 English dataset as an example ([model download link](https://paddleocr.bj.bcebos.com/det_r50_vd_east.tar)), you can use the following command to convert:

```
# Set the yml configuration file of the training algorithm after -c
# The Global.checkpoints parameter sets the address of the training model to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
# The Global.save_inference_dir parameter sets the address where the converted model will be saved.

python3 tools/export_model.py -c configs/det/det_r50_vd_east.yml -o Global.checkpoints="./models/det_r50_vd_east/best_accuracy" Global.save_inference_dir="./inference/det_east"
```

**For EAST text detection model inference, you need to set the parameter ``--det_algorithm="EAST"``**, run the following command:

```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_east/" --det_algorithm="EAST"
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![](../imgs_results/det_res_img_10_east.jpg)

**Note**: EAST post-processing locality aware NMS has two versions: Python and C++. The speed of C++ version is obviously faster than that of Python version. Due to the compilation version problem of NMS of C++ version, C++ version NMS will be called only in Python 3.5 environment, and python version NMS will be called in other cases.


<a name="SAST_DETECTION"></a>
### 4. SAST TEXT DETECTION MODEL INFERENCE
#### (1). Quadrangle text detection model (ICDAR2015)  
First, convert the model saved in the SAST text detection training process into an inference model. Taking the model based on the Resnet50_vd backbone network and trained on the ICDAR2015 English dataset as an example ([model download link](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_icdar2015.tar)), you can use the following command to convert:

```
python3 tools/export_model.py -c configs/det/det_r50_vd_sast_icdar15.yml -o Global.checkpoints="./models/sast_r50_vd_icdar2015/best_accuracy" Global.save_inference_dir="./inference/det_sast_ic15"
```

**For SAST quadrangle text detection model inference, you need to set the parameter `--det_algorithm="SAST"`**, run the following command:

```
python3 tools/infer/predict_det.py --det_algorithm="SAST" --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_sast_ic15/"
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![](../imgs_results/det_res_img_10_sast.jpg)

#### (2). Curved text detection model (Total-Text)  
First, convert the model saved in the SAST text detection training process into an inference model. Taking the model based on the Resnet50_vd backbone network and trained on the Total-Text English dataset as an example ([model download link](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_total_text.tar)), you can use the following command to convert:

```
python3 tools/export_model.py -c configs/det/det_r50_vd_sast_totaltext.yml -o Global.checkpoints="./models/sast_r50_vd_total_text/best_accuracy" Global.save_inference_dir="./inference/det_sast_tt"
```

**For SAST curved text detection model inference, you need to set the parameter `--det_algorithm="SAST"` and `--det_sast_polygon=True`**, run the following command:

```
python3 tools/infer/predict_det.py --det_algorithm="SAST" --image_dir="./doc/imgs_en/img623.jpg" --det_model_dir="./inference/det_sast_tt/" --det_sast_polygon=True
```

The visualized text detection results are saved to the `./inference_results` folder by default, and the name of the result file is prefixed with 'det_res'. Examples of results are as follows:

![](../imgs_results/det_res_img623_sast.jpg)

**Note**: SAST post-processing locality aware NMS has two versions: Python and C++. The speed of C++ version is obviously faster than that of Python version. Due to the compilation version problem of NMS of C++ version, C++ version NMS will be called only in Python 3.5 environment, and python version NMS will be called in other cases.

<a name="RECOGNITION_MODEL_INFERENCE"></a>
## TEXT RECOGNITION MODEL INFERENCE

The following will introduce the lightweight Chinese recognition model inference, other CTC-based and Attention-based text recognition models inference. For Chinese text recognition, it is recommended to choose the recognition model based on CTC loss. In practice, it is also found that the result of the model based on Attention loss is not as good as the one based on CTC loss. In addition, if the characters dictionary is modified during training, make sure that you use the same characters set during inferencing. Please check below for details.


<a name="LIGHTWEIGHT_RECOGNITION"></a>
### 1. LIGHTWEIGHT CHINESE TEXT RECOGNITION MODEL REFERENCE

For lightweight Chinese recognition model inference, you can execute the following commands:

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/ch/word_4.jpg" --rec_model_dir="./inference/rec_crnn/"
```

![](../imgs_words/ch/word_4.jpg)

After executing the command, the prediction results (recognized text and score) of the above image will be printed on the screen.

Predicts of ./doc/imgs_words/ch/word_4.jpg:['实力活力', 0.89552695]


<a name="CTC-BASED_RECOGNITION"></a>
### 2. CTC-BASED TEXT RECOGNITION MODEL INFERENCE

Taking STAR-Net as an example, we introduce the recognition model inference based on CTC loss. CRNN and Rosetta are used in a similar way, by setting the recognition algorithm parameter `rec_algorithm`.

First, convert the model saved in the STAR-Net text recognition training process into an inference model. Taking the model based on Resnet34_vd backbone network, using MJSynth and SynthText (two English text recognition synthetic datasets) for training, as an example ([model download address](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_ctc.tar)). It can be converted as follow:

```
# Set the yml configuration file of the training algorithm after -c
# The Global.checkpoints parameter sets the address of the training model to be converted without adding the file suffix .pdmodel, .pdopt or .pdparams.
# The Global.save_inference_dir parameter sets the address where the converted model will be saved.

python3 tools/export_model.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml -o Global.checkpoints="./models/rec_r34_vd_tps_bilstm_ctc/best_accuracy" Global.save_inference_dir="./inference/starnet"
```

For STAR-Net text recognition model inference, execute the following commands:

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./inference/starnet/" --rec_image_shape="3, 32, 100" --rec_char_type="en"
```

<a name="ATTENTION-BASED_RECOGNITION"></a>
### 3. ATTENTION-BASED TEXT RECOGNITION MODEL INFERENCE
![](../imgs_words_en/word_336.png)

After executing the command, the recognition result of the above image is as follows:

Predicts of ./doc/imgs_words_en/word_336.png:['super', 0.9999555]

**Note**：Since the above model refers to [DTRB](https://arxiv.org/abs/1904.01906) text recognition training and evaluation process, it is different from the training of lightweight Chinese recognition model in two aspects:

- The image resolution used in training is different: the image resolution used in training the above model is [3，32，100], while during our Chinese model training, in order to ensure the recognition effect of long text, the image resolution used in training is [3, 32, 320]. The default shape parameter of the inference stage is the image resolution used in training phase, that is [3, 32, 320]. Therefore, when running inference of the above English model here, you need to set the shape of the recognition image through the parameter `rec_image_shape`.

- Character list: the experiment in the DTRB paper is only for 26 lowercase English characters and 10 numbers, a total of 36 characters. All upper and lower case characters are converted to lower case characters, and characters not in the above list are ignored and considered as spaces. Therefore, no characters dictionary file is used here, but a dictionary is generated by the below command. Therefore, the parameter `rec_char_type` needs to be set during inference, which is specified as "en" in English.

```
self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
dict_character = list(self.character_str)
```

<a name="SRN-BASED_RECOGNITION"></a>
### 4. SRN-BASED TEXT RECOGNITION MODEL INFERENCE

The recognition model based on SRN requires additional setting of the recognition algorithm parameter --rec_algorithm="SRN".
At the same time, it is necessary to ensure that the predicted shape is consistent with the training, such as: --rec_image_shape="1, 64, 256"

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" \
                                    --rec_model_dir="./inference/srn/" \
                                    --rec_image_shape="1, 64, 256" \
                                    --rec_char_type="en" \
                                    --rec_algorithm="SRN"
```


<a name="USING_CUSTOM_CHARACTERS"></a>
### 5. TEXT RECOGNITION MODEL INFERENCE USING CUSTOM CHARACTERS DICTIONARY
If the chars dictionary is modified during training, you need to specify the new dictionary path by setting the parameter `rec_char_dict_path` when using your inference model to predict.

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./your inference model" --rec_image_shape="3, 32, 100" --rec_char_type="en" --rec_char_dict_path="your text dict path"
```

<a name="MULTILINGUAL_MODEL_INFERENCE"></a>
### 6. MULTILINGAUL MODEL INFERENCE
If you need to predict other language models, when using inference model prediction, you need to specify the dictionary path used by `--rec_char_dict_path`. At the same time, in order to get the correct visualization results,
You need to specify the visual font path through `--vis_font_path`. There are small language fonts provided by default under the `doc/` path, such as Korean recognition:

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/korean/1.jpg" --rec_model_dir="./your inference model" --rec_char_type="korean" --rec_char_dict_path="ppocr/utils/dict/korean_dict.txt" --vis_font_path="doc/korean.ttf"
```
![](../imgs_words/korean/1.jpg)

After executing the command, the prediction result of the above figure is:

``` text
2020-09-19 16:15:05,076-INFO:      index: [205 206  38  39]
2020-09-19 16:15:05,077-INFO:      word : 바탕으로
2020-09-19 16:15:05,077-INFO:      score: 0.9171358942985535
```

<a name="ANGLE_CLASSIFICATION_MODEL_INFERENCE"></a>
## ANGLE CLASSIFICATION MODEL INFERENCE

The following will introduce the angle classification model inference.


<a name="ANGLE_CLASS_MODEL_INFERENCE"></a>
### 1.ANGLE CLASSIFICATION MODEL INFERENCE

For angle classification model inference, you can execute the following commands:

```
python3 tools/infer/predict_cls.py --image_dir="./doc/imgs_words/ch/word_4.jpg" --cls_model_dir="./inference/cls/"
```

![](../imgs_words/ch/word_4.jpg)

After executing the command, the prediction results (classification angle and score) of the above image will be printed on the screen.

Predicts of ./doc/imgs_words/ch/word_4.jpg:['0', 0.9999963]


<a name="CONCATENATION"></a>
## TEXT DETECTION ANGLE CLASSIFICATION AND RECOGNITION INFERENCE CONCATENATION

<a name="LIGHTWEIGHT_CHINESE_MODEL"></a>
### 1. LIGHTWEIGHT CHINESE MODEL

When performing prediction, you need to specify the path of a single image or a folder of images through the parameter `image_dir`, the parameter `det_model_dir` specifies the path to detect the inference model, the parameter `cls_model_dir` specifies the path to angle classification inference model and the parameter `rec_model_dir` specifies the path to identify the inference model. The parameter `use_angle_cls` is used to control whether to enable the angle classification model.The visualized recognition results are saved to the `./inference_results` folder by default.

```
# use direction classifier
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/" --cls_model_dir="./inference/cls/" --rec_model_dir="./inference/rec_crnn/" --use_angle_cls=true

# not use use direction classifier
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/" --rec_model_dir="./inference/rec_crnn/"
```

After executing the command, the recognition result image is as follows:

![](../imgs_results/2.jpg)

<a name="OTHER_MODELS"></a>
### 2. OTHER MODELS

If you want to try other detection algorithms or recognition algorithms, please refer to the above text detection model inference and text recognition model inference, update the corresponding configuration and model.

**Note: due to the limitation of rotation logic of detected box, SAST curved text detection model (using the parameter `det_sast_polygon=True`) is not supported for model combination yet.**

The following command uses the combination of the EAST text detection and STAR-Net text recognition:

```
python3 tools/infer/predict_system.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_east/" --det_algorithm="EAST" --rec_model_dir="./inference/starnet/" --rec_image_shape="3, 32, 100" --rec_char_type="en"
```

After executing the command, the recognition result image is as follows:

![](../imgs_results/img_10.jpg)
