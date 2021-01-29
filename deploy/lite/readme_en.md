
# Tutorial of PaddleOCR Mobile deployment

This tutorial will introduce how to use [paddle-lite](https://github.com/PaddlePaddle/Paddle-Lite) to deploy paddleOCR ultra-lightweight Chinese and English detection models on mobile phones.

paddle-lite is a lightweight inference engine for PaddlePaddle.
It provides efficient inference capabilities for mobile phones and IoTs,
and extensively integrates cross-platform hardware to provide lightweight
deployment solutions for end-side deployment issues.

## 1. Preparation

- Computer (for Compiling Paddle Lite)
- Mobile phone (arm7 or arm8)

***Note: PaddleOCR lite deployment currently does not support dynamic graph models, only models saved with static graph. The static branch of PaddleOCR is `develop`.***

## 2. Build PaddleLite library
1. [Docker](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#docker)
2. [Linux](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#linux)
3. [MAC OS](https://paddle-lite.readthedocs.io/zh/latest/source_compile/compile_env.html#mac-os)

## 3. Prepare prebuild library for android and ios

### 3.1 Download prebuild library
|Platform|Prebuild library Download Link|
|---|---|
|Android|[arm7](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.8/inference_lite_lib.android.armv7.gcc.c++_shared.with_extra.with_cv.tar.gz) / [arm8](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.8/inference_lite_lib.android.armv8.gcc.c++_shared.with_extra.with_cv.tar.gz)|
|IOS|[arm7](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.8/inference_lite_lib.ios.armv7.with_cv.with_extra.with_log.tiny_publish.tar.gz) / [arm8](https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.8/inference_lite_lib.ios.armv8.with_cv.with_extra.with_log.tiny_publish.tar.gz)|

note: The above pre-build inference library is compiled from the PaddleLite `release/v2.8` branch. For more information about PaddleLite 2.8, please refer to [link](https://github.com/PaddlePaddle/Paddle-Lite/releases/tag/v2.8).

### 3.2 Compile prebuild library (Recommended)
```
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
# checkout to Paddle-Lite release/v2.8 branch
git checkout release/v2.8
./lite/tools/build_android.sh  --arch=armv8  --with_cv=ON --with_extra=ON
```

The structure of the prediction library is as follows:

```
inference_lite_lib.android.armv8/
|-- cxx                                        C++ prebuild library
|   |-- include                                C++
|   |   |-- paddle_api.h
|   |   |-- paddle_image_preprocess.h
|   |   |-- paddle_lite_factory_helper.h
|   |   |-- paddle_place.h
|   |   |-- paddle_use_kernels.h
|   |   |-- paddle_use_ops.h
|   |   `-- paddle_use_passes.h
|   `-- lib  
|       |-- libpaddle_api_light_bundled.a             C++ static library
|       `-- libpaddle_light_api_shared.so             C++ dynamic library
|-- java                                     Java predict library
|   |-- jar
|   |   `-- PaddlePredictor.jar
|   |-- so
|   |   `-- libpaddle_lite_jni.so
|   `-- src
|-- demo                                     C++ and java demo
|   |-- cxx  
|   `-- java  
```


## 4. Inference Model Optimization

Paddle Lite provides a variety of strategies to automatically optimize the original training model, including quantization, sub-graph fusion, hybrid scheduling, Kernel optimization and so on. In order to make the optimization process more convenient and easy to use, Paddle Lite provide opt tools to automatically complete the optimization steps and output a lightweight, optimal executable model.

If you have prepared the model file ending in `.nb`, you can skip this step.

The following table also provides a series of models that can be deployed on mobile phones to recognize Chinese.
You can directly download the optimized model.

| Version | Introduction | Model size | Detection model | Text Direction model | Recognition model | Paddle Lite branch |
| --- | --- | --- | --- | --- | --- | --- |
| V1.1 | extra-lightweight chinese OCR optimized model | 8.1M | [Download](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_det_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_cls_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_rec_opt.nb) | develop |
| [slim] V1.1 | extra-lightweight chinese OCR optimized model | 3.5M | [Download](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_det_prune_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_cls_quant_opt.nb) | [Download](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.1_rec_quant_opt.nb) | develop |
| V1.0 | lightweight Chinese OCR optimized model | 8.6M | [Download](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.0_det_opt.nb) | - | [Download](https://paddleocr.bj.bcebos.com/20-09-22/mobile/lite/ch_ppocr_mobile_v1.0_rec_opt.nb) | develop |

If the model to be deployed is not in the above table, you need to follow the steps below to obtain the optimized model.

```
git clone https://github.com/PaddlePaddle/Paddle-Lite.git
cd Paddle-Lite
git checkout release/v2.7
./lite/tools/build.sh build_optimize_tool
```

The `opt` tool can be obtained by compiling Paddle Lite.

After the compilation is complete, the opt file is located under `build.opt/lite/api/`.

The `opt` can optimize the inference model saved by paddle.io.save_inference_model to get the model that the paddlelite API can use.

The usage of opt is as follows：
```
# 【Recommend】V1.1 is better than V1.0. steps for convert V1.1 model to nb file are as follows
wget  https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/det/ch_ppocr_mobile_v1.1_det_prune_infer.tar && tar xf  ch_ppocr_mobile_v1.1_det_prune_infer.tar
wget  https://paddleocr.bj.bcebos.com/20-09-22/mobile-slim/rec/ch_ppocr_mobile_v1.1_rec_quant_infer.tar && tar xf  ch_ppocr_mobile_v1.1_rec_quant_infer.tar

./opt --model_file=./ch_ppocr_mobile_v1.1_det_prune_infer/model  --param_file=./ch_ppocr_mobile_v1.1_det_prune_infer/params  --optimize_out=./ch_ppocr_mobile_v1.1_det_prune_opt --valid_targets=arm
./opt --model_file=./ch_ppocr_mobile_v1.1_rec_quant_infer/model  --param_file=./ch_ppocr_mobile_v1.1_rec_quant_infer/params  --optimize_out=./ch_ppocr_mobile_v1.1_rec_quant_opt --valid_targets=arm

# or use V1.0 model
wget  https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db_infer.tar && tar xf ch_det_mv3_db_infer.tar
wget  https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn_infer.tar && tar xf ch_rec_mv3_crnn_infer.tar

./opt --model_file=./ch_det_mv3_db/model --param_file=./ch_det_mv3_db/params --optimize_out_type=naive_buffer --optimize_out=./ch_det_mv3_db_opt --valid_targets=arm
./opt --model_file=./ch_rec_mv3_crnn/model --param_file=./ch_rec_mv3_crnn/params --optimize_out_type=naive_buffer --optimize_out=./ch_rec_mv3_crnn_opt --valid_targets=arm

```

When the above code command is completed, there will be two more files `.nb` in the current directory, which is the converted model file.

## 5. Run optimized model on Phone

1. Prepare an Android phone with arm8. If the compiled prediction library and opt file are armv7, you need an arm7 phone and modify ARM_ABI = arm7 in the Makefile.

2. Make sure the phone is connected to the computer, open the USB debugging option of the phone, and select the file transfer mode.

3. Install the adb tool on the computer.
    3.1 Install ADB for MAC
    ```
    brew cask install android-platform-tools
    ```
    3.2 Install ADB for Linux
    ```
    sudo apt update
    sudo apt install -y wget adb
    ```
    3.3 Install ADB for windows
    [Download Link](https://developer.android.com/studio)

    Verify whether adb is installed successfully
    ```
    $ adb devices

    List of devices attached
    744be294    device
    ```

    If there is `device` output, it means the installation was successful.

4. Prepare optimized models, prediction library files, test images and dictionary files used.

```
 git clone https://github.com/PaddlePaddle/PaddleOCR.git
 cd PaddleOCR/deploy/lite/
 # run prepare.sh
 sh prepare.sh /{lite prediction library path}/inference_lite_lib.android.armv8

 #
 cd /{lite prediction library path}/inference_lite_lib.android.armv8/
 cd demo/cxx/ocr/
 # copy paddle-lite C++ .so file to debug/ directory
 cp ../../../cxx/lib/libpaddle_light_api_shared.so ./debug/

 cd inference_lite_lib.android.armv8/demo/cxx/ocr/
 cp ../../../cxx/lib/libpaddle_light_api_shared.so ./debug/

```

Prepare the test image, taking `PaddleOCR/doc/imgs/11.jpg` as an example, copy the image file to the `demo/cxx/ocr/debug/` folder.
Prepare the model files optimized by the lite opt tool, `ch_det_mv3_db_opt.nb, ch_rec_mv3_crnn_opt.nb`,
and place them under the `demo/cxx/ocr/debug/` folder.


The structure of the OCR demo is as follows after the above command is executed:
```
demo/cxx/ocr/
|-- debug/  
|   |--ch_ppocr_mobile_v1.1_det_prune_opt.nb           Detection model
|   |--ch_ppocr_mobile_v1.1_rec_quant_opt.nb           Recognition model
|   |--ch_ppocr_mobile_cls_quant_opt.nb                Text direction classification model
|   |--11.jpg                           Image for OCR
|   |--ppocr_keys_v1.txt                Dictionary file
|   |--libpaddle_light_api_shared.so    C++ .so file
|   |--config.txt                       Config file
|-- config.txt  
|-- crnn_process.cc  
|-- crnn_process.h
|-- db_post_process.cc  
|-- db_post_process.h
|-- Makefile  
|-- ocr_db_crnn.cc  

```

#### Note:
1. ppocr_keys_v1.txt is a Chinese dictionary file.
If the nb model is used for English recognition or other language recognition, dictionary file should be replaced with a dictionary of the corresponding language.
PaddleOCR provides a variety of dictionaries under ppocr/utils/, including:
```
dict/french_dict.txt     # french
dict/german_dict.txt     # german
ic15_dict.txt       # english
dict/japan_dict.txt      # japan
dict/korean_dict.txt     # korean
ppocr_keys_v1.txt   # chinese
```

2. `config.txt`  of the detector and classifier, as shown below:
```
max_side_len  960         #  Limit the maximum image height and width to 960
det_db_thresh  0.3        # Used to filter the binarized image of DB prediction, setting 0.-0.3 has no obvious effect on the result
det_db_box_thresh  0.5    # DDB post-processing filter box threshold, if there is a missing box detected, it can be reduced as appropriate
det_db_unclip_ratio  1.6  # Indicates the compactness of the text box, the smaller the value, the closer the text box to the text
use_direction_classify  0  # Whether to use the direction classifier, 0 means not to use, 1 means to use
```

5. Run Model on phone

```
cd inference_lite_lib.android.armv8/demo/cxx/ocr/
make -j
mv ocr_db_crnn ./debug/
adb push debug /data/local/tmp/
adb shell
cd /data/local/tmp/debug
export LD_LIBRARY_PATH=/data/local/tmp/debug:$LD_LIBRARY_PATH
# run model
 ./ocr_db_crnn ch_ppocr_mobile_v1.1_det_prune_opt.nb  ch_ppocr_mobile_v1.1_rec_quant_opt.nb  ch_ppocr_mobile_cls_quant_opt.nb  ./11.jpg  ppocr_keys_v1.txt
```

The outputs are as follows:

<div align="center">
    <img src="../imgs_results/lite_demo.png" width="600">
</div>

## FAQ

Q1: What if I want to change the model, do I need to run it again according to the process?
A1: If you have performed the above steps, you only need to replace the .nb model file to complete the model replacement.

Q2: How to test with another picture?
A2: Replace the .jpg test image under `./debug` with the image you want to test, and run `adb push` to push new image to the phone.

Q3: How to package it into the mobile APP?
A3: This demo aims to provide the core algorithm part that can run OCR on mobile phones.  Further,
PaddleOCR/deploy/android_demo is an example of encapsulating this demo into a mobile app for reference.
