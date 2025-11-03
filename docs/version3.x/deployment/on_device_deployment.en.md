---
comments: true
---

# OCR On-Device Deployment Demo Usage Guide

- [Quick Start](#quick-start)
    - [Environment Preparation](#environment-preparation)
    - [Deployment Steps](#deployment-steps)
- [Code Introduction](#code-introduction)
- [Project Explanation](#project-explanation)
- [Advanced Usage](#advanced-usage)
    - [Update Prediction Library](#update-prediction-library) 
    - [Convert NB Model](#convert-nb-model) 
    - [Update Model, Label File, and Prediction Image](#update-model-label-file-and-prediction-image)
        - [Update Model](#update-model)
        - [Update Label File](#update-label-file)
        - [Update Prediction Image](#update-prediction-image)
    - [Update Input/Output Preprocessing](#update-inputoutput-preprocessing)

This guide mainly introduces how to run the PaddleX on-device deployment demo for OCR text recognition on an Android shell.

The following OCR models are supported in this guide:

- PP-OCRv3_mobile (cpu)
- PP-OCRv4_mobile (cpu)
- PP-OCRv5_mobile (cpu)

## Quick Start

### Environment Preparation

1. Install the CMAKE compilation tool in your local environment and download an NDK package for your current system from the [Android NDK official website](https://developer.android.google.cn/ndk/downloads). For example, if developing on a Mac, download the NDK package for the Mac platform from the Android NDK official website.

    **Environment Requirements**

    - `CMake >= 3.10` (the minimum version has not been verified; 3.20 or above is recommended)
    - `Android NDK >= r17c` (the minimum version has not been verified; r20b or above is recommended)

    **Test Environment Used in This Guide**:

    - `cmake == 3.20.0`
    - `android-ndk == r20b`

2. Prepare an Android phone and enable USB debugging mode. Method: `Phone Settings -> Find Developer Options -> Enable Developer Options and USB Debugging Mode`

3. Install the ADB tool on your computer for debugging. The installation methods for ADB are as follows:

    3.1. Install ADB on a Mac:

    ```shell
    brew cask install android-platform-tools
    ```

    3.2. Install ADB on Linux:

    ```shell
    sudo apt update
    sudo apt install -y wget adb
    ```

    3.3. Install ADB on Windows:

    For installation on Windows, download and install the ADB software package from Google's Android platform: [Link](https://developer.android.com/studio)

    Open a terminal, connect your phone to the computer, and enter the following command in the terminal:

    ```shell
    adb devices
    ```

    If there is a `device` output, the installation is successful.

    ```shell
    List of devices attached
    744be294    device
    ```

### Material Preparation

1. Clone the `feature/paddle-x` branch of the `Paddle-Lite-Demo` repository to the `PaddleX-Lite-Deploy` directory.

    ```shell
    git clone -b feature/paddle-x https://github.com/PaddlePaddle/Paddle-Lite-Demo.git PaddleX-Lite-Deploy
    ```

2. Fill out the [questionnaire](https://paddle.wjx.cn/vm/eaaBo0H.aspx#) to download the compressed package. Place the compressed package in the specified extraction directory, switch to the specified extraction directory, and execute the extraction command.

    ```shell
    # 1. Switch to the specified extraction directory
    cd PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo

    # 2. Execute the extraction command
    unzip ocr.zip
    ```

### Deployment Steps

1. Switch the working directory to `PaddleX-Lite-Deploy/libs` and run the `download.sh` script to download the required Paddle Lite prediction library. This step only needs to be executed once to support each demo.

2. Switch the working directory to `PaddleX-Lite-Deploy/ocr/assets` and run the `download.sh` script to download the [paddle_lite_opt tool](https://www.paddlepaddle.org.cn/lite/v2.10/user_guides/model_optimize_tool.html)-optimized NB model file, prediction images, dictionary files, and other materials.

3. Switch the working directory to `PaddleX-Lite-Deploy/ocr/android/shell/cxx/ppocr_demo` and run the `build.sh` script to complete the compilation of the executable file.

4. Switch the working directory to `PaddleX-Lite-Deploy/ocr/android/shell/cxx/ppocr_demo` and run the `run.sh` script to complete the on-device prediction.

**Notes**:

- Before running the `build.sh` script, change the path specified by `NDK_ROOT` to the actual installation path of NDK.
- On Windows systems, you can use Git Bash to execute the deployment steps.
- If compiling on a Windows system, set `CMAKE_SYSTEM_NAME` to `windows` in `CMakeLists.txt`.
- If compiling on a Mac system, set `CMAKE_SYSTEM_NAME` to `darwin` in `CMakeLists.txt`.
- Keep the ADB connection active when running the `run.sh` script.
- The `download.sh` and `run.sh` scripts support passing parameters to specify the model. If no model is specified, the `PP-OCRv5_mobile` model is used by default. The following models are currently supported:
    - `PP-OCRv3_mobile`
    - `PP-OCRv4_mobile`
    - `PP-OCRv5_mobile`

Here is an example of the actual operation:

```shell
# 1. Download the required Paddle Lite prediction library
cd PaddleX-Lite-Deploy/libs
sh download.sh

# 2. Download the paddle_lite_opt tool-optimized NB model file, prediction images, dictionary files, and other materials
cd ../ocr/assets
sh download.sh PP-OCRv5_mobile

# 3. Complete the compilation of the executable file
cd ../android/shell/ppocr_demo
sh build.sh

# 4. Prediction
sh run.sh PP-OCRv5_mobile
```

The output is as follows:

```text
The detection visualized image saved in ./test_img_result.jpg
0       纯臻营养护发素  0.998541
1       产品信息/参数   0.999094
2       (45元/每公斤，100公斤起订）     0.948841
3       每瓶22元，1000瓶起订)   0.961245
4       【品牌】：代加工方式/OEMODM     0.970401
5       【品名】：纯臻营养护发素        0.977496
6       ODMOEM  0.955396
7       【产品编号】：YM-X-3011 0.977864
8       【净含量】：220ml       0.970538
9       【适用人群】：适合所有肤质      0.995907
10      【主要成分】：鲸蜡硬脂醇、燕麦β-葡聚    0.975813
11      糖、椰油酰胺丙基甜菜碱、泛醌    0.964397
12      （成品包材）    0.97298
13      【主要功能】：可紧致头发磷层，从而达到  0.989097
14      即时持久改善头发光泽的效果，给干燥的头  0.990088
15      发足够的滋养    0.998037
``` 

![Prediction Result](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipeline_deploy/edge_PP-OCRv5_mobile.jpg)

## Code Introduction

```
.
├── ...
├── ocr 
│    ├── ...
│    ├── android
│    │    ├── ...
│    │    └── shell
│    │        └── ppocr_demo
│    │            ├── src # Contains prediction code
│    │            |   ├── cls_process.cc # Full inference process for orientation classifier, including preprocessing, prediction, and postprocessing
│    │            |   ├── rec_process.cc # Full inference process for recognition model CRNN, including preprocessing, prediction, and postprocessing
│    │            |   ├── det_process.cc # Full inference process for detection model CRNN, including preprocessing, prediction, and postprocessing
│    │            |   ├── det_post_process.cc # Postprocessing file for detection model DB
│    │            |   ├── pipeline.cc # Full inference process code for OCR text recognition demo
│    │            |   └── MakeFile # MakeFile file for prediction code
│    │            |   
│    │            ├── CMakeLists.txt # CMake file that defines the compilation method for the executable
│    │            ├── README.md
│    │            ├── build.sh # Used for compiling the executable
│    │            └── run.sh # Used for prediction
│    └── assets # Stores models, test images, label files, and config files
│        ├── images # Stores test images
│        ├── labels # Stores dictionary files (see remarks below for details)
│        ├── models # Stores nb models
│        ├── config.txt
│        └── download.sh # Download script for paddle_lite_opt tool-optimized models
└── libs # Stores prediction libraries and OpenCV libraries for different platforms.
    ├── ...
    └── download.sh # Download script for Paddle Lite prediction libraries and OpenCV libraries
```

**Remarks**:

 - The `PaddleX-Lite-Deploy/ocr/assets/labels/` directory contains the dictionary files `ppocr_keys_v1.txt` for PP-OCRv3 and PP-OCRv4 models, and `ppocr_keys_ocrv5.txt` for the PP-OCRv5 model. The appropriate dictionary file is automatically selected during inference based on the model name, so no manual intervention is required.
 - If you are using an English/numeric or other language model, you need to replace it with the corresponding language dictionary. The PaddleOCR repository provides [some dictionary files](../../../ppocr/utils).

```shell
# Parameters of the executable in run.sh script:
adb shell "cd ${ppocr_demo_path} \
           && chmod +x ./ppocr_demo \
           && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
           && ./ppocr_demo \
                \"./models/${MODEL_NAME}_det.nb\" \
                \"./models/${MODEL_NAME}_rec.nb\" \
                ./models/${CLS_MODEL_FILE} \
                ./images/test.jpg \
                ./test_img_result.jpg \
                ./labels/${LABEL_FILE} \
                ./config.txt"

First parameter: ppocr_demo executable
Second parameter: ./models/${MODEL_NAME}_det.nb  Detection model .nb file
Third parameter: ./models/${MODEL_NAME}_rec.nb  Recognition model .nb file
Fourth parameter: ./models/${CLS_MODEL_FILE} Text line orientation classification model .nb file (automatically selected based on model name by default)
Fifth parameter: ./images/test.jpg  Test image
Sixth parameter: ./test_img_result.jpg  Result save file
Seventh parameter: ./labels/${LABEL_FILE}  Label file (automatically selected based on model name by default)
Eighth parameter: ./config.txt  Configuration file containing hyperparameters for the detection and classification models
```

```shell
# List of Specific Parameters in config.txt:
max_side_len  960         # When the width or height of the input image is greater than 960, the image is scaled proportionally so that the longest side of the image is 960.
det_db_thresh  0.3        # Used to filter the binarized images predicted by DB; setting it to 0.3 has no significant impact on the results.
det_db_box_thresh  0.5    # Threshold for filtering boxes in the DB post-processing; if there are missing boxes in detection, you may reduce this value.
det_db_unclip_ratio  1.6  # Represents the compactness of the text box; the smaller the value, the closer the text box is to the text.
use_direction_classify  0  # Whether to use a direction classifier: 0 means not using it, 1 means using it.
```

## Engineering Details

The OCR text recognition demo accomplishes the OCR text recognition function using three models collaboratively. First, the input image undergoes detection processing via the `${MODEL_NAME}_det.nb` model, followed by text direction classification using the `ch_ppocr_mobile_v2.0_cls_slim_opt.nb` model, and finally, text recognition with the `${MODEL_NAME}_rec.nb` model.

1. `pipeline.cc`: Full-process prediction code for the OCR text recognition demo
  This file handles the entire process control for serial inference of the three models, including scheduling for the entire processing flow.

    - The `Pipeline::Pipeline(...)` method initializes the three model class constructors, accomplishes model loading, thread count, core binding, and predictor creation.
    - The `Pipeline::Process(...)` method manages the entire process control for serial inference of the three models.
  
2. `cls_process.cc`: Prediction file for the direction classifier
  This file handles the preprocessing, prediction, and postprocessing for the direction classifier.

    - The `ClsPredictor::ClsPredictor()` method initializes model loading, thread count, core binding, and predictor creation.
    - The `ClsPredictor::Preprocess()` method handles model preprocessing.
    - The `ClsPredictor::Postprocess()` method handles model postprocessing.

3. `rec_process.cc`: Prediction file for the CRNN recognition model
  This file handles the preprocessing, prediction, and postprocessing for the CRNN recognition model.

    - The `RecPredictor::RecPredictor()` method initializes model loading, thread count, core binding, and predictor creation.
    - The `RecPredictor::Preprocess()` method handles model preprocessing.
    - The `RecPredictor::Postprocess()` method handles model postprocessing.

4. `det_process.cc`: Prediction file for the DB detection model
  This file handles the preprocessing, prediction, and postprocessing for the DB detection model.

    - The `DetPredictor::DetPredictor()` method initializes model loading, thread count, core binding, and predictor creation.
    - The `DetPredictor::Preprocess()` method handles model preprocessing.
    - The `DetPredictor::Postprocess()` method handles model postprocessing.

5. `db_post_process`: Postprocessing functions for the DB detection model, including calls to the clipper library
  This file implements third-party library calls and other postprocessing methods for the DB detection model.

    - The `std::vector<std::vector<std::vector<int>>> BoxesFromBitmap(...)` method retrieves detection boxes from a Bitmap.
    - The `std::vector<std::vector<std::vector<int>>> FilterTagDetRes(...)` method retrieves target box positions based on recognition results.

## Advanced Usage

If the quick start section does not meet your needs, refer to this section for custom modifications to the demo.

This section mainly includes four parts:

- Updating the prediction library;
- Converting `.nb` models;
- Updating models, label files, and prediction images;
- Updating input/output preprocessing.

### Updating the Prediction Library

The prediction library used in this guide is the latest version (214rc), and manual updates are not recommended.

If you need to use a different version, follow these steps to update the prediction library:

* Paddle Lite project: https://github.com/PaddlePaddle/Paddle-Lite
  * Refer to the [Paddle Lite Source Code Compilation Documentation](https://www.paddlepaddle.org.cn/lite/develop/source_compile/compile_env.html) to compile the Android prediction library.
  * The final compilation output is located in `build.lite.xxx.xxx.xxx` under `inference_lite_lib.xxx.xxx`.
    * Replace the C++ library:
        * Header files:
          Replace the `PaddleX-Lite-Deploy/libs/android/cxx/include` folder in the demo with the generated `build.lite.android.xxx.gcc/inference_lite_lib.android.xxx/cxx/include` folder.
        * armeabi-v7a:
          Replace the `PaddleX-Lite-Deploy/libs/android/cxx/libs/armeabi-v7a/libpaddle_lite_api_shared.so` library in the demo with the generated `build.lite.android.armv7.gcc/inference_lite_lib.android.armv7/cxx/libs/libpaddle_lite_api_shared.so` library.
        * arm64-v8a:
          Replace the `PaddleX-Lite-Deploy/libs/android/cxx/libs/arm64-v8a/libpaddle_lite_api_shared.so` library in the demo with the generated `build.lite.android.armv8.gcc/inference_lite_lib.android.armv8/cxx/libs/libpaddle_lite_api_shared.so` library.

### Converting .nb Models

If you want to use your own trained models, follow the process below to obtain `.nb` models.

#### Terminal Command Method (Supports Mac/Ubuntu)

1. Navigate to the [release interface](https://github.com/PaddlePaddle/Paddle-Lite/releases) of the Paddle-Lite GitHub repository and download the corresponding conversion tool, opt, for the desired version (the latest version is recommended).

2. After downloading the opt tool, execute the following command (using the 2.14rc version of the linux_x86 opt tool to convert the PP-OCRv5_mobile_det model as an example):

    ```bash
    ./opt_linux_x86 \
      --model_file=PP-OCRv5_mobile_det/inference.pdmodel \
      --param_file=PP-OCRv5_mobile_det/inference.pdiparams \
      --optimize_out=PP-OCRv5_mobile_det \
      --valid_targets=arm
    ```

For detailed instructions on converting `.nb` models using the terminal command method, refer to the [Using the Executable opt](https://www.paddlepaddle.org.cn/lite/v2.12/user_guides/opt/opt_bin.html) section in the Paddle-Lite repository.

#### Python Script Method (Supports Windows/Mac/Ubuntu)

1. Install the latest version of the paddlelite wheel package.

    ```bash
    pip install --pre paddlelite
    ```

2. Use the Python script to convert the model. Below is an example code snippet for converting the PP-OCRv5_mobile_det model:

    ```python
    from paddlelite.lite import Opt

    # 1. Create an Opt instance
    opt = Opt()
    # 2. Specify the input model paths 
    opt.set_model_file("./PP-OCRv5_mobile_det/inference.pdmodel")
    opt.set_param_file("./PP-OCRv5_mobile_det/inference.pdiparams")
    # 3. Specify the target platform for optimization
    opt.set_valid_places("arm")
    # 4. Specify the output path for the optimized model
    opt.set_optimize_out("./PP-OCRv5_mobile_det")
    # 5. Execute model optimization
    opt.run()
    ```

For detailed instructions on converting `.nb` models using the Python script method, refer to the [Python Script opt Usage](https://www.paddlepaddle.org.cn/lite/v2.12/api_reference/python_api/opt.html) section in the Paddle-Lite repository.

**Notes**

- For detailed information about the model optimization tool `opt`, refer to Paddle-Lite's [Model Optimization Tool opt](https://www.paddlepaddle.org.cn/lite/v2.12/user_guides/model_optimize_tool.html).
- Currently, only static graph models in `.pdmodel` format can be converted to `.nb` format. In PaddlePaddle version 3.0 and above, the default exported model format is `.json`. If you want to export the model in `.pdmodel` format, simply add `-o Global.export_with_pir=False` during export.

### Updating Models, Label Files, and Prediction Images

#### Updating Models

This guide has only validated the `PP-OCRv3_mobile`, `PP-OCRv4_mobile`, and `PP-OCRv5_mobile` models. Other models may not be compatible.

If you fine-tune the `PP-OCRv5_mobile` model and generate a new model named `PP-OCRv5_mobile_ft`, follow these steps to replace the original model with your fine-tuned model:

1. Place the `.nb` models of `PP-OCRv5_mobile_ft` into the directory `PaddleX-Lite-Deploy/ocr/assets/models/`. The resulting file structure should be:

    ```text
    .
    ├── ocr 
    │    ├── ...
    │    └── assets 
    │        ├── models
    │        │   ├── ...
    │        │   ├── PP-OCRv5_mobile_ft_det.nb 
    │        │   └── PP-OCRv5_mobile_ft_rec.nb 
    │        └── ...
    └── ...
    ```

2. Add the model name to the `MODEL_LIST` in the `run.sh` script.

    ```shell
    MODEL_LIST="PP-OCRv3_mobile PP-OCRv4_mobile PP-OCRv5_mobile PP-OCRv5_mobile_ft" # Models are separated by spaces
    ```

3. Specify the model directory name when running the `run.sh` script.

    ```shell
    sh run.sh PP-OCRv5_mobile_ft
    ```

**Notes**:

- If the input Tensor, Shape, or Dtype of the model is updated:

    - For the text direction classifier model, update the `ClsPredictor::Preprocess` function in `ppocr_demo/src/cls_process.cc`.
    - For the detection model, update the `DetPredictor::Preprocess` function in `ppocr_demo/src/det_process.cc`.
    - For the recognition model, update the `RecPredictor::Preprocess` function in `ppocr_demo/src/rec_process.cc`.

- If the output Tensor or Dtype of the model is updated:

    - For the text direction classifier model, update the `ClsPredictor::Postprocess` function in `ppocr_demo/src/cls_process.cc`.
    - For the detection model, update the `DetPredictor::Postprocess` function in `ppocr_demo/src/det_process.cc`.
    - For the recognition model, update the `RecPredictor::Postprocess` function in `ppocr_demo/src/rec_process.cc`.

#### Updating Label Files

To update the label file, place the new label file in the directory `PaddleX-Lite-Deploy/ocr/assets/labels/` and update the execution command in `PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo/run.sh` following the model update method.

For example, to update to `new_labels.txt`:

  ```shell
  # File: `PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo/run.sh`
  # Original command
  adb shell "cd ${ppocr_demo_path} \
            && chmod +x ./ppocr_demo \
            && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
            && ./ppocr_demo \
                  \"./models/${MODEL_NAME}_det.nb\" \
                  \"./models/${MODEL_NAME}_rec.nb\" \
                  ./models/${CLS_MODEL_FILE} \
                  ./images/test.jpg \
                  ./test_img_result.jpg \
                  ./labels/${LABEL_FILE} \
                  ./config.txt"
  # Updated command
  adb shell "cd ${ppocr_demo_path} \
            && chmod +x ./ppocr_demo \
            && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
            && ./ppocr_demo \
                  \"./models/${MODEL_NAME}_det.nb\" \
                  \"./models/${MODEL_NAME}_rec.nb\" \
                  ./models/${CLS_MODEL_FILE} \
                  ./images/test.jpg \
                  ./test_img_result.jpg \
                  ./labels/new_labels.txt \
                  ./config.txt"
  ```
  
#### Updating Prediction Images

If you need to update the prediction images, place the updated images in the `PaddleX-Lite-Deploy/ocr/assets/images/` directory and update the execution command in the `PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo/run.sh` file.

Here is an example of updating to `new_pics.jpg`:

  ```shell
  # File: `PaddleX-Lite-Deploy/ocr/android/shell/ppocr_demo/run.sh`
  ## Original command
  adb shell "cd ${ppocr_demo_path} \
            && chmod +x ./ppocr_demo \
            && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
            && ./ppocr_demo \
                  \"./models/${MODEL_NAME}_det.nb\" \
                  \"./models/${MODEL_NAME}_rec.nb\" \
                  ./models/${CLS_MODEL_FILE} \
                  ./images/test.jpg \
                  ./test_img_result.jpg \
                  ./labels/${LABEL_FILE} \
                  ./config.txt"
  # Updated command
  adb shell "cd ${ppocr_demo_path} \
            && chmod +x ./ppocr_demo \
            && export LD_LIBRARY_PATH=${ppocr_demo_path}:${LD_LIBRARY_PATH} \
            && ./ppocr_demo \
                  \"./models/${MODEL_NAME}_det.nb\" \
                  \"./models/${MODEL_NAME}_rec.nb\" \
                  ./models/${CLS_MODEL_FILE} \
                  ./images/new_pics.jpg \
                  ./test_img_result.jpg \
                  ./labels/${LABEL_FILE} \
                  ./config.txt"
  ```

### Updating Input/Output Preprocessing

- Updating Input Preprocessing
    - For the text direction classifier model, update the `ClsPredictor::Preprocess` function in `ppocr_demo/src/cls_process.cc`.
    - For the detection model, update the `DetPredictor::Preprocess` function in `ppocr_demo/src/det_process.cc`.
    - For the recognition model, update the `RecPredictor::Preprocess` function in `ppocr_demo/src/rec_process.cc`.

- Updating Output Preprocessing
    - For the text direction classifier model, update the `ClsPredictor::Postprocess` function in `ppocr_demo/src/cls_process.cc`.
    - For the detection model, update the `DetPredictor::Postprocess` function in `ppocr_demo/src/det_process.cc`.
    - For the recognition model, update the `RecPredictor::Postprocess` function in `ppocr_demo/src/rec_process.cc`.
