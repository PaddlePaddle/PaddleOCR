English | [简体中文](readme_ch.md)

# Server-side C++ Inference

- [1. Prepare the Environment](#1)
    - [1.1 Environment](#11)
    - [1.2 Compile OpenCV](#12)
    - [1.3 Compile or Download or the Paddle Inference Library](#13)
- [2. Compile and Run the Demo](#2)
    - [2.1 Export the inference model](#21)
    - [2.2 Compile PaddleOCR C++ inference demo](#22)
    - [2.3 Run the demo](#23)
- [3. FAQ](#3)


This chapter introduces the C++ deployment steps of the PaddleOCR model. C++ is better than Python in terms of performance. Therefore, in CPU and GPU deployment scenarios, C++ deployment is mostly used.
This section will introduce how to configure the C++ environment and deploy PaddleOCR in Linux (CPU\GPU) environment. For Windows deployment please refer to [Windows](./docs/windows_vs2019_build.md) compilation guidelines.


<a name="1"></a>
## 1. Prepare the Environment

<a name="11"></a>
### 1.1 Environment

- Linux, docker is recommended.
- Windows.


<a name="12"></a>
### 1.2 Compile OpenCV

* First of all, you need to download the source code compiled package in the Linux environment from the OpenCV official website. Taking OpenCV 3.4.7 as an example, the download command is as follows.

```bash
cd deploy/cpp_infer
wget https://paddleocr.bj.bcebos.com/libs/opencv/opencv-3.4.7.tar.gz
tar -xf opencv-3.4.7.tar.gz
```

Finally, you will see the folder of `opencv-3.4.7/` in the current directory.

* Compile OpenCV, the OpenCV source path (`root_path`) and installation path (`install_path`) should be set by yourself. Enter the OpenCV source code path and compile it in the following way.


```shell
root_path=your_opencv_root_path
install_path=${root_path}/opencv3

rm -rf build
mkdir build
cd build

cmake .. \
    -DCMAKE_INSTALL_PREFIX=${install_path} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_IPP=OFF \
    -DBUILD_IPP_IW=OFF \
    -DWITH_LAPACK=OFF \
    -DWITH_EIGEN=OFF \
    -DCMAKE_INSTALL_LIBDIR=lib64 \
    -DWITH_ZLIB=ON \
    -DBUILD_ZLIB=ON \
    -DWITH_JPEG=ON \
    -DBUILD_JPEG=ON \
    -DWITH_PNG=ON \
    -DBUILD_PNG=ON \
    -DWITH_TIFF=ON \
    -DBUILD_TIFF=ON

make -j
make install
```

In the above commands, `root_path` is the downloaded OpenCV source code path, and `install_path` is the installation path of OpenCV. After `make install` is completed, the OpenCV header file and library file will be generated in this folder for later OCR source code compilation.



The final file structure under the OpenCV installation path is as follows.

```
opencv3/
|-- bin
|-- include
|-- lib
|-- lib64
|-- share
```

<a name="13"></a>
### 1.3 Compile or Download or the Paddle Inference Library

* There are 2 ways to obtain the Paddle inference library, described in detail below.

#### 1.3.1 Direct download and installation

[Paddle inference library official website](https://paddleinference.paddlepaddle.org.cn/user_guides/download_lib.html#linux). You can review and select the appropriate version of the inference library on the official website.


* After downloading, use the following command to extract files.

```
tar -xf paddle_inference.tgz
```

Finally you will see the folder of `paddle_inference/` in the current path.

#### 1.3.2 Compile the inference source code
* If you want to get the latest Paddle inference library features, you can download the latest code from Paddle GitHub repository and compile the inference library from the source code. It is recommended to download the inference library with paddle version greater than or equal to 2.0.1.
* You can refer to [Paddle inference library] (https://www.paddlepaddle.org.cn/documentation/docs/en/advanced_guide/inference_deployment/inference/build_and_install_lib_en.html) to get the Paddle source code from GitHub, and then compile To generate the latest inference library. The method of using git to access the code is as follows.


```shell
git clone https://github.com/PaddlePaddle/Paddle.git
git checkout develop
```

* Enter the Paddle directory and run the following commands to compile the paddle inference library.

```shell
rm -rf build
mkdir build
cd build

cmake  .. \
    -DWITH_CONTRIB=OFF \
    -DWITH_MKL=ON \
    -DWITH_MKLDNN=ON  \
    -DWITH_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_INFERENCE_API_TEST=OFF \
    -DON_INFER=ON \
    -DWITH_PYTHON=ON
make -j
make inference_lib_dist
```

For more compilation parameter options, please refer to the [document](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.0/guides/05_inference_deployment/inference/build_and_install_lib_cn.html#congyuanmabianyi).


* After the compilation process, you can see the following files in the folder of `build/paddle_inference_install_dir/`.

```
build/paddle_inference_install_dir/
|-- CMakeCache.txt
|-- paddle
|-- third_party
|-- version.txt
```

`paddle` is the Paddle library required for C++ prediction later, and `version.txt` contains the version information of the current inference library.


<a name="2"></a>
## 2. Compile and Run the Demo

<a name="21"></a>
### 2.1 Export the inference model

* You can refer to [Model inference](../../doc/doc_ch/inference.md) and export the inference model. After the model is exported, assuming it is placed in the `inference` directory, the directory structure is as follows.

```
inference/
|-- det_db
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- rec_rcnn
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- cls
|   |--inference.pdiparams
|   |--inference.pdmodel
|-- table
|   |--inference.pdiparams
|   |--inference.pdmodel
```


<a name="22"></a>
### 2.2 Compile PaddleOCR C++ inference demo

* The compilation commands are as follows. The addresses of Paddle C++ inference library, opencv and other Dependencies need to be replaced with the actual addresses on your own machines.

```shell
sh tools/build.sh
```

Specifically, you should modify the paths in `tools/build.sh`. The related content is as follows.

```shell
OPENCV_DIR=your_opencv_dir
LIB_DIR=your_paddle_inference_dir
CUDA_LIB_DIR=your_cuda_lib_dir
CUDNN_LIB_DIR=your_cudnn_lib_dir
```

`OPENCV_DIR` is the OpenCV installation path; `LIB_DIR` is the download (`paddle_inference` folder)
or the generated Paddle inference library path (`build/paddle_inference_install_dir` folder);
`CUDA_LIB_DIR` is the CUDA library file path, in docker; it is `/usr/local/cuda/lib64`; `CUDNN_LIB_DIR` is the cuDNN library file path, in docker it is `/usr/lib/x86_64-linux-gnu/`.


* After the compilation is completed, an executable file named `ppocr` will be generated in the `build` folder.


<a name="23"></a>
### 2.3 Run the demo

Execute the built executable file:
```shell
./build/ppocr [--param1] [--param2] [...]
```

**Note**:ppocr uses the `PP-OCRv3` model by default, and the input shape used by the recognition model is `3, 48, 320`, if you want to use the old version model, you should add the parameter `--rec_img_h=32`.

Specifically,

##### 1. det+cls+rec：
```shell
./build/ppocr --det_model_dir=inference/det_db \
    --rec_model_dir=inference/rec_rcnn \
    --cls_model_dir=inference/cls \
    --image_dir=../../doc/imgs/12.jpg \
    --use_angle_cls=true \
    --det=true \
    --rec=true \
    --cls=true \
```

##### 2. det+rec：
```shell
./build/ppocr --det_model_dir=inference/det_db \
    --rec_model_dir=inference/rec_rcnn \
    --image_dir=../../doc/imgs/12.jpg \
    --use_angle_cls=false \
    --det=true \
    --rec=true \
    --cls=false \
```

##### 3. det
```shell
./build/ppocr --det_model_dir=inference/det_db \
    --image_dir=../../doc/imgs/12.jpg \
    --det=true \
    --rec=false
```

##### 4. cls+rec：
```shell
./build/ppocr --rec_model_dir=inference/rec_rcnn \
    --cls_model_dir=inference/cls \
    --image_dir=../../doc/imgs_words/ch/word_1.jpg \
    --use_angle_cls=true \
    --det=false \
    --rec=true \
    --cls=true \
```

##### 5. rec
```shell
./build/ppocr --rec_model_dir=inference/rec_rcnn \
    --image_dir=../../doc/imgs_words/ch/word_1.jpg \
    --use_angle_cls=false \
    --det=false \
    --rec=true \
    --cls=false \
```

##### 6. cls
```shell
./build/ppocr --cls_model_dir=inference/cls \
    --cls_model_dir=inference/cls \
    --image_dir=../../doc/imgs_words/ch/word_1.jpg \
    --use_angle_cls=true \
    --det=false \
    --rec=false \
    --cls=true \
```


##### 7. table
```shell
./build/ppocr --det_model_dir=inference/det_db \
    --rec_model_dir=inference/rec_rcnn \
    --table_model_dir=inference/table \
    --image_dir=../../ppstructure/docs/table/table.jpg \
    --type=structure \
    --table=true
```

More parameters are as follows,

- Common parameters

|parameter|data type|default|meaning|
| --- | --- | --- | --- |
|use_gpu|bool|false|Whether to use GPU|
|gpu_id|int|0|GPU id when use_gpu is true|
|gpu_mem|int|4000|GPU memory requested|
|cpu_math_library_num_threads|int|10|Number of threads when using CPU inference. When machine cores is enough, the large the value, the faster the inference speed|
|enable_mkldnn|bool|true|Whether to use mkdlnn library|
|output|str|./output|Path where visualization results are saved|


- forward

|parameter|data type|default|meaning|
| :---: | :---: | :---: | :---: |
|det|bool|true|Whether to perform text detection in the forward direction|
|rec|bool|true|Whether to perform text recognition in the forward direction|
|cls|bool|false|Whether to perform text direction classification in the forward direction|


- Detection related parameters

|parameter|data type|default|meaning|
| --- | --- | --- | --- |
|det_model_dir|string|-|Address of detection inference model|
|max_side_len|int|960|Limit the maximum image height and width to 960|
|det_db_thresh|float|0.3|Used to filter the binarized image of DB prediction, setting 0.-0.3 has no obvious effect on the result|
|det_db_box_thresh|float|0.5|DB post-processing filter box threshold, if there is a missing box detected, it can be reduced as appropriate|
|det_db_unclip_ratio|float|1.6|Indicates the compactness of the text box, the smaller the value, the closer the text box to the text|
|det_db_score_mode|string|slow| slow: use polygon box to calculate bbox score, fast: use rectangle box to calculate. Use rectangular box to calculate faster, and polygonal box more accurate for curved text area.|
|visualize|bool|true|Whether to visualize the results，when it is set as true, the prediction results will be saved in the folder specified by the `output` field on an image with the same name as the input image.|

- Classifier related parameters

|parameter|data type|default|meaning|
| --- | --- | --- | --- |
|use_angle_cls|bool|false|Whether to use the direction classifier|
|cls_model_dir|string|-|Address of direction classifier inference model|
|cls_thresh|float|0.9|Score threshold of the  direction classifier|
|cls_batch_num|int|1|batch size of classifier|

- Recognition related parameters

|parameter|data type|default|meaning|
| --- | --- | --- | --- |
|rec_model_dir|string|-|Address of recognition inference model|
|rec_char_dict_path|string|../../ppocr/utils/ppocr_keys_v1.txt|dictionary file|
|rec_batch_num|int|6|batch size of recognition|
|rec_img_h|int|48|image height of recognition|
|rec_img_w|int|320|image width of recognition|

- Table recognition related parameters

|parameter|data type|default|meaning|
| :---: | :---: | :---: | :---: |
|table_model_dir|string|-|Address of table recognition inference model|
|table_char_dict_path|string|../../ppocr/utils/dict/table_structure_dict.txt|dictionary file|
|table_max_len|int|488|The size of the long side of the input image of the table recognition model, the final input image size of the network is（table_max_len，table_max_len）|
|merge_no_span_structure|bool|true|Whether to merge <td> and </td> to <td></td|


* Multi-language inference is also supported in PaddleOCR, you can refer to [recognition tutorial](../../doc/doc_en/recognition_en.md) for more supported languages and models in PaddleOCR. Specifically, if you want to infer using multi-language models, you just need to modify values of `rec_char_dict_path` and `rec_model_dir`.


The detection results will be shown on the screen, which is as follows.

```bash
predict img: ../../doc/imgs/12.jpg
../../doc/imgs/12.jpg
0       det boxes: [[74,553],[427,542],[428,571],[75,582]] rec text: 打浦路252935号 rec score: 0.947724
1       det boxes: [[23,507],[513,488],[515,529],[24,548]] rec text: 绿洲仕格维花园公寓 rec score: 0.993728
2       det boxes: [[187,456],[399,448],[400,480],[188,488]] rec text: 打浦路15号 rec score: 0.964994
3       det boxes: [[42,413],[483,391],[484,428],[43,450]] rec text: 上海斯格威铂尔大酒店 rec score: 0.980086
The detection visualized image saved in ./output//12.jpg
```

- table

```bash
predict img: ../../ppstructure/docs/table/table.jpg
0       type: table, region: [0,0,371,293], res: <html><body><table><thead><tr><td>Methods</td><td>R</td><td>P</td><td>F</td><td>FPS</td></tr></thead><tbody><tr><td>SegLink [26]</td><td>70.0</td><td>86.0</td><td>77.0</td><td>8.9</td></tr><tr><td>PixelLink [4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td>-</td></tr><tr><td>TextSnake [18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField [37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2 </td></tr><tr><td>MSR[38]</td><td>76.7</td><td>87.4</td><td>81.7</td><td>-</td></tr><tr><td>FTSN [3]</td><td>77.1</td><td>87.6</td><td>82.0</td><td>-</td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><td>-</td></tr><tr><td>CRAFT [2]</td><td>78.2</td><td>88.2</td><td>82.9</td><td>8.6</td></tr><tr><td>MCN [16]</td><td>79</td><td>88</td><td>83</td><td>-</td></tr><tr><td>ATRR[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td>-</td></tr><tr><td>PAN [34]</td><td>83.8</td><td>84.4</td><td>84.1</td><td>30.2</td></tr><tr><td>DB[12]</td><td>79.2</td><td>91.5</td><td>84.9</td><td>32.0</td></tr><tr><td>DRRG [41]</td><td>82.30</td><td>88.05</td><td>85.08</td><td>-</td></tr><tr><td>Ours (SynText)</td><td>80.68</td><td>85.40</td><td>82.97</td><td>12.68</td></tr><tr><td>Ours (MLT-17)</td><td>84.54</td><td>86.62</td><td>85.57</td><td>12.31</td></tr></tbody></table></body></html>
```

<a name="3"></a>
## 3. FAQ

 1.  Encountered the error `unable to access 'https://github.com/LDOUBLEV/AutoLog.git/': gnutls_handshake() failed: The TLS connection was non-properly terminated.`, change the github address in `deploy/cpp_infer/external-cmake/auto-log.cmake` to the https://gitee.com/Double_V/AutoLog address.
