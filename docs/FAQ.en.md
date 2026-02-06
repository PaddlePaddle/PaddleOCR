---
comments: true
hide:
  - navigation
  - toc
---

> 🎉 **Welcome to PaddleOCR FAQ!**  
> This document compiles common issues and solutions from GitHub Issues and Discussions, providing reference for OCR developers.

## 1. Installation and Environment Setup

### 1.1 Basic Installation Issues

#### Q: PaddleOCR installation failed with dependency conflicts

**A**: This is a common issue that can be resolved by:
(1) Create a new virtual environment: `conda create -n paddleocr python=3.8`, then activate and install
(2) Install with specific versions: `pip install paddleocr==3.2.0 --no-deps`, then install dependencies separately
(3) If using conda, try: `conda install -c conda-forge paddleocr`

#### Q: GPU environment configuration issues, CUDA version mismatch

**A**: First check CUDA version: `nvidia-smi`, then install corresponding PaddlePaddle version:
- CUDA 11.8: `pip install paddlepaddle-gpu==3.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html`
- CUDA 12.0: `pip install paddlepaddle-gpu==3.0.0 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html`
Verify GPU availability: `python -c "import paddle; print(paddle.is_compiled_with_cuda())"`

#### Q: Model download failed or slow download

**A**: Can be resolved by:
(1) Set model download source: `os.environ['PADDLE_PDX_MODEL_SOURCE'] = 'BOS'` (use Baidu Cloud Storage)
(2) Manual model download: directly access model links to download and extract to local directory
(3) Use local models: specify `model_dir` parameter pointing to local model path during initialization

#### Q: How to run on Windows or Mac?

**A**: PaddleOCR has completed adaptation to Windows and MAC systems. Two points should be noted during operation:
    1. In [Quick installation](./installation_en.md), if you do not want to install docker, you can skip the first step and start with the second step.
    2. When downloading the inference model, if wget is not installed, you can directly click the model link or copy the link address to the browser to download, then extract and place it in the corresponding directory.

## 2. Model Usage and Configuration

### 2.1 Model Selection

#### Q: How to choose the right model?

**A**: Choose based on application scenario:
- Server high accuracy: Use `PP-OCRv5_server` series, highest accuracy
- Mobile deployment: Use `PP-OCRv5_mobile` series, small model fast speed
- Real-time processing: Use `PP-OCRv5_mobile` series, fast inference speed
- Batch processing: Use `PP-OCRv5_server` series, high accuracy
- Multi-language recognition: Use `PP-OCRv5_multi_languages`, supports 37 languages

#### Q: Local model path configuration, how to use in isolated network environment?

**A**: Can be configured by:
(1) Use local model path: specify `model_dir` parameter during initialization
(2) Set model download source: `os.environ['PADDLE_PDX_MODEL_SOURCE'] = 'BOS'`
(3) Manual model download: download models from official links and extract locally
(4) Example code:
```python
ocr = PaddleOCR(
    det_model_dir='./models/PP-OCRv5_server_det_infer/',
    rec_model_dir='./models/PP-OCRv5_server_rec_infer/',
    cls_model_dir='./models/PP-OCRv5_cls_infer/',
    use_angle_cls=True,
    lang='ch'
)
```

## 3. Performance Optimization

### 3.1 GPU Optimization

#### Q: Slow GPU inference speed, how to optimize performance?

**A**: Can be optimized by:
(1) Enable high-performance inference: set `enable_hpi=True`, automatically select optimal acceleration strategy
(2) Enable TensorRT acceleration: set `use_tensorrt=True`, requires CUDA 11.8+ and TensorRT 8.6+
(3) Use half precision: set `precision="fp16"`, can significantly improve speed
(4) Adjust batch size: set appropriate `batch_size` based on GPU memory
(5) Use mobile models: use `PP-OCRv5_mobile` series when accuracy requirements are not high

#### Q: GPU memory insufficient (CUDA out of memory) what to do?

**A**: Can be resolved by:
(1) Reduce batch size: set `batch_size` to 1
(2) Reduce image size: set `det_limit_side_len=640`
(3) Enable memory optimization: set `enable_memory_optim=True`
(4) Limit GPU memory usage: set `gpu_mem=200`
(5) Use mobile models: switch to `PP-OCRv5_mobile` series models

## 4. Deployment Issues

### 4.1 Service Deployment

#### Q: How to deploy PaddleOCR as a web service?

**A**: Can be deployed by:
(1) Use Flask deployment: create simple Web API service
(2) Use gunicorn deployment: `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
(3) Use asynchronous processing: combine asyncio and ThreadPoolExecutor
(4) Example code:
```python
from flask import Flask, request, jsonify
from paddleocr import PaddleOCR

app = Flask(__name__)
ocr = PaddleOCR(use_angle_cls=True, lang='ch')

@app.route('/ocr', methods=['POST'])
def ocr_api():
    file = request.files['image']
    result = ocr.ocr(file, cls=True)
    return jsonify({'result': result})
```

#### Q: C++ deployment common issues and solutions?

**A**: Common issues and solutions:
(1) Cannot find dynamic library: set `export LD_LIBRARY_PATH=/path/to/paddle/lib:$LD_LIBRARY_PATH`
(2) OpenCV version mismatch: ensure OpenCV version matches compilation time
(3) Model format issues: ensure correct inference model format
(4) Compilation issues: ensure CMake configuration is correct, use `cmake .. -DCMAKE_BUILD_TYPE=Release`

#### Q: Service performance optimization suggestions?

**A**: Can be optimized by:
(1) Enable high-performance inference: set `enable_hpi=True`
(2) Use batch processing: set appropriate `batch_size`
(3) Enable TensorRT: set `use_tensorrt=True`
(4) Use asynchronous processing: avoid blocking requests
(5) Load balancing: use multiple service instances

## 5. Legacy Issues (for reference)

1. **Prediction error: got an unexpected keyword argument 'gradient_clip'**
The installed version of paddle is incorrect. Currently, this project only supports Paddle 1.7, which will be adapted to 1.8 in the near future.

2. **Error when converting attention recognition model: KeyError: 'predict'**
Solved. Please update to the latest version of the code.

3. **About inference speed**
When there are many words in the picture, the prediction time will increase. You can use `--rec_batch_num` to set a smaller prediction batch num. The default value is 30, which can be changed to 10 or other values.

4. **Service deployment and mobile deployment**
It is expected that the service deployment based on Serving and the mobile deployment based on Paddle Lite will be released successively in mid-to-late June. Stay tuned for more updates.

5. **Release time of self-developed algorithm**
Baidu Self-developed algorithms such as SAST, SRN and end2end PSL will be released in June or July. Please be patient.

6. **How to run on Windows or Mac?**
PaddleOCR has completed the adaptation to Windows and MAC systems. Two points should be noted during operation:
    1. In [Quick installation](./version3.x/installation.en.md), if you do not want to install docker, you can skip the first step and start with the second step.
    2. When downloading the inference model, if wget is not installed, you can directly click the model link or copy the link address to the browser to download, then extract and place it in the corresponding directory.

7. **The difference between ultra-lightweight model and General OCR model**
At present, PaddleOCR has opensourced two Chinese models, namely 8.6M ultra-lightweight Chinese model and general Chinese OCR model. The comparison information between the two is as follows:
    - Similarities: Both use the same **algorithm** and **training data**；
    - Differences: The difference lies in **backbone network** and **channel parameters**, the ultra-lightweight model uses MobileNetV3 as the backbone network, the general model uses Resnet50_vd as the detection model backbone, and Resnet34_vd as the recognition model backbone. You can compare the two model training configuration files to see the differences in parameters.

|Model|Backbone|Detection configuration file|Recognition configuration file|
|-|-|-|-|
|8.6M ultra-lightweight Chinese OCR model|MobileNetV3+MobileNetV3|det_mv3_db.yml|rec_chinese_lite_train.yml|
|General Chinese OCR model|Resnet50_vd+Resnet34_vd|det_r50_vd_db.yml|rec_chinese_common_train.yml|

8. **Is there a plan to opensource a model that only recognizes numbers or only English + numbers?**
It is not planned to opensource numbers only, numbers + English only, or other vertical text models. PaddleOCR has opensourced a variety of detection and recognition algorithms for customized training. The two Chinese models are also based on the training output of the open-source algorithm library. You can prepare the data according to the tutorial, choose the appropriate configuration file, train yourselves, and we believe that you can get good result. If you have any questions during the training, you are welcome to open issues or ask in the communication group. We will answer them in time.

9. **What is the training data used by the open-source model? Can it be opensourced?**
At present, the open source model, dataset and magnitude are as follows:
    - Detection:
    English dataset: ICDAR2015
    Chinese dataset: LSVT street view dataset with 3w pictures
    - Recognition:
    English dataset: MJSynth and SynthText synthetic dataset, the amount of data is tens of millions.
    Chinese dataset: LSVT street view dataset with cropped text area, a total of 30w images. In addition, the synthesized data based on LSVT corpus is 500w.

    Among them, the public datasets are opensourced, users can search and download by themselves, or refer to [Chinese data set](./datasets/datasets.en.md), synthetic data is not opensourced, users can use open-source synthesis tools to synthesize data themselves. Current available synthesis tools include [text_renderer](https://github.com/Sanster/text_renderer), [SynthText](https://github.com/ankush-me/SynthText), [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator), etc.

10. **Error in using the model with TPS module for prediction**
Error message: Input(X) dims[3] and Input(Grid) dims[2] should be equal, but received X dimension[3]\(108) != Grid dimension[2]\(100)
Solution: TPS does not support variable shape. Please set --rec_image_shape='3,32,100' and --rec_char_type='en'

11. **Custom dictionary used during training, the recognition results show that words do not appear in the dictionary**
The used custom dictionary path is not set when making prediction. The solution is setting parameter `rec_char_dict_path` to the corresponding dictionary file.

12. **Results of cpp_infer and python_inference are very different**
Versions of exported inference model and inference library should be same. For example, on Windows platform, version of the inference library that PaddlePaddle provides is 1.8, but version of the inference model that PaddleOCR provides is 1.7, you should export model yourself(`tools/export_model.py`) on PaddlePaddle 1.8 and then use the exported model for inference.

13. **How to identify artistic fonts in signs or advertising images**
Recognizing artistic fonts in signs or advertising images is a very challenging task because the variation in individual characters is much greater compared to standard fonts. If the artistic font to be identified is within a dictionary list, each word in the dictionary can be treated as a template for recognition using a general image retrieval system. You can try using PaddleClas image recognition system.

14. **How to change the font when visualizing the OCR prediction results?**

**A**: You can specify the local font file path by using the environment variable `PADDLE_PDX_LOCAL_FONT_FILE_PATH`, such as `PADDLE_PDX_LOCAL_FONT_FILE_PATH=/root/fonts/simfang.ttf`.
