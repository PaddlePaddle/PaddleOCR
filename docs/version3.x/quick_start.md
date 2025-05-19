---
comments: true
hide:
  - navigation
---

### 安装

#### 1. 安装PaddlePaddle

CPU端安装：

```bash
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

GPU端安装，由于GPU端需要根据具体CUDA版本来对应安装使用，以下仅以Linux平台，pip安装英伟达GPU， CUDA11.8为例，其他平台，请参考[飞桨官网安装文档](https://www.paddlepaddle.org.cn/install/quick)中的说明进行操作。

```bash
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

#### 2. 安装`paddleocr`

```bash
pip install paddleocr
```

### 命令行使用

OCR产线：

```bash
paddleocr ocr -i ./general_ocr_002.png
```

文本检测模块：

```bash
paddleocr text_detection -i ./general_ocr_001.png
```

文本识别模块：

```bash
paddleocr text_recognition -i ./general_ocr_rec_001.png
```

### Python脚本使用

OCR产线：

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR() # 文本图像预处理+文本检测+方向分类+文本识别
# ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False) # 文本检测+方向分类+文本识别
# ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False) # 文本检测+文本识别
result = ocr.predict("./general_ocr_002.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

文本检测模块：

```python
from paddleocr import TextDetection

model = TextDetection()
output = model.predict("general_ocr_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

文本识别模块：

```python
from paddleocr import TextRecognition

model = TextRecognition()
output = model.predict(input="general_ocr_rec_001.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```
