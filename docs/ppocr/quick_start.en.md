---
comments: true
---

# PaddleOCR Quick Start

**Note:** This tutorial mainly introduces the usage of PP-OCR series models, please refer to [PP-Structure Quick Start](../ppstructure/overview.en.md) for the quick use of document analysis related functions. In addition, the All-in-One development tool PaddleX relies on the advanced technology of PaddleOCR to support low-code full-process development capabilities in the OCR field, significantly reducing development time and complexity. It also integrates the 17 models involved in text image intelligent analysis, OCR, layout parsing, table recognition, formula recognition, and seal text recognition into 6 pipelines, which can be invoked with a simple Python API. For more details, please see [Low-Code Full-Process Development](https://paddlepaddle.github.io/PaddleOCR/latest/en/paddlex/quick_start.html).

## 1. Installation

### 1.1 Install PaddlePaddle

> If you do not have a Python environment, please refer to [Environment Preparation](./environment.en.md).

- If you have CUDA 11 installed on your machine, please run the following command to install

  ```bash linenums="1"
  pip install paddlepaddle-gpu
  ```

- If you have no available GPU on your machine, please run the following command to install the CPU version

  ```bash linenums="1"
  python -m pip install paddlepaddle
  ```

For more software version requirements, please refer to the instructions in [Installation Document](https://www.paddlepaddle.org.cn/en/install/quick) for operation.

### 1.2 Install PaddleOCR Whl Package

```bash linenums="1"
pip install "paddleocr>=2.0.1" # Recommend to use version 2.0.1+
```

- **For windows users:** If you getting this error `OSError: [WinError 126] The specified module could not be found` when you install shapely on windows. Please try to download Shapely whl file [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely).

  Reference: [Solve shapely installation on windows](https://stackoverflow.com/questions/44398265/install-shapely-oserror-winerror-126-the-specified-module-could-not-be-found)

## 2. Easy-to-Use

### 2.1 Use by Command Line

PaddleOCR provides a series of test images, click [here](https://paddleocr.bj.bcebos.com/dygraph_v2.1/ppocr_img.zip) to download, and then switch to the corresponding directory in the terminal

```bash linenums="1"
cd /path/to/ppocr_img
```

If you do not use the provided test image, you can replace the following `--image_dir` parameter with the corresponding test image path

#### 2.1.1 Chinese and English Model

- Detection, direction classification and recognition: set the parameter`--use_gpu false` to disable the gpu device

  ```bash linenums="1"
  paddleocr --image_dir ./imgs_en/img_12.jpg --use_angle_cls true --lang en --use_gpu false
  ```

  Output will be a list, each item contains bounding box, text and recognition confidence

  ```bash linenums="1"
  [[[441.0, 174.0], [1166.0, 176.0], [1165.0, 222.0], [441.0, 221.0]], ('ACKNOWLEDGEMENTS', 0.9971134662628174)]
  [[[403.0, 346.0], [1204.0, 348.0], [1204.0, 384.0], [402.0, 383.0]], ('We would like to thank all the designers and', 0.9761400818824768)]
  [[[403.0, 396.0], [1204.0, 398.0], [1204.0, 434.0], [402.0, 433.0]], ('contributors who have been involved in the', 0.9791957139968872)]
  ......
  ```

  pdf file is also supported, you can infer the first few pages by using the `page_num` parameter, the default is 0, which means infer all pages

  ```bash linenums="1"
  paddleocr --image_dir ./xxx.pdf --use_angle_cls true --use_gpu false --page_num 2
  ```

- Only detection: set `--rec` to `false`

  ```bash linenums="1"
  paddleocr --image_dir ./imgs_en/img_12.jpg --rec false
  ```

  Output will be a list, each item only contains bounding box

  ```bash linenums="1"
  [[397.0, 802.0], [1092.0, 802.0], [1092.0, 841.0], [397.0, 841.0]]
  [[397.0, 750.0], [1211.0, 750.0], [1211.0, 789.0], [397.0, 789.0]]
  [[397.0, 702.0], [1209.0, 698.0], [1209.0, 734.0], [397.0, 738.0]]
  ......
  ```

- Only recognition: set `--det` to `false`

  ```bash linenums="1"
  paddleocr --image_dir ./imgs_words_en/word_10.png --det false --lang en
  ```

  Output will be a list, each item contains text and recognition confidence

  ```bash linenums="1"
  ['PAIN', 0.9934559464454651]
  ```

**Version**
paddleocr uses the PP-OCRv4 model by default(`--ocr_version PP-OCRv4`). If you want to use other versions, you can set the parameter `--ocr_version`, the specific version description is as follows:

|  version name |  description |
|    ---    |   ---   |
| PP-OCRv4 | support Chinese and English detection and recognition, direction classifier, support multilingual recognition |
| PP-OCRv3 | support Chinese and English detection and recognition, direction classifier, support multilingual recognition |
| PP-OCRv2 | only supports Chinese and English detection and recognition, direction classifier, multilingual model is not updated |
| PP-OCR   | support Chinese and English detection and recognition, direction classifier, support multilingual recognition |

If you want to add your own trained model, you can add model links and keys in [paddleocr](https://github.com/PaddlePaddle/PaddleOCR/blob/c65a66c5fd37dee64916a8b2a2c84ea273d98cac/paddleocr.py) and recompile.

More whl package usage can be found in [whl package](./blog/whl.en.md)

#### 2.1.2 Multi-language Model

PaddleOCR currently supports 80 languages, which can be switched by modifying the `--lang` parameter.

``` bash
paddleocr --image_dir ./doc/imgs_en/254.jpg --lang=en
```

![](./images/254.jpg)

![](./images/multi_lang/img_02.jpg)

The result is a list, each item contains a text box, text and recognition confidence

```text linenums="1"
[[[67.0, 51.0], [327.0, 46.0], [327.0, 74.0], [68.0, 80.0]], ('PHOCAPITAL', 0.9944712519645691)]
[[[72.0, 92.0], [453.0, 84.0], [454.0, 114.0], [73.0, 122.0]], ('107 State Street', 0.9744491577148438)]
[[[69.0, 135.0], [501.0, 125.0], [501.0, 156.0], [70.0, 165.0]], ('Montpelier Vermont', 0.9357033967971802)]
......
```

Commonly used multilingual abbreviations include

| Language            | Abbreviation |      | Language | Abbreviation |      | Language | Abbreviation |
| ------------------- | ------------ | ---- | -------- | ------------ | ---- | -------- | ------------ |
| Chinese & English   | ch           |      | French   | fr           |      | Japanese | japan        |
| English             | en           |      | German   | german       |      | Korean   | korean       |
| Chinese Traditional | chinese_cht  |      | Italian  | it           |      | Russian  | ru           |

A list of all languages and their corresponding abbreviations can be found in [Multi-Language Model Tutorial](./blog/multi_languages.en.md)

### 2.2 Use by Code

#### 2.2.1 Chinese & English Model and Multilingual Model

- detection, angle classification and recognition:

```python linenums="1"
from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = './imgs_en/img_12.jpg'
result = ocr.ocr(img_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

Output will be a list, each item contains bounding box, text and recognition confidence

```bash linenums="1"
[[[441.0, 174.0], [1166.0, 176.0], [1165.0, 222.0], [441.0, 221.0]], ('ACKNOWLEDGEMENTS', 0.9971134662628174)]
  [[[403.0, 346.0], [1204.0, 348.0], [1204.0, 384.0], [402.0, 383.0]], ('We would like to thank all the designers and', 0.9761400818824768)]
  [[[403.0, 396.0], [1204.0, 398.0], [1204.0, 434.0], [402.0, 433.0]], ('contributors who have been involved in the', 0.9791957139968872)]
  ......
```

Visualization of results

![](./images/11_det_rec.jpg)

If the input is a PDF file, you can refer to the following code for visualization

```python linenums="1"
from paddleocr import PaddleOCR, draw_ocr

# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
PAGE_NUM = 10 # Set the recognition page number
pdf_path = 'default.pdf'
ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=PAGE_NUM)  # need to run only once to download and load model into memory
# ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=PAGE_NUM,use_gpu=0) # To Use GPU,uncomment this line and comment the above one.
result = ocr.ocr(pdf_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    if res == None: # Skip when empty result detected to avoid TypeError:NoneType
        print(f"[DEBUG] Empty page {idx+1} detected, skip it.")
        continue
    for line in res:
        print(line)

# draw the result
import fitz
from PIL import Image
import cv2
import numpy as np
imgs = []
with fitz.open(pdf_path) as pdf:
    for pg in range(0, PAGE_NUM):
        page = pdf[pg]
        mat = fitz.Matrix(2, 2)
        pm = page.get_pixmap(matrix=mat, alpha=False)
        # if width or height > 2000 pixels, don't enlarge the image
        if pm.width > 2000 or pm.height > 2000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        imgs.append(img)
for idx in range(len(result)):
    res = result[idx]
    if res == None:
        continue
    image = imgs[idx]
    boxes = [line[0] for line in res]
    txts = [line[1][0] for line in res]
    scores = [line[1][1] for line in res]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result_page_{}.jpg'.format(idx))
```

- Detection and Recognition Using Sliding Windows

To perform OCR using sliding windows, the following code snippet can be employed:

```python linenums="1"
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# Initialize OCR engine
ocr = PaddleOCR(use_angle_cls=True, lang="en")

img_path = "./very_large_image.jpg"
slice = {'horizontal_stride': 300, 'vertical_stride': 500, 'merge_x_thres': 50, 'merge_y_thres': 35}
results = ocr.ocr(img_path, cls=True, slice=slice)

# Load image
image = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("./doc/fonts/simfang.ttf", size=20)  # Adjust size as needed

# Process and draw results
for res in results:
    for line in res:
        box = [tuple(point) for point in line[0]]
        # Finding the bounding box
        box = [(min(point[0] for point in box), min(point[1] for point in box)),
               (max(point[0] for point in box), max(point[1] for point in box))]
        txt = line[1][0]
        draw.rectangle(box, outline="red", width=2)  # Draw rectangle
        draw.text((box[0][0], box[0][1] - 25), txt, fill="blue", font=font)  # Draw text above the box

# Save result
image.save("result.jpg")

```

This example initializes the PaddleOCR instance with angle classification enabled and sets the language to English. The `ocr` method is then called with several parameters to customize the detection and recognition process, including the `slice` parameter for handling image slices.

For a more comprehensive understanding of the slicing operation, please refer to the [slice operation documentation](./blog/slice.en.md).

## 3. Summary

In this section, you have mastered the use of PaddleOCR whl package.
