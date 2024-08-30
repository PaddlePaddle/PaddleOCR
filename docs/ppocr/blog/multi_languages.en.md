---
comments: true
typora-copy-images-to: images
---

# Multi-language model

**Recent Update**

- 2022.5.8 update the `PP-OCRv3` version of the multi-language detection and recognition model, and the average recognition accuracy has increased by more than 5%.
- 2021.4.9 supports the detection and recognition of 80 languages
- 2021.4.9 supports **lightweight high-precision** English model detection and recognition

PaddleOCR aims to create a rich, leading, and practical OCR tool library, which not only provides
Chinese and English models in general scenarios, but also provides models specifically trained
in English scenarios. And multilingual models covering [80 languages](#language_abbreviations).

Among them, the English model supports the detection and recognition of uppercase and lowercase
letters and common punctuation, and the recognition of space characters is optimized:

![img](./images/img_12.jpg)

The multilingual models cover Latin, Arabic, Traditional Chinese, Korean, Japanese, etc.:

![img](./images/japan_2-20240709081138234.jpg)

![img](./images/french_0.jpg)

![img](./images/korean_0.jpg)

![img](./images/arabic_0.jpg)

This document will briefly introduce how to use the multilingual model.

## 1 Installation

### 1.1 Paddle installation

```bash linenums="1"
# cpu
pip install paddlepaddle

# gpu
pip install paddlepaddle-gpu
```

### 1.2 PaddleOCR package installation

```bash linenums="1"
pip install paddleocr
```

Build and install locally

```bash linenums="1"
python3 -m build
pip3 install dist/paddleocr-x.x.x-py3-none-any.whl # x.x.x is the version number of paddleocr
```

## 2 Quick use

### 2.1 Command line operation

View help information

```bash linenums="1"
paddleocr -h
```

- Whole image prediction (detection + recognition)

PaddleOCR currently supports 80 languages, which can be specified by the --lang parameter.
The supported languages are listed in the [table](#language_abbreviations).

``` bash
paddleocr --image_dir doc/imgs_en/254.jpg --lang=en
```

![](./images/254-20240709081442260.jpg)

![img](./images/img_02.jpg)

The result is a list. Each item contains a text box, text and recognition confidence

```text linenums="1"
[('PHO CAPITAL', 0.95723116), [[66.0, 50.0], [327.0, 44.0], [327.0, 76.0], [67.0, 82.0]]]
[('107 State Street', 0.96311164), [[72.0, 90.0], [451.0, 84.0], [452.0, 116.0], [73.0, 121.0]]]
[('Montpelier Vermont', 0.97389287), [[69.0, 132.0], [501.0, 126.0], [501.0, 158.0], [70.0, 164.0]]]
[('8022256183', 0.99810505), [[71.0, 175.0], [363.0, 170.0], [364.0, 202.0], [72.0, 207.0]]]
[('REG 07-24-201706:59 PM', 0.93537045), [[73.0, 299.0], [653.0, 281.0], [654.0, 318.0], [74.0, 336.0]]]
[('045555', 0.99346405), [[509.0, 331.0], [651.0, 325.0], [652.0, 356.0], [511.0, 362.0]]]
[('CT1', 0.9988654), [[535.0, 367.0], [654.0, 367.0], [654.0, 406.0], [535.0, 406.0]]]
......
```

- Recognition

```bash linenums="1"
paddleocr --image_dir doc/imgs_words_en/word_308.png --det false --lang=en
```

![img](./images/word_308.png)

The result is a 2-tuple, which contains the recognition result and recognition confidence

```text linenums="1"
(0.99879867, 'LITTLE')
```

- Detection

```bash linenums="1"
paddleocr --image_dir PaddleOCR/doc/imgs/11.jpg --rec false
```

The result is a list. Each item represents the coordinates of a text box.

```bash linenums="1"
[[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]]
[[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]]
[[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]]
......
```

### 2.2 Run with Python script

PPOCR is able to run with Python scripts for easy integration with your own code:

- Whole image prediction (detection + recognition)

```python linenums="1"
from paddleocr import PaddleOCR, draw_ocr

# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="korean") # The model file will be downloaded automatically when executed for the first time
img_path ='doc/imgs/korean_1.jpg'
result = ocr.ocr(img_path)
# Recognition and detection can be performed separately through parameter control
# result = ocr.ocr(img_path, det=False)  Only perform recognition
# result = ocr.ocr(img_path, rec=False)  Only perform detection
# Print detection frame and recognition result
for line in result:
    print(line)

# Visualization
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/korean.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

Visualization of results:

![img](./images/korean.jpg)

PPOCR also supports direction classification. For more detailed usage, please refer to: [whl package instructions](whl_en.md).

## 3 Custom training

PPOCR supports using your own data for custom training or fine-tune, where the recognition model can refer to [French configuration file](https://github.com/PaddlePaddle/PaddleOCR/tree/main/configs/rec/multi_language/rec_french_lite_train.yml)
Modify the training data path, dictionary and other parameters.

For specific data preparation and training process, please refer to: [Text Detection](../doc_en/detection_en.md), [Text Recognition](../doc_en/recognition_en.md), more functions such as predictive deployment,
For functions such as data annotation, you can read the complete [Document Tutorial](../../README.md).

## 4 Inference and Deployment

In addition to installing the whl package for quick forecasting,
PPOCR also provides a variety of forecasting deployment methods.
If necessary, you can read related documents:

- [Python Inference](./inference_ppocr_en.md)
- [C++ Inference](../../deploy/cpp_infer/readme.md)
- [Serving](../../deploy/hubserving/readme_en.md)
- [Mobile](../../deploy/lite/readme.md)
- [Benchmark](./benchmark_en.md)

## 5 Support languages and abbreviations

| Language  | Abbreviation | | Language  | Abbreviation |
| ---  | --- | --- | ---  | --- |
|Chinese & English|ch| |Arabic|ar|
|English|en| |Hindi|hi|
|French|fr| |Uyghur|ug|
|German|german| |Persian|fa|
|Japan|japan| |Urdu|ur|
|Korean|korean| | Serbian(latin) |rs_latin|
|Chinese Traditional |chinese_cht| |Occitan |oc|
| Italian |it| |Marathi|mr|
|Spanish |es| |Nepali|ne|
| Portuguese|pt| |Serbian(cyrillic)|rs_cyrillic|
|Russia|ru||Bulgarian |bg|
|Ukranian|uk| |Estonian |et|
|Belarusian|be| |Irish |ga|
|Telugu |te| |Croatian |hr|
|Saudi Arabia|sa| |Hungarian |hu|
|Tamil |ta| |Indonesian|id|
|Afrikaans |af| |Icelandic|is|
|Azerbaijani  |az||Kurdish|ku|
|Bosnian|bs| |Lithuanian |lt|
|Czech|cs| |Latvian |lv|
|Welsh |cy| |Maori|mi|
|Danish|da| |Malay|ms|
|Maltese |mt| |Adyghe |ady|
|Dutch |nl| |Kabardian |kbd|
|Norwegian |no| |Avar |ava|
|Polish |pl| |Dargwa |dar|
|Romanian |ro| |Ingush |inh|
|Slovak |sk| |Lak |lbe|
|Slovenian |sl| |Lezghian |lez|
|Albanian |sq| |Tabassaran |tab|
|Swedish |sv| |Bihari |bh|
|Swahili |sw| |Maithili |mai|
|Tagalog |tl| |Angika |ang|
|Turkish |tr| |Bhojpuri |bho|
|Uzbek |uz| |Magahi |mah|
|Vietnamese |vi| |Nagpur |sck|
|Mongolian |mn| |Newari |new|
|Abaza |abq| |Goan Konkani|gom|
