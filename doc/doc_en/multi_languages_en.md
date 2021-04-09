# Multi-language model

**Recent Update**

-2021.4.9 supports the detection and recognition of 80 languages
-2021.4.9 supports **lightweight high-precision** English model detection and recognition

-[1 Installation](#Install)
    -[1.1 paddle installation](#paddleinstallation)
    -[1.2 paddleocr package installation](#paddleocr_package_install)

-[2 Quick Use](#Quick_Use)
    -[2.1 Command line operation](#Command_line_operation)
     -[2.1.1 Prediction of the whole image](#bash_detection+recognition)
     -[2.1.2 Recognition](#bash_Recognition)
     -[2.1.3 Detection](#bash_detection)
    -[2.2 python script running](#python_Script_running)
     -[2.2.1 Whole image prediction](#python_detection+recognition)
     -[2.2.2 Recognition](#python_Recognition)
     -[2.2.3 Detection](#python_detection)
-[3 Custom Training](#Custom_Training)
-[4 Supported languages and abbreviations](#language_abbreviations)

<a name="Install"></a>
## 1 Installation

<a name="paddle_install"></a>
### 1.1 paddle installation
```
# cpu
pip install paddlepaddle

# gpu
pip instll paddlepaddle-gpu
```

<a name="paddleocr_package_install"></a>
### 1.2 paddleocr package installation


pip install
```
pip install "paddleocr>=2.0.4" # 2.0.4 version is recommended
```
Build and install locally
```
python3 setup.py bdist_wheel
pip3 install dist/paddleocr-x.x.x-py3-none-any.whl # x.x.x is the version number of paddleocr
```

<a name="Quick_use"></a>
## 2 Quick use

<a name="Command_line_operation"></a>
### 2.1 Command line operation

View help information

```
paddleocr -h
```

* Whole image prediction (detection + recognition)

Paddleocr currently supports 80 languages, which can be switched by modifying the --lang parameter.
The specific supported [language] (#language_abbreviations) can be viewed in the table.

``` bash

paddleocr --image_dir doc/imgs/japan_2.jpg --lang=japan
```
![](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.0/doc/imgs/japan_2.jpg)

The result is a list, each item contains a text box, text and recognition confidence
```text
[[[671.0, 60.0], [847.0, 63.0], [847.0, 104.0], [671.0, 102.0]], ('もちもち', 0.9993342)]
[[[394.0, 82.0], [536.0, 77.0], [538.0, 127.0], [396.0, 132.0]], ('自然の', 0.9919842)]
[[[880.0, 89.0], [1014.0, 93.0], [1013.0, 127.0], [879.0, 124.0]], ('とろっと', 0.9976762)]
[[[1067.0, 101.0], [1294.0, 101.0], [1294.0, 138.0], [1067.0, 138.0]], ('后味のよい', 0.9988712)]
......
```

* Recognition

```bash
paddleocr --image_dir doc/imgs_words/japan/1.jpg --det false --lang=japan
```

![](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.0/doc/imgs_words/japan/1.jpg)

The result is a tuple, which returns the recognition result and recognition confidence

```text
('したがって', 0.99965394)
```

* Detection

```
paddleocr --image_dir PaddleOCR/doc/imgs/11.jpg --rec false
```

The result is a list, each item contains only text boxes

```
[[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]]
[[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]]
[[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]]
......
```

<a name="python_script_running"></a>
### 2.2 python script running

ppocr also supports running in python scripts for easy embedding in your own code:

* Whole image prediction (detection + recognition)

```
from paddleocr import PaddleOCR, draw_ocr

# Also switch the language by modifying the lang parameter
ocr = PaddleOCR(lang="korean") # The model file will be downloaded automatically when executed for the first time
img_path ='doc/imgs/korean_1.jpg'
result = ocr.ocr(img_path)
# Print detection frame and recognition result
for line in result:
    print(line)

# Visualization
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/korean.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

Visualization of results:
![](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.0/doc/imgs_results/korean.jpg)


* Recognition

```
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang="german")
img_path ='PaddleOCR/doc/imgs_words/german/1.jpg'
result = ocr.ocr(img_path, det=False, cls=True)
for line in result:
    print(line)
```

![](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.0/doc/imgs_words/german/1.jpg)

The result is a tuple, which only contains the recognition result and recognition confidence

```
('leider auch jetzt', 0.97538936)
```

* Detection

```python
from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR() # need to run only once to download and load model into memory
img_path ='PaddleOCR/doc/imgs_en/img_12.jpg'
result = ocr.ocr(img_path, rec=False)
for line in result:
    print(line)

# show result
from PIL import Image

image = Image.open(img_path).convert('RGB')
im_show = draw_ocr(image, result, txts=None, scores=None, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```
The result is a list, each item contains only text boxes
```bash
[[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]]
[[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]]
[[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]]
......
```

Visualization of results:
![](https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.0/doc/imgs_results/whl/12_det.jpg)

ppocr also supports direction classification. For more usage methods, please refer to: [whl package instructions](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.0/doc/doc_ch/whl.md).

<a name="Custom_training"></a>
## 3 Custom training

ppocr supports using your own data for custom training or finetune, where the recognition model can refer to [French configuration file](../../configs/rec/multi_language/rec_french_lite_train.yml)
Modify the training data path, dictionary and other parameters.

For specific data preparation and training process, please refer to: [Text Detection](../doc_ch/detection.md), [Text Recognition](../doc_ch/recognition.md), more functions such as predictive deployment,
For functions such as data annotation, you can read the complete [Document Tutorial](../../README_ch.md).

<a name="language_abbreviation"></a>
## 4 Support languages and abbreviations

| Language  | Abbreviation |
| ---  | --- |
|chinese and english|ch|
|english|en|
|french|fr|
|german|german|
|japan|japan|
|korean|korean|
|chinese traditional |ch_tra|
| Italian |it|
|Spanish |es|
| Portuguese|pt|
|Russia|ru|
|Arabic|ar|
|Hindi|hi|
|Uyghur|ug|
|Persian|fa|
|Urdu|ur|
| Serbian(latin) |rs_latin|
|Occitan |oc|
|Marathi|mr|
|Nepali|ne|
|Serbian(cyrillic)|rs_cyrillic|
|Bulgarian |bg|
|Ukranian|uk|
|Belarusian|be|
|Telugu |te|
|Kannada |kn|
|Tamil |ta|
|Afrikaans |af|
|Azerbaijani    |az|
|Bosnian|bs|
|Czech|cs|
|Welsh |cy|
|Danish|da|
|Estonian |et|
|Irish |ga|
|Croatian |hr|
|Hungarian |hu|
|Indonesian|id|
|Icelandic|is|
|Kurdish|ku|
|Lithuanian |lt|
 |Latvian |lv|
|Maori|mi|
|Malay|ms|
|Maltese |mt|
|Dutch |nl|
|Norwegian |no|
|Polish |pl|
|Romanian |ro|
|Slovak |sk|
|Slovenian |sl|
|Albanian |sq|
|Swedish |sv|
|Swahili |sw|
|Tagalog |tl|
|Turkish |tr|
|Uzbek |uz|
|Vietnamese |vi|
|Mongolian |mn|
|Abaza |abq|
|Adyghe |ady|
|Kabardian |kbd|
|Avar |ava|
|Dargwa |dar|
|Ingush |inh|
|Lak |lbe|
|Lezghian |lez|
|Tabassaran |tab|
|Bihari |bh|
|Maithili |mai|
|Angika |ang|
|Bhojpuri |bho|
|Magahi |mah|
|Nagpur |sck|
|Newari |new|
|Goan Konkani|gom|
|Saudi Arabia|sa|
