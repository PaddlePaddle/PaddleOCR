---
comments: true
---

# 1. Introduction to PP-OCRv5 Multilingual Text Recognition

PP-OCRv5 is the latest generation of the PP-OCR series text recognition solutions, focusing on text recognition tasks across multiple scenarios and languages. By default, the recognition model supports accurate recognition of five mainstream text types: Simplified Chinese, Chinese Pinyin, Traditional Chinese, English, and Japanese. In addition, PP-OCRv5 provides multilingual recognition capabilities covering 37 languages, including Korean, Spanish, French, Portuguese, German, Italian, Russian, and more (see [Section 4](#4-supported-languages-and-abbreviations) for the full list of supported languages and abbreviations). Compared to the previous PP-OCRv3 version, PP-OCRv5 achieves more than a 30% improvement in recognition accuracy for multilingual tasks.

![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/japan_2_res.jpg)

![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/french_0_res.jpg)

![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/german_0_res.png)

![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/korean_1_res.jpg)

![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/ru_0.jpeg)

## 2. Quick Start

You can use the `--lang` parameter in the command line to specify the text recognition model for your target language when running the general OCR pipeline:

```bash
# Use the `--lang` parameter to specify the French recognition model
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_french01.png \
    --lang fr \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --save_path ./output \
    --device gpu:0 
```
For explanations of other command line parameters, please refer to the [command line usage](../../pipeline_usage/OCR.en.md#21-command-line) of the general OCR pipeline. After execution, results will be printed to the terminal:

```bash
{'res': {'input_path': '/root/.paddlex/predict_input/general_ocr_french01.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_textline_orientation': False}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'dt_polys': array([[[119,  23],
        ...,
        [118,  75]],

       ...,

       [[109, 506],
        ...,
        [108, 556]]], dtype=int16), 'text_det_params': {'limit_side_len': 64, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([-1, ..., -1]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['mifere; la profpérité & les fuccès ac-', 'compagnent l’homme induftrieux.', 'Quel eft celui qui a acquis des ri-', 'cheffes, qui eft devenu puiffant, qui', 's’eft couvert de gloire, dont l’éloge', 'retentit par-tout, qui fiege au confeil', "du Roi? C'eft celui qui bannit la pa-", "reffe de fa maifon, & qui a dit à l'oifi-", 'veté : tu es mon ennemie.'], 'rec_scores': array([0.98409832, ..., 0.98091048]), 'rec_polys': array([[[119,  23],
        ...,
        [118,  75]],

       ...,

       [[109, 506],
        ...,
        [108, 556]]], dtype=int16), 'rec_boxes': array([[118, ...,  81],
       ...,
       [108, ..., 562]], dtype=int16)}}
```

If you specify `save_path`, the visualization results will be saved in the `save_path` directory. An example visualization is shown below:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/general_ocr_french01_res.png"/>

You can also use Python code to specify the recognition model for your target language using the `lang` parameter when initializing the general OCR pipeline:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="fr", # Specify the French recognition model via the lang parameter
    use_doc_orientation_classify=False, # Disable document orientation classification
    use_doc_unwarping=False, # Disable text image unwarping
    use_textline_orientation=False, # Disable textline orientation classification
)
result = ocr.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_french01.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```
For more details on the `PaddleOCR` class parameters, refer to the [Python script integration](../../pipeline_usage/OCR.en.md#22-python-script-integration) of the general OCR pipeline.

## 3. Benchmark Comparison

| Model | Korean Dataset Accuracy (%) |   | Model | Latin Script Languages Dataset Accuracy (%) |   | Model | East Slavic Languages Dataset Accuracy (%) |
|--|--|--|--|--|--|--|--|
| korean_PP-OCRv5_mobile_rec  | 88.0  |  | latin_PP-OCRv5_mobile_rec   | 84.7  |  | eslav_PP-OCRv5_mobile_rec   | 85.8  |
| korean_PP-OCRv3_mobile_rec  | 23.0  |  | latin_PP-OCRv3_mobile_rec   | 37.9  |  | cyrillic_PP-OCRv3_mobile_rec| 50.2  |

 **Notes:**
 - Korean Dataset: PP-OCRv5's latest dataset containing 5,007 Korean text images.
 - Latin Script Languages Dataset: The latest PP-OCRv5 recognition dataset, containing 3,111 text images in Latin script languages.
 - East Slavic Languages Dataset: PP-OCRv5's latest dataset containing a total of 7,031 Russian, Belarusian, and Ukrainian text images.

## 4. Supported Languages and Abbreviations

| Language | Description | Abbreviation | | Language | Description | Abbreviation |
| --- | --- | --- | ---|--- | --- | --- |
| Chinese | Chinese & English | ch | | Hungarian | Hungarian | hu |
| English | English | en | | Serbian (Latin) | Serbian(latin) | rslatin |
| French | French | fr | | Indonesian | Indonesian | id |
| German | German | de | | Occitan | Occitan | oc |
| Japanese | Japanese | japan | | Icelandic | Icelandic | is |
| Korean | Korean | korean | | Lithuanian | Lithuanian | lt |
| Chinese Traditional | Chinese Traditional | chinese_cht | | Maori | Maori | mi |
| Afrikaans | Afrikaans | af | | Malay | Malay | ms |
| Italian | Italian | it | | Dutch | Dutch | nl |
| Spanish | Spanish | es | | Norwegian | Norwegian | no |
| Bosnian | Bosnian | bs | | Polish | Polish | pl |
| Portuguese | Portuguese | pt | | Slovak | Slovak | sk |
| Czech | Czech | cs | | Slovenian | Slovenian | sl |
| Welsh | Welsh | cy | | Albanian | Albanian | sq |
| Danish | Danish | da | | Swedish | Swedish | sv |
| Estonian | Estonian | et | | Swahili | Swahili | sw |
| Irish | Irish | ga | | Tagalog | Tagalog | tl |
| Croatian | Croatian | hr | | Turkish | Turkish | tr |
| Uzbek | Uzbek | uz | | Latin | Latin | la |
| Russian | Russian | ru | | Belarusian | Belarusian | be |
| Ukrainian | Ukranian | uk | |  |  |  |
