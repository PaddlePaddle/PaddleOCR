---
comments: true
---

# 1. Introduction to PP-OCRv5 Multilingual Text Recognition

[PP-OCRv5](./PP-OCRv5.md) is the latest generation text recognition solution in the PP-OCR series, focusing on multi-scenario and multilingual text recognition tasks. In terms of supported text types, the default configuration of the recognition model can accurately identify five major types: Simplified Chinese, Pinyin, Traditional Chinese, English, and Japanese. Additionally, PP-OCRv5 offers multilingual text recognition capabilities covering 106 languages, including Korean, Spanish, French, Portuguese, German, Italian, Russian, Thai, Greek and more (for a full list of supported languages and abbreviations, see [Section 4](#4-supported-languages-and-abbreviations)). Compared to the previous PP-OCRv3 version, PP-OCRv5 achieves over a 30% improvement in accuracy for multilingual text recognition.

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/french_0_res.jpg" alt="French recognition result" width="500"/>
  <br>
  <b>French Recognition Result</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/german_0_res.png" alt="German recognition result" width="500"/>
  <br>
  <b>German Recognition Result</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/korean_1_res.jpg" alt="Korean recognition result" width="500"/>
  <br>
  <b>Korean Recognition Result</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/ru_0.jpeg" alt="Russian recognition result" width="500"/>
  <br>
  <b>Russian Recognition Result</b>
</div>


<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/th_0_res.jpg" alt="Thai recognition result" width="500"/>
  <br>
  <b>Thai recognition result</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/el_0_res.jpg" alt="Greek recognition result" width="500"/>
  <br>
  <b>Greek recognition result</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/ar_0_res.jpg" alt="Arabic OCR Result" width="500"/>
  <br>
  <b>Arabic recognition Result</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/hi_0_res.jpg" alt="Hindi OCR Result" width="500"/>
  <br>
  <b>Hindi recognition Result</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/ta_0_rec.jpg" alt="Tamil OCR Result" width="500"/>
  <br>
  <b>Tamil recognition Result</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/te_0_res.png" alt="Telugu OCR Result" width="500"/>
  <br>
  <b>Telugu recognition Result</b>
</div>


## 2. Quick Start

You can specify the language for text recognition by using the `--lang` parameter when running the general OCR pipeline in the command line:

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
For explanations of the other command-line parameters, please refer to the [Command Line Usage](../../pipeline_usage/OCR.md#21-command-line-usage) section of the general OCR pipeline documentation. After running, the results will be displayed in the terminal:

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

If you specify `save_path`, the visualization results will be saved to the specified path. An example of the visualized result is shown below:

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/general_ocr_french01_res.png"/>


You can also use Python code to specify the recognition model for a particular language when initializing the general OCR pipeline via the `lang` parameter:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="fr", # Specify French recognition model with the lang parameter
    use_doc_orientation_classify=False, # Disable document orientation classification model
    use_doc_unwarping=False, # Disable text image unwarping model
    use_textline_orientation=False, # Disable text line orientation classification model
)
result = ocr.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_french01.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```
For more details on the `PaddleOCR` class parameters, please refer to the [Python Scripting Integration](../../pipeline_usage/OCR.md#22-python-scripting-integration) section of the general OCR pipeline documentation.


## 3. Performance Comparison

| Model | Download Link | Accuracy on the corresponding dataset (%) | Improvement over the previous generation model (%) ｜
|-|-|-|-|
| korean_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a> | 88.0|  65.0  | 
| latin_PP-OCRv5_mobile_rec | <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a> | 84.7 | 46.8 |
| eslav_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/eslav_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/eslav_PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a> | 81.6 | 31.4 | 
| th_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/th_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/th_PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a> | 82.68 | - |
| el_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/el_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/el_PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a> | 89.28 | - |
| en_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv5_mobile_rec_pretrained.pdparams">Pretrained Model</a> | 85.25 | 11.0 | 
| cyrillic_PP-OCRv5_mobile_rec | <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/cyrillic_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/cyrillic_PP-OCRv5_mobile_rec_pretrained.pdparams">Trained Model</a> | 80.27 | 21.2 |
| arabic_PP-OCRv5_mobile_rec | <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/arabic_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/arabic_PP-OCRv5_mobile_rec_pretrained.pdparams">Trained Model</a> | 81.27 | 22.83 |
| devanagari_PP-OCRv5_mobile_rec | <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/devanagari_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/devanagari_PP-OCRv5_mobile_rec_pretrained.pdparams">Trained Model</a> | 84.96 | 68.26 |
| te_PP-OCRv5_mobile_rec | <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/te_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/te_PP-OCRv5_mobile_rec_pretrained.pdparams">Trained Model</a> | 87.65 | 43.47 |
| ta_PP-OCRv5_mobile_rec | <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/ta_PP-OCRv5_mobile_rec_infer.tar">Inference Model</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv5_mobile_rec_pretrained.pdparams">Trained Model</a> | 94.2 | 39.23 |

 **Notes:**
 - Korean Dataset: The latest PP-OCRv5 dataset containing 5,007 Korean text images.
 - Latin Script Language Dataset: The latest PP-OCRv5 dataset containing 3,111 images of Latin script languages.
 - East Slavic Language Dataset: The latest PP-OCRv5 dataset containing a total of 7,031 text images in Russian, Belarusian, and Ukrainian.
 - Thai dataset: The latest PP-OCRv5 constructed Thai dataset contains a total of 4,261 text images for recognition.
 - Greek dataset: The latest PP-OCRv5 constructed Greek dataset contains a total of 2,799 text images for recognition.
 - English dataset: The latest PP-OCRv5 constructed English dataset contains a total of 6,530 text images for recognition.
 - Cyrillic Dataset: The latest PP-OCRv5 Cyrillic recognition dataset contains a total of 7,600 text images.
 - Tamil Dataset: The latest PP-OCRv5 Tamil recognition dataset contains a total of 2,121 text images.
 - Telugu Dataset: The latest PP-OCRv5 Telugu recognition dataset contains a total of 2,478 text images.
 - Arabic Dataset: The latest PP-OCRv5 Arabic, Sanskrit, etc. recognition dataset contains a total of 2,676 text images.
 - Devanagari Dataset: The latest PP-OCRv5 Devanagari recognition dataset contains a total of 3,611 text images.

## 4. Supported Languages and Abbreviations

| Language | Description | Abbreviation | | Language | Description | Abbreviation |
| --- | --- | --- | ---|--- | --- | --- |
| Chinese | Chinese & English | ch | | Hungarian | Hungarian | hu |
| English | English | en | | Serbian (latin) | Serbian (latin) | rs_latin |
| French | French | fr | | Indonesian | Indonesian | id |
| German | German | de | | Occitan | Occitan | oc |
| Japanese | Japanese | japan | | Icelandic | Icelandic | is |
| Korean | Korean | korean | | Lithuanian | Lithuanian | lt |
| Traditional Chinese | Chinese Traditional | chinese_cht | | Maori | Maori | mi |
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
| Ukrainian | Ukrainian | uk | | Thai | Thai | th | 
| Greek | Greek | el | | Azerbaijani | Azerbaijani | az |
| Kurdish | Kurdish | ku | | Latvian | Latvian | lv |
| Maltese | Maltese | mt | | Pali | Pali | pi |
| Romanian | Romanian | ro | | Vietnamese | Vietnamese | vi |
| Finnish | Finnish | fi | | Basque | Basque | eu | 
| Galician | Galician | gl | | Luxembourgish | Luxembourgish | lb |
| Romansh | Romansh | rm | | Catalan | Catalan | ca |
| Quechua | Quechua | qu | | Telugu | Telugu | te |
| Serbian (Cyrillic) | Serbian (Cyrillic) | sr | | Bulgarian | Bulgarian | bg |
| Mongolian | Mongolian | mn | | Abkhaz | Abkhaz | ab |
| Adyghe | Adyghe | ady | | Kabardian | Kabardian | kbd |
| Avar | Avar | av | | Dargwa | Dargwa | dar |
| Ingush | Ingush | inh | | Chechen | Chechen | ce |
| Lak | Lak | lki | | Lezgian | Lezgian | lez |
| Tabasaran | Tabasaran | tab | | Kazakh | Kazakh | kk |
| Kyrgyz | Kyrgyz | ky | | Tajik | Tajik | tg |
| Macedonian | Macedonian | mk | | Tatar | Tatar | tt |
| Chuvash | Chuvash | cv | | Bashkir | Bashkir | ba |
| Mari | Mari | mhr | | Moldovan | Moldovan | mo |
| Udmurt | Udmurt | udm | | Komi | Komi | kv |
| Ossetian | Ossetian | os | | Buriat | Buriat | bua |
| Kalmyk | Kalmyk | xal | | Tuvinian | Tuvinian | tyv |
| Sakha | Sakha | sah | | Karakalpak | Karakalpak | kaa |
| Arabic | Arabic | ar | | Persian | Persian | fa |
| Uyghur | Uyghur | ug | | Urdu | Urdu | ur |
| Pashto | Pashto | ps | | Kurdish | Kurdish | ku |
| Sindhi | Sindhi | sd | | Balochi | Balochi | bal |
| Hindi | Hindi | hi | | Marathi | Marathi | mr |
| Nepali | Nepali | ne | | Bihari | Bihari | bh |
| Maithili | Maithili | mai | | Old English | Old English | ang |
| Bhojpuri | Bhojpuri | bho | | Magahi | Magahi | mah |
| Sadri | Sadri | sck | | Newar | Newar | new |
| Konkani | Konkani | gom | | Sanskrit | Sanskrit | sa |
| Haryanvi | Haryanvi | bgc | | Tamil | Tamil | ta |


## 5. Models and Their Supported Languages

| Model | Supported Languages |
|-|-|
| korean_PP-OCRv5_mobile_rec | Korean, English |
| latin_PP-OCRv5_mobile_rec | French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua |
| eslav_PP-OCRv5_mobile_rec | Russian, Belarusian, Ukrainian, English |
| th_PP-OCRv5_mobile_rec | Thai, English |
| el_PP-OCRv5_mobile_rec | Greek, English |
| en_PP-OCRv5_mobile_rec | English |
| cyrillic_PP-OCRv5_mobile_rec | Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English |
| arabic_PP-OCRv5_mobile_rec | Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English |
| devanagari_PP-OCRv5_mobile_rec | Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi, English |
| ta_PP-OCRv5_mobile_rec | Tamil, English |
| te_PP-OCRv5_mobile_rec | Telugu, English |

**note:** `en_PP-OCRv5_mobile_rec` is an optimized version based on the `PP-OCRv5` model, specifically fine-tuned for English scenarios. It demonstrates higher recognition accuracy and better adaptability when processing English text.
