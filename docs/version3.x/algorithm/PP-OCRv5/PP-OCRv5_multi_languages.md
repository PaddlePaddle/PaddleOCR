---
comments: true
---

# 一、PP-OCRv5多语种文字识别介绍


[PP-OCRv5](./PP-OCRv5.md) 是 PP-OCR 系列的最新一代文字识别解决方案，专注于多场景、多语种的文字识别任务。在文字类型支持方面，默认配置的识别模型可准确识别简体中文、中文拼音、繁体中文、英文和日文这五大主流文字类型。同时，PP-OCRv5还提供了覆盖39种语言的多语种文字识别能力，包括韩文、西班牙文、法文、葡萄牙文、德文、意大利文、俄罗斯文、泰文、希腊文等（具体支持语种及缩写详见[第四节](#四-支持语种及缩写)）。相较于前代 PP-OCRv3 版本，PP-OCRv5 在多语言文字识别准确率上实现了超过30%的提升。

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/french_0_res.jpg" alt="法文识别结" width="500"/>
  <br>
  <b>法文识别结果</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/german_0_res.png" alt="德文识别结" width="500"/>
  <br>
  <b>德文识别结果</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/korean_1_res.jpg" alt="韩文识别结果" width="500"/>
  <br>
  <b>韩文识别结果</b>
</div>

<br>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/ru_0.jpeg" alt="俄文识别结果" width="500"/>
  <br>
  <b>俄文识别结果</b>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/th_0_res.jpg" alt="泰文识别结" width="500"/>
  <br>
  <b>泰文识别结果</b>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/el_0_res.jpg" alt="希腊文识别结" width="500"/>
  <br>
  <b>希腊文识别结果</b>
</div>


## 二、快速使用

您可以通过在命令行中使用 `--lang` 参数，来使用指定语种的文本识别模型进行通用 OCR 产线的推理：

```bash
# 通过 `--lang` 参数指定使用法语的识别模型
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_french01.png \
    --lang fr \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --save_path ./output \
    --device gpu:0 
```
上述命令行的其他参数说明请参考通用 OCR 产线的[命令行使用方式](../../pipeline_usage/OCR.md#21-命令行方式), 运行后结果会被打印到终端上：

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

若指定了`save_path`，则会保存可视化结果在`save_path`下。可视化结果如下：

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/general_ocr_french01_res.png"/>


您也可以使用 Python 代码，在通用 OCR 产线初始化时，通过 `lang` 参数来使用指定语种的识别模型：

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="fr" # 通过 lang 参数指定使用法语的识别模型
    use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
    use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
    use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
)
result = ocr.predict("https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_french01.png")
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```
更过关于 `PaddleOCR` 类参数的说明参考通用 OCR 产线的[脚本方式集成](../../pipeline_usage/OCR.md#22-python脚本方式集成)。


## 三、指标对比

| 模型 | 模型下载链接 | 对应数据集精度（%） | 相比前代模型提升幅度 (%) |
|-|-|-|-|
| korean_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/korean_PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/korean_PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a> | 88.0| 65.0 |
| latin_PP-OCRv5_mobile_rec | <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/latin_PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a> | 84.7 | 46.8 |
| eslav_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/eslav_PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/eslav_PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a> | 81.6 | 31.4 | 
| th_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/th_PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/th_PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a> | 82.68 | - |
| el_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/el_PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/el_PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a> | 89.28 | - | 
| en_PP-OCRv5_mobile_rec |<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/en_PP-OCRv5_mobile_rec_infer.tar">推理模型</a>/<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv5_mobile_rec_pretrained.pdparams">训练模型</a> | 85.25 | 11.0 | 

 **注：**
 - 韩语数据集：PP-OCRv5 最新构建的包含了 5007 张韩语文本图片的识别数据集。
 - 拉丁字母语言数据集：PP-OCRv5 最新构建的包含了 3111 张拉丁字母语言的文本图片识别数据集。
 - 东斯拉夫语言数据集：PP-OCRv5 最新构建的包含了俄语、 白俄罗斯语和乌克兰语共计 7031 张文本图片的识别数据集。
 - 泰文数据集：PP-OCRv5 最新构建的泰文共计 4261 张文本图片的识别数据集。
 - 希腊文数据集：PP-OCRv5 最新构建的希腊文共计 2799 张文本图片的识别数据集。
 - 英文数据集：PP-OCRv5 最新构建的英文共计 6530 张文本图片的识别数据集。

## 四、 支持语种及缩写

| 语种 | 描述 | 缩写 | | 语种 | 描述 | 缩写 |
| --- | --- | --- | ---|--- | --- | --- |
| 中文 | Chinese & English | ch | | 匈牙利文 | Hungarian | hu |
| 英文 | English | en | | 塞尔维亚文（latin） | Serbian(latin) | rs_latin |
| 法文 | French | fr | | 印度尼西亚文 | Indonesian | id |
| 德文 | German | de | | 欧西坦文 | Occitan | oc |
| 日文 | Japanese | japan | | 冰岛文 | Icelandic | is |
| 韩文 | Korean | korean | | 立陶宛文 | Lithuanian | lt |
| 中文繁体 | Chinese Traditional | chinese_cht | | 毛利文 | Maori | mi |
| 南非荷兰文 | Afrikaans | af | | 马来文 | Malay | ms |
| 意大利文 | Italian | it | | 荷兰文 | Dutch | nl |
| 西班牙文 | Spanish | es | | 挪威文 | Norwegian | no |
| 波斯尼亚文 | Bosnian | bs | | 波兰文 | Polish | pl |
| 葡萄牙文 | Portuguese | pt | | 斯洛伐克文 | Slovak | sk |
| 捷克文 | Czech | cs | | 斯洛文尼亚文 | Slovenian | sl |
| 威尔士文 | Welsh | cy | | 阿尔巴尼亚文 | Albanian | sq |
| 丹麦文 | Danish | da | | 瑞典文 | Swedish | sv |
| 爱沙尼亚文 | Estonian | et | | 西瓦希里文 | Swahili | sw |
| 爱尔兰文 | Irish | ga | | 塔加洛文 | Tagalog | tl |
| 克罗地亚文 | Croatian | hr | | 土耳其文 | Turkish | tr |
| 乌兹别克文 | Uzbek | uz | | 拉丁文 | Latin | la |
| 俄罗斯文 | Russian | ru | | 白俄罗斯文 | Belarusian | be |
| 乌克兰文 | Ukranian | uk |  | 泰文 | Thai | th | 
| 希腊文 | Greek | el | |  |  |  |


## 五、模型及其支持的语种

| 模型 | 支持语种 |
|-|-|
| PP-OCRv5_server_rec | 简体中文、繁体中文、英文、日文  |
| PP-OCRv5_mobile_rec | 简体中文、繁体中文、英文、日文 |
| korean_PP-OCRv5_mobile_rec | 韩文、英文 |
| latin_PP-OCRv5_mobile_rec |英文、法文、德文、南非荷兰文、意大利文、西班牙文、波斯尼亚文、葡萄牙文、捷克文、威尔士文、丹麦文、爱沙尼亚文、爱尔兰文、克罗地亚文、乌兹别克文、匈牙利文、塞尔维亚文（latin）、印度尼西亚文、欧西坦文、冰岛文、立陶宛文、毛利文、马来文、荷兰文、挪威文、波兰文、斯洛伐克文、斯洛文尼亚文、阿尔巴尼亚文、瑞典文、西瓦希里文、塔加洛文、土耳其文、拉丁文|
| eslav_PP-OCRv5_mobile_rec | 俄罗斯文、白俄罗斯文、乌克兰文、英文 |
| th_PP-OCRv5_mobile_rec | 泰文、英文 |
| el_PP-OCRv5_mobile_rec | 希腊文、英文 |
| en_PP-OCRv5_mobile_rec | 英文 |

**注：** `en_PP-OCRv5_mobile_rec` 是在 `PP-OCRv5` 模型基础上，针对英文场景进行了定向优化，在处理英文文本时表现出更高的识别精度和更强的场景适应能力。
