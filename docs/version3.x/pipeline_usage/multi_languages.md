---
comments: true
---

# 一、PP-OCRv5多语种文本识别介绍


PP-OCRv5 是 PP-OCR 系列的最新一代文字识别解决方案，专注于多场景、多语种的文字识别任务。在文字类型支持方面，默认配置的识别模型可准确识别简体中文、中文拼音、繁体中文、英文和日文这五大主流文字类型。同时，PP-OCRv5还提供了覆盖37种语言的多语种识别能力，包括韩文、西班牙文、法文、葡萄牙文、德文、意大利文、俄罗斯文等（具体支持语种及缩写详见[第三节](#三-支持语种及缩写)）。相较于前代 PP-OCRv3 版本，PP-OCRv5 在多语言识别准确率上实现了超过30%的提升。


![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/japan_2_res.jpg)

![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/french_0_res.jpg)

![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/german_0_res.jpg)

![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/korean_1_res.jpg)

![img](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/pipelines/ocr/ru_0.jpeg)

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
上述命令行的其他参数说明请参考通用 OCR 产线的[命令行使用方式](./OCR.md#21-命令行方式), 运行后结果会被打印到终端上：

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

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/pipelines/ocr/french01_res.png"/>


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
更过关于 `PaddleOCR` 类参数的说明参考通用 OCR 产线的[脚本方式集成](./OCR.md#22-python脚本方式集成)


## 三、指标对比

<div style="display: flex; gap: 20px;">
  <table border="1">
    <thead>
      <tr>
        <th>模型</th>
        <th>精度</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>korean_PP-OCRv5_mobile_rec</td>
        <td>90.5</td>
      </tr>
      <tr>
        <td>korean_PP-OCRv3_mobile_rec</td>
        <td>18.5</td>
      </tr>
    </tbody>
  </table>
  <table border="1">
    <thead>
      <tr>
        <th>模型</th>
        <th>精度</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>latin_PP-OCRv5_mobile_rec</td>
        <td>84.7</td>
      </tr>
      <tr>
        <td>latin_PP-OCRv3_mobile_rec</td>
        <td>37.9</td>
      </tr>
    </tbody>
  </table>
  <table border="1">
    <thead>
      <tr>
        <th>模型</th>
        <th>精度</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>eslav_PP-OCRv5_mobile_rec</td>
        <td>85.8</td>
      </tr>
      <tr>
        <td>cyrillic_PP-OCRv3_mobile_rec</td>
        <td>50.2</td>
      </tr>
    </tbody>
  </table>
</div>


## 四、 支持语种及缩写

<div style="display: flex; gap: 32px;">
  <table border="1">
    <tr>
      <th>语种</th>
      <th>描述</th>
      <th>缩写</th>
    </tr>
    <tr><td>中文</td><td>Chinese & English</td><td>ch</td></tr>
    <tr><td>英文</td><td>English</td><td>en</td></tr>
    <tr><td>法文</td><td>French</td><td>fr</td></tr>
    <tr><td>德文</td><td>German</td><td>de</td></tr>
    <tr><td>日文</td><td>Japanese</td><td>japan</td></tr>
    <tr><td>韩文</td><td>Korean</td><td>korean</td></tr>
    <tr><td>中文繁体</td><td>Chinese Traditional</td><td>chinese_cht</td></tr>
    <tr><td>南非荷兰文</td><td>Afrikaans</td><td>af</td></tr>
    <tr><td>意大利文</td><td>Italian</td><td>it</td></tr>
    <tr><td>西班牙文</td><td>Spanish</td><td>es</td></tr>
    <tr><td>波斯尼亚文</td><td>Bosnian</td><td>bs</td></tr>
    <tr><td>葡萄牙文</td><td>Portuguese</td><td>pt</td></tr>
    <tr><td>捷克文</td><td>Czech</td><td>cs</td></tr>
    <tr><td>威尔士文</td><td>Welsh</td><td>cy</td></tr>
    <tr><td>丹麦文</td><td>Danish</td><td>da</td></tr>
    <tr><td>爱沙尼亚文</td><td>Estonian</td><td>et</td></tr>
    <tr><td>爱尔兰文</td><td>Irish</td><td>ga</td></tr>
    <tr><td>克罗地亚文</td><td>Croatian</td><td>hr</td></tr>
  </table>
  <table border="1">
    <tr>
      <th>语种</th>
      <th>描述</th>
      <th>缩写</th>
    </tr>
    <tr><td>匈牙利文</td><td>Hungarian</td><td>hu</td></tr>
    <tr><td>塞尔维亚文（latin）</td><td>Serbian(latin)</td><td>rslatin</td></tr>
    <tr><td>印尼文</td><td>Indonesian</td><td>id</td></tr>
    <tr><td>欧西坦文</td><td>Occitan</td><td>oc</td></tr>
    <tr><td>冰岛文</td><td>Icelandic</td><td>is</td></tr>
    <tr><td>立陶宛文</td><td>Lithuanian</td><td>lt</td></tr>
    <tr><td>毛利文</td><td>Maori</td><td>mi</td></tr>
    <tr><td>马来文</td><td>Malay</td><td>ms</td></tr>
    <tr><td>荷兰文</td><td>Dutch</td><td>nl</td></tr>
    <tr><td>挪威文</td><td>Norwegian</td><td>no</td></tr>
    <tr><td>波兰文</td><td>Polish</td><td>pl</td></tr>
    <tr><td>斯洛伐克文</td><td>Slovak</td><td>sk</td></tr>
    <tr><td>斯洛文尼亚文</td><td>Slovenian</td><td>sl</td></tr>
    <tr><td>阿尔巴尼亚文</td><td>Albanian</td><td>sq</td></tr>
    <tr><td>瑞典文</td><td>Swedish</td><td>sv</td></tr>
    <tr><td>西瓦希里文</td><td>Swahili</td><td>sw</td></tr>
    <tr><td>塔加洛文</td><td>Tagalog</td><td>tl</td></tr>
    <tr><td>土耳其文</td><td>Turkish</td><td>tr</td></tr>
    <tr><td>乌兹别克文</td><td>Uzbek</td><td>uz</td></tr>
    <tr><td>拉丁文</td><td>Latin</td><td>la</td></tr>
    <tr><td>俄罗斯文</td><td>Russian</td><td>ru</td></tr>
    <tr><td>白俄罗斯文</td><td>Belarusian</td><td>be</td></tr>
    <tr><td>乌克兰文</td><td>Ukranian</td><td>uk</td></tr>
  </table>
</div>
