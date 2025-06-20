<div dir="rtl">
<div align="center">
  <p>
      <img width="100%" src="./docs/images/Banner.png" alt="PaddleOCR Banner"></a>
  </p>

<!-- language -->
<div dir="ltr" align="center">

[中文](./README.md) | [English](./README_en.md) | العربية | [Español](./README_es.md) | [Français](./README_fr.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Русский](./README_ru.md) | [繁体中文](./README_zh_TW.md)

</div>

<!-- icon -->

[![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![Downloads](https://img.shields.io/pypi/dm/paddleocr)](https://pypi.org/project/PaddleOCR/)
![python](https://img.shields.io/badge/python-3.8～3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)

[![Website](https://img.shields.io/badge/Website-PaddleOCR-blue?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAABmmRkdj0AAAAASUVORK5CYII=)](https://www.paddleocr.ai/)
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

</div>

## 🚀 مقدمة
منذ إصداره الأولي، حظي PaddleOCR بتقدير واسع النطاق في الأوساط الأكاديمية والصناعية والبحثية، بفضل خوارزمياته المتطورة وأدائه المثبت في تطبيقات العالم الحقيقي. وهو يدعم بالفعل مشاريع مفتوحة المصدر شهيرة مثل Umi-OCR، و OmniParser، و MinerU، و RAGFlow، مما يجعله مجموعة أدوات التعرف الضوئي على الحروف المفضلة للمطورين في جميع أنحاء العالم.

في 20 مايو 2025، كشف فريق PaddlePaddle عن PaddleOCR 3.0، المتوافق تمامًا مع الإصدار الرسمي لإطار العمل **PaddlePaddle 3.0**. يعزز هذا التحديث **دقة التعرف على النصوص**، ويضيف دعمًا لـ **التعرف على أنواع نصوص متعددة** و **التعرف على الكتابة اليدوية**، ويلبي الطلب المتزايد من التطبيقات القائمة على النماذج الكبيرة على **التحليل عالي الدقة للمستندات المعقدة**. عند دمجه مع **ERNIE 4.5T**، فإنه يعزز بشكل كبير دقة استخراج المعلومات الرئيسية. كما يقدم PaddleOCR 3.0 دعمًا لمنصات الأجهزة المحلية مثل **KUNLUNXIN** و **Ascend**. للحصول على وثائق الاستخدام الكاملة، يرجى الرجوع إلى [وثائق PaddleOCR 3.0](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

ثلاث ميزات رئيسية جديدة في PaddleOCR 3.0:
- نموذج التعرف على النصوص في جميع السيناريوهات [PP-OCRv5](./docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md): نموذج واحد يعالج خمسة أنواع مختلفة من النصوص بالإضافة إلى الكتابة اليدوية المعقدة. زادت دقة التعرف الإجمالية بمقدار 13 نقطة مئوية عن الجيل السابق. [تجربة مباشرة](https://aistudio.baidu.com/community/app/91660/webUI)

- حل تحليل المستندات العام [PP-StructureV3](./docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md): يقدم تحليلًا عالي الدقة لملفات PDF متعددة التخطيطات والسيناريوهات، متفوقًا على العديد من الحلول المفتوحة والمغلقة المصدر في المعايير العامة. [تجربة مباشرة](https://aistudio.baidu.com/community/app/518494/webUI)

- حل فهم المستندات الذكي [PP-ChatOCRv4](./docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md): مدعوم أصلاً بنموذج WenXin الكبير 4.5T، ويحقق دقة أعلى بنسبة 15 نقطة مئوية من سابقه. [تجربة مباشرة](https://aistudio.baidu.com/community/app/518493/webUI)

بالإضافة إلى توفير مكتبة نماذج متميزة، يقدم PaddleOCR 3.0 أيضًا أدوات سهلة الاستخدام تغطي تدريب النماذج والاستدلال ونشر الخدمات، حتى يتمكن المطورون من إدخال تطبيقات الذكاء الاصطناعي إلى الإنتاج بسرعة.
<div align="center">
  <p>
      <img width="100%" src="./docs/images/Arch.png" alt="PaddleOCR Architecture"></a>
  </p>
</div>



## 📣 آخر التحديثات

#### **🔥🔥 2025.06.05: إصدار PaddleOCR 3.0.1، يتضمن:**

- **تحسين بعض النماذج وتكويناتها:**
  - تحديث تكوين النموذج الافتراضي لـ PP-OCRv5، وتغيير كل من الكشف والتعرف من `mobile` إلى `server`. لتحسين الأداء الافتراضي في معظم السيناريوهات، تم تغيير المعلمة `limit_side_len` في التكوين من 736 إلى 64.
  - إضافة نموذج جديد لتصنيف اتجاه أسطر النص `PP-LCNet_x1_0_textline_ori` بدقة 99.42%. تم تحديث مصنف اتجاه أسطر النص الافتراضي لخطوط أنابيب OCR و PP-StructureV3 و PP-ChatOCRv4 إلى هذا النموذج.
  - تحسين نموذج تصنيف اتجاه أسطر النص `PP-LCNet_x0_25_textline_ori`، مما أدى إلى تحسين الدقة بمقدار 3.3 نقطة مئوية لتصل إلى الدقة الحالية البالغة 98.85%.

- **تحسينات وإصلاحات لبعض المشكلات في الإصدار 3.0.0، [التفاصيل](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)**

🔥🔥2025.05.20: الإصدار الرسمي لـ **PaddleOCR v3.0**، بما في ذلك:
- **PP-OCRv5**: نموذج التعرف على النصوص عالي الدقة لجميع السيناريوهات - نص فوري من الصور/PDF.
   1. 🌐 دعم نموذج واحد **لخمسة** أنواع من النصوص - معالجة سلسة **للصينية المبسطة والصينية التقليدية وبينين الصينية المبسطة والإنجليزية** و**اليابانية** ضمن نموذج واحد.
   2. ✍️ تحسين **التعرف على الكتابة اليدوية**: أداء أفضل بشكل ملحوظ في النصوص المتصلة المعقدة والكتابة اليدوية غير القياسية.
   3. 🎯 **زيادة في الدقة بمقدار 13 نقطة** عن PP-OCRv4، مما يحقق أداءً على أحدث طراز في مجموعة متنوعة من سيناريوهات العالم الحقيقي.

- **PP-StructureV3**: تحليل المستندات للأغراض العامة – أطلق العنان لتحليل الصور/PDFs بأحدث التقنيات لسيناريوهات العالم الحقيقي!
   1. 🧮 **تحليل PDF عالي الدقة متعدد السيناريوهات**، يتصدر كلاً من الحلول المفتوحة والمغلقة المصدر على معيار OmniDocBench.
   2. 🧠 تشمل القدرات المتخصصة **التعرف على الأختام**، **تحويل المخططات إلى جداول**، **التعرف على الجداول التي تحتوي على صيغ/صور متداخلة**، **تحليل المستندات ذات النصوص العمودية**، و**تحليل هياكل الجداول المعقدة**.

- **PP-ChatOCRv4**: فهم المستندات الذكي – استخرج المعلومات الأساسية، وليس فقط النصوص من الصور/PDFs.
   1. 🔥 **زيادة في الدقة بمقدار 15 نقطة** في استخراج المعلومات الأساسية من ملفات PDF/PNG/JPG مقارنة بالجيل السابق.
   2. 💻 دعم أصلي لـ **ERINE4.5 Turbo**، مع التوافق مع عمليات نشر النماذج الكبيرة عبر PaddleNLP و Ollama و vLLM والمزيد.
   3. 🤝 دمج [PP-DocBee2](https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2)، مما يتيح استخراج وفهم النصوص المطبوعة والمخطوطة والأختام والجداول والمخططات والعناصر الشائعة الأخرى في المستندات المعقدة.

<details>
   <summary dir="rtl"><strong>سجل التحديثات</strong></summary>


- 🔥🔥2025.03.07: إصدار **PaddleOCR v2.10**، بما في ذلك:

  - **12 نموذجًا جديدًا مطورًا ذاتيًا:**
    - **[سلسلة الكشف عن التخطيط](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)** (3 نماذج): PP-DocLayout-L، M، و S -- قادرة على اكتشاف 23 نوعًا شائعًا من التخطيطات عبر تنسيقات مستندات متنوعة (أوراق بحثية، تقارير، امتحانات، كتب، مجلات، عقود، إلخ) باللغتين الإنجليزية والصينية. تحقق ما يصل إلى **90.4% mAP@0.5**، ويمكن للميزات خفيفة الوزن معالجة أكثر من 100 صفحة في الثانية.
    - **[سلسلة التعرف على الصيغ الرياضية](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html)** (نموذجان): PP-FormulaNet-L و S -- يدعمان التعرف على أكثر من 50,000 تعبير LaTeX، ويعالجان الصيغ المطبوعة والمكتوبة بخط اليد. يوفر PP-FormulaNet-L **دقة أعلى بنسبة 6%** من النماذج المماثلة؛ PP-FormulaNet-S أسرع 16 مرة مع الحفاظ على دقة مماثلة.
    - **[سلسلة التعرف على بنية الجداول](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html)** (نموذجان): SLANeXt_wired و SLANeXt_wireless -- نماذج مطورة حديثًا مع **تحسين الدقة بنسبة 6%** على SLANet_plus في التعرف على الجداول المعقدة.
    - **[تصنيف الجداول](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_classification.html)** (نموذج واحد):
PP-LCNet_x1_0_table_cls -- مصنف فائق الخفة للجداول ذات الحدود المرئية وغير المرئية.

[اعرف المزيد](https://paddlepaddle.github.io/PaddleOCR/latest/en/update.html)

</details>

## ⚡ التشغيل السريع
### 1. تشغيل العرض التوضيحي عبر الإنترنت
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

### 2. التثبيت

قم بتثبيت PaddlePaddle بالرجوع إلى [دليل التثبيت](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)، وبعد ذلك، قم بتثبيت مجموعة أدوات PaddleOCR.

```bash
# تثبيت paddleocr
pip install paddleocr
```

### 3. تشغيل الاستدلال عبر واجهة سطر الأوامر (CLI)
```bash
# تشغيل استدلال PP-OCRv5
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False  

# تشغيل استدلال PP-StructureV3
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# احصل على مفتاح Qianfan API أولاً، ثم قم بتشغيل استدلال PP-ChatOCRv4
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# احصل على مزيد من المعلومات حول "paddleocr ocr"
paddleocr ocr --help
```

### 4. تشغيل الاستدلال عبر واجهة برمجة التطبيقات (API)
**4.1 مثال PP-OCRv5**
```python
# تهيئة كائن PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# تشغيل استدلال OCR على صورة عينة
result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

# عرض النتائج وحفظها بصيغة JSON
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

<details>
    <summary dir="rtl"><strong>4.2 مثال PP-StructureV3</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3()

# للصور
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
    )

# عرض النتائج وحفظها بصيغة JSON
for res in output:
    res.print() 
    res.save_to_json(save_path="output") 
    res.save_to_markdown(save_path="output")           
```

</details>

<details>
   <summary dir="rtl"><strong>4.3 مثال PP-ChatOCRv4</strong></summary>

```python
from paddleocr import PPChatOCRv4Doc

chat_bot_config = {
    "module_name": "chat_bot",
    "model_name": "ernie-3.5-8k",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "openai",
    "api_key": "api_key",  # your api_key
}

retriever_config = {
    "module_name": "retriever",
    "model_name": "embedding-v1",
    "base_url": "https://qianfan.baidubce.com/v2",
    "api_type": "qianfan",
    "api_key": "api_key",  # your api_key
}

pipeline = PPChatOCRv4Doc(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

visual_predict_res = pipeline.visual_predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png",
    use_common_ocr=True,
    use_seal_recognition=True,
    use_table_recognition=True,
)

mllm_predict_info = None
use_mllm = False
# إذا تم استخدام نموذج كبير متعدد الوسائط، فيجب بدء خدمة mllm المحلية. يمكنك الرجوع إلى الوثائق: https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.md لتنفيذ النشر وتحديث تكوين mllm_chat_bot_config.
if use_mllm:
    mllm_chat_bot_config = {
        "module_name": "chat_bot",
        "model_name": "PP-DocBee",
        "base_url": "http://127.0.0.1:8080/",  # your local mllm service url
        "api_type": "openai",
        "api_key": "api_key",  # your api_key
    }

    mllm_predict_res = pipeline.mllm_pred(
        input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png",
        key_list=["驾驶室准乘人数"],
        mllm_chat_bot_config=mllm_chat_bot_config,
    )
    mllm_predict_info = mllm_predict_res["mllm_res"]

visual_info_list = []
for res in visual_predict_res:
    visual_info_list.append(res["visual_info"])
    layout_parsing_result = res["layout_parsing_result"]

vector_info = pipeline.build_vector(
    visual_info_list, flag_save_bytes_vector=True, retriever_config=retriever_config
)
chat_result = pipeline.chat(
    key_list=["驾驶室准乘人数"],
    visual_info=visual_info_list,
    vector_info=vector_info,
    mllm_predict_info=mllm_predict_info,
    chat_bot_config=chat_bot_config,
    retriever_config=retriever_config,
)
print(chat_result)
```

</details>

### 5. مسرّعات الذكاء الاصطناعي المحلية
- [Huawei Ascend](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_NPU.html)
- [KUNLUNXIN](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_XPU.html)

## ⛰️ دروس متقدمة
- [درس PP-OCRv5 التعليمي](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [درس PP-StructureV3 التعليمي](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [درس PP-ChatOCRv4 التعليمي](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 نظرة سريعة على نتائج التنفيذ

<div align="center">
  <p>
     <img width="100%" src="./docs/images/demo.gif" alt="PP-OCRv5 Demo"></a>
  </p>
</div>

<div align="center">
  <p>
     <img width="100%" src="./docs/images/blue_v3.gif" alt="PP-StructureV3 Demo"></a>
  </p>
</div>

## 👩‍👩‍👧‍👦 المجتمع

| حساب PaddlePaddle الرسمي على WeChat | انضم إلى مجموعة النقاش التقني |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |


## 😃 مشاريع رائعة تستخدم PaddleOCR
لم يكن PaddleOCR ليصل إلى ما هو عليه اليوم بدون مجتمعه المذهل! 💗 شكرًا جزيلاً لجميع شركائنا القدامى، والمتعاونين الجدد، وكل من صب شغفه في PaddleOCR - سواء ذكرنا اسمك أم لا. دعمكم يشعل نارنا!

| اسم المشروع | الوصف |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|محرك RAG يعتمد على فهم عميق للوثائق.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|أداة تحويل المستندات متعددة الأنواع إلى Markdown|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|برنامج OCR مجاني ومفتوح المصدر للعمل دفعة واحدة دون اتصال بالإنترنت.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |أداة OmniParser: أداة تحليل الشاشة لوكيل واجهة المستخدم الرسومية المستند إلى الرؤية البحتة.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |نظام سؤال وجواب يعتمد على أي شيء.|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|مجموعة أدوات قوية مفتوحة المصدر مصممة لاستخراج محتوى عالي الجودة بكفاءة من مستندات PDF المعقدة والمتنوعة.|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |يتعرف على النص على الشاشة، ويترجمه ويعرض نتائج الترجمة في الوقت الفعلي.|
| [تعرف على المزيد من المشاريع](./awesome_projects.md) | [مشاريع أخرى تعتمد على PaddleOCR](./awesome_projects.md)|

## 👩‍👩‍👧‍👦 المساهمون

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>


## 🌟 نجمة

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)


## 📄 الترخيص
هذا المشروع مرخص بموجب [ترخيص Apache 2.0](LICENSE).

## 🎓 الاستشهاد الأكاديمي

```
@misc{paddleocr2020,
title={PaddleOCR, Awesome multilingual OCR toolkits based on PaddlePaddle.},
author={PaddlePaddle Authors},
howpublished = {\url{https://github.com/PaddlePaddle/PaddleOCR}},
year={2020}
}
```
</div>
</rewritten_file>
