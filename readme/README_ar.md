

<div align="center">
  <p>
      <img width="800" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Banner.png" alt="تاريخ النجوم">
  </p>



<h3>مجموعة أدوات التعرف الضوئي على الحروف (OCR) الرائدة عالمياً ومحرك الذكاء الاصطناعي للمستندات</h3>

[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | العربية

<!-- icon -->

[![تنزيلات PyPI](https://static.pepy.tech/badge/paddleocr)](https://pepy.tech/projects/paddleocr)
[![مُستخدَم بواسطة](https://img.shields.io/badge/Used%20by-6k%2B%20repositories-blue)](https://github.com/PaddlePaddle/PaddleOCR/network/dependents)
![python](https://img.shields.io/badge/python-3.8~3.12-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)

[![AI Studio](https://img.shields.io/badge/PaddleOCR-_Offiical_Website-1927BA?logo=data:image/png;base64,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&labelColor=white)](https://www.paddleocr.com)
[![اسأل DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PaddlePaddle/PaddleOCR)
[![الرخصة](https://img.shields.io/badge/license-Apache_2.0-green)](../LICENSE)

</div>






**يحوّل PaddleOCR المستندات والصور إلى بيانات منظمة جاهزة للنماذج اللغوية الكبيرة (JSON/Markdown) بدقة رائدة في المجال. بأكثر من 70 ألف نجمة وثقة مشاريع رائدة مثل Dify وRAGFlow وCherry Studio، يُعد PaddleOCR الأساس المتين لبناء تطبيقات RAG والتطبيقات الوكيلية الذكية.**


## 🚀 الميزات الرئيسية

### 📄 تحليل ذكي للمستندات (جاهز للنماذج اللغوية الكبيرة)
> *تحويل المرئيات المعقدة إلى بيانات منظمة لعصر النماذج اللغوية الكبيرة.*

* **نموذج رؤية-لغة رائد للمستندات**: يتميز بنموذج **PaddleOCR-VL-1.5 (0.9B)**، النموذج خفيف الحجم الرائد في المجال للرؤية واللغة لتحليل المستندات. يتفوق في تحليل المستندات المعقدة عبر 5 تحديات رئيسية من "العالم الواقعي": **التشوه، المسح الضوئي، تصوير الشاشة، الإضاءة، والمستندات المائلة**، مع مخرجات منظمة بصيغ **Markdown** و**JSON**.
* **تحويل مدرك للبنية**: بدعم من **PP-StructureV3**، يتم تحويل ملفات PDF والصور المعقدة بسلاسة إلى **Markdown** أو **JSON**. على عكس نماذج سلسلة PaddleOCR-VL، يوفر معلومات إحداثية أدق تشمل إحداثيات خلايا الجداول وإحداثيات النصوص وغيرها.
* **كفاءة جاهزة للإنتاج**: تحقيق دقة بمستوى تجاري مع حجم صغير للغاية. يتفوق على العديد من الحلول المغلقة المصدر في المعايير المرجعية العامة مع الحفاظ على كفاءة استخدام الموارد للنشر على الأجهزة الطرفية والسحابية.

### 🔍 التعرف الشامل على النصوص (OCR للمشاهد)
> *المعيار الذهبي العالمي للكشف السريع عن النصوص متعددة اللغات.*

* **دعم أكثر من 100 لغة**: تعرف أصلي على مكتبة عالمية واسعة من اللغات. حل **PP-OCRv5** بنموذج واحد يتعامل بأناقة مع المستندات متعددة اللغات المختلطة (الصينية، الإنجليزية، اليابانية، البينيين، وغيرها).
* **إتقان العناصر المعقدة**: بالإضافة إلى التعرف القياسي على النصوص، ندعم **الكشف عن النصوص في المشاهد الطبيعية** عبر مجموعة واسعة من البيئات، بما في ذلك بطاقات الهوية، ومشاهد الشوارع، والكتب، والمكونات الصناعية.
* **قفزة في الأداء**: يقدم PP-OCRv5 **تحسيناً بنسبة 13% في الدقة** مقارنة بالإصدارات السابقة، مع الحفاظ على "الكفاءة القصوى" التي اشتهر بها PaddleOCR.

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="هندسة PaddleOCR">
  </p>
</div>

### 🛠️ منظومة مُركّزة على المطورين
* **تكامل سلس**: الخيار الأمثل لمنظومة الوكلاء الذكيين — متكامل بعمق مع **Dify وRAGFlow وPathway وCherry Studio**.
* **حلقة بيانات النماذج اللغوية الكبيرة**: خط أنابيب كامل لبناء مجموعات بيانات عالية الجودة، يوفر "محرك بيانات" مستداماً لضبط النماذج اللغوية الكبيرة.
* **نشر بنقرة واحدة**: دعم لمختلف منصات العتاد (وحدات معالجة الرسومات NVIDIA، معالجات Intel، وحدات Kunlunxin XPU، ومسرّعات الذكاء الاصطناعي المتنوعة).


## 📣 آخر التحديثات

### 🔥 إصدار PaddleOCR v3.5.0: مرونة أكبر في واجهات الاستدلال ومخرجات وثائق أغنى
* **مرونة أكبر في تبديل واجهات الاستدلال**: يدعم التبديل السلس بين الرسم البياني الثابت في Paddle، والرسم البياني الديناميكي في Paddle، وTransformers. أصبح PaddleOCR الآن متكاملاً بعمق مع منظومة Hugging Face، كما تدعم 20 من النماذج الرئيسية استخدام Transformers كواجهة للاستدلال.
* **تحويل مستندات Office إلى Markdown**: يدعم تحويل صيغ المستندات الشائعة مثل Word وExcel وPowerPoint إلى Markdown.
* **تصدير نتائج التحليل إلى DOCX**: أصبحت سلاسل `PaddleOCR-VL` و`PP-StructureV3` و`PP-DocTranslation` تدعم الآن تصدير نتائج التحليل إلى DOCX لسهولة العرض والتحرير في Microsoft Word.
* **حزمة SDK الرسمية للاستدلال في المتصفح**: تم إصدار `PaddleOCR.js`، وهي حزمة SDK الرسمية للاستدلال في المتصفح، وتدعم تشغيل `PP-OCRv5` مباشرة داخل المتصفح.

<details>
<summary><strong>2026.01.29: إصدار PaddleOCR 3.4.0</strong></summary>
* **PaddleOCR-VL-1.5 (نموذج VLM رائد بحجم 0.9B)**: نموذجنا الرائد الأحدث لتحليل المستندات متاح الآن!
    * **دقة 94.5% على OmniDocBench**: يتفوق على النماذج الكبيرة العامة الرائدة ومحللات المستندات المتخصصة.
    * **متانة في العالم الواقعي**: أول من يقدم خوارزمية **PP-DocLayoutV3** لتحديد موقع الأشكال غير المنتظمة، مع إتقان 5 سيناريوهات صعبة: *الميل، التشوه، المسح الضوئي، الإضاءة، وتصوير الشاشة*.
    * **توسيع القدرات**: يدعم الآن **التعرف على الأختام**، و**الكشف عن النصوص**، ويتوسع ليشمل **111 لغة** (بما في ذلك الخط التبتي الصيني والبنغالية).
    * **إتقان المستندات الطويلة**: يدعم الدمج التلقائي للجداول عبر الصفحات وتحديد العناوين الهرمية.
    * **جرّبه الآن**: متاح على [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) أو [موقعنا الرسمي](https://www.paddleocr.com).

</details>

<details>
<summary><strong>2025.10.16: إصدار PaddleOCR 3.3.0</strong></summary>

- إصدار PaddleOCR-VL:
    - **تقديم النموذج**:
        - **PaddleOCR-VL** هو نموذج رائد وموفر للموارد مصمم خصيصاً لتحليل المستندات. مكونه الأساسي هو PaddleOCR-VL-0.9B، وهو نموذج رؤية-لغة (VLM) صغير الحجم لكنه قوي يدمج مشفر بصري ديناميكي الدقة بأسلوب NaViT مع النموذج اللغوي ERNIE-4.5-0.3B لتمكين التعرف الدقيق على العناصر. **يدعم هذا النموذج المبتكر 109 لغات بكفاءة ويتفوق في التعرف على العناصر المعقدة (مثل النصوص والجداول والصيغ الرياضية والرسوم البيانية)، مع الحفاظ على الحد الأدنى من استهلاك الموارد**. من خلال التقييمات الشاملة على المعايير المرجعية العامة المستخدمة على نطاق واسع والمعايير الداخلية، يحقق PaddleOCR-VL أداءً رائداً في كل من تحليل المستندات على مستوى الصفحة والتعرف على العناصر. يتفوق بشكل كبير على الحلول الحالية، ويُظهر تنافسية قوية أمام نماذج VLM الرائدة، ويقدم سرعات استدلال عالية. هذه المزايا تجعله مناسباً جداً للنشر العملي في السيناريوهات الواقعية. تم إصدار النموذج على [HuggingFace](https://huggingface.co/PaddlePaddle/PaddleOCR-VL). نرحب بالجميع لتنزيله واستخدامه! يمكن العثور على مزيد من المعلومات في [PaddleOCR-VL](https://www.paddleocr.ai/latest/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.html).

    - **الميزات الأساسية**:
        - **بنية VLM صغيرة لكنها قوية**: نقدم نموذج رؤية-لغة مبتكراً مصمماً خصيصاً للاستدلال الموفر للموارد، يحقق أداءً متميزاً في التعرف على العناصر. من خلال دمج مشفر بصري ديناميكي عالي الدقة بأسلوب NaViT مع النموذج اللغوي خفيف الحجم ERNIE-4.5-0.3B، نعزز بشكل كبير قدرات التعرف وكفاءة فك التشفير. يحافظ هذا الدمج على دقة عالية مع تقليل المتطلبات الحسابية، مما يجعله مناسباً تماماً لتطبيقات معالجة المستندات الفعالة والعملية.
        - **أداء رائد في تحليل المستندات**: يحقق PaddleOCR-VL أداءً رائداً في كل من تحليل المستندات على مستوى الصفحة والتعرف على العناصر. يتفوق بشكل كبير على الحلول القائمة على خطوط الأنابيب ويُظهر تنافسية قوية أمام نماذج الرؤية-اللغة (VLMs) الرائدة في تحليل المستندات. علاوة على ذلك، يتفوق في التعرف على عناصر المستندات المعقدة، مثل النصوص والجداول والصيغ الرياضية والرسوم البيانية، مما يجعله مناسباً لمجموعة واسعة من أنواع المحتوى الصعبة، بما في ذلك النصوص المكتوبة بخط اليد والمستندات التاريخية. وهذا يجعله متعدد الاستخدامات ومناسباً لمجموعة واسعة من أنواع المستندات والسيناريوهات.
        - **دعم متعدد اللغات**: يدعم PaddleOCR-VL 109 لغات، تغطي اللغات العالمية الرئيسية، بما في ذلك على سبيل المثال لا الحصر الصينية والإنجليزية واليابانية واللاتينية والكورية، بالإضافة إلى لغات ذات خطوط وبنى مختلفة مثل الروسية (الخط السيريلي) والعربية والهندية (خط ديفاناغاري) والتايلاندية. تعزز هذه التغطية اللغوية الواسعة بشكل كبير من قابلية تطبيق نظامنا في سيناريوهات معالجة المستندات متعددة اللغات والعالمية.

- إصدار نموذج التعرف متعدد اللغات PP-OCRv5:
    - تحسين دقة وتغطية التعرف على الخط اللاتيني؛ إضافة دعم للأنظمة الكتابية السيريلية والعربية والديفاناغارية والتيلوغية والتاميلية وغيرها، مع تغطية التعرف على 109 لغات. يحتوي النموذج على 2 مليون معامل فقط، وقد زادت دقة بعض النماذج بأكثر من 40% مقارنة بالجيل السابق.

</details>


<details>
<summary><strong>2025.08.21: إصدار PaddleOCR 3.2.0</strong></summary>

- **إضافات نموذجية مهمة:**
    - تقديم التدريب والاستدلال والنشر لنماذج التعرف PP-OCRv5 باللغات الإنجليزية والتايلاندية واليونانية. **يقدم نموذج PP-OCRv5 الإنجليزي تحسيناً بنسبة 11% في السيناريوهات الإنجليزية مقارنة بنموذج PP-OCRv5 الرئيسي، مع تحقيق نماذج التعرف التايلاندي واليوناني دقة 82.68% و89.28% على التوالي.**

- **ترقيات قدرات النشر:**
    - **دعم كامل لإصداري إطار عمل PaddlePaddle 3.1.0 و3.1.1.**
    - **ترقية شاملة لحل النشر المحلي بلغة C++ لـ PP-OCRv5، يدعم الآن كلاً من Linux وWindows، مع تكافؤ الميزات ودقة مطابقة لتنفيذ Python.**
    - **يدعم الاستدلال عالي الأداء الآن CUDA 12، ويمكن إجراء الاستدلال باستخدام واجهة Paddle Inference أو ONNX Runtime.**
    - **حل النشر الخدمي عالي الاستقرار أصبح الآن مفتوح المصدر بالكامل، مما يتيح للمستخدمين تخصيص صور Docker وحزم SDK حسب الحاجة.**
    - يدعم حل النشر الخدمي عالي الاستقرار أيضاً الاستدعاء عبر طلبات HTTP المُنشأة يدوياً، مما يتيح تطوير الشيفرة البرمجية من جانب العميل بأي لغة برمجة.

- **دعم المعايير المرجعية:**
    - **تدعم جميع خطوط الإنتاج الآن قياس الأداء الدقيق، مما يتيح قياس زمن الاستدلال الشامل بالإضافة إلى بيانات زمن الاستجابة لكل طبقة ووحدة للمساعدة في تحليل الأداء. [إليك](docs/version3.x/pipeline_usage/instructions/benchmark.en.md) كيفية إعداد واستخدام ميزة قياس الأداء.**
    - **تم تحديث الوثائق لتشمل المقاييس الرئيسية للتكوينات الشائعة الاستخدام على العتاد الرئيسي، مثل زمن الاستجابة واستخدام الذاكرة، لتوفير مراجع النشر للمستخدمين.**

- **إصلاح الأخطاء:**
    - حل مشكلة فشل حفظ السجلات أثناء تدريب النموذج.
    - ترقية مكون تعزيز البيانات لنماذج الصيغ الرياضية للتوافق مع الإصدارات الأحدث من مكتبة albumentations، وإصلاح تحذيرات الجمود عند استخدام حزمة tokenizers في سيناريوهات متعددة العمليات.
    - إصلاح التناقضات في سلوكيات المفاتيح (مثل `use_chart_parsing`) في ملفات تكوين PP-StructureV3 مقارنة بخطوط الأنابيب الأخرى.

- **تحسينات أخرى:**
    - **فصل التبعيات الأساسية والاختيارية. التبعيات الأساسية الدنيا فقط مطلوبة للتعرف الأساسي على النصوص؛ يمكن تثبيت التبعيات الإضافية لتحليل المستندات واستخراج المعلومات حسب الحاجة.**
    - **تمكين دعم بطاقات الرسومات NVIDIA RTX من سلسلة 50 على Windows؛ يمكن للمستخدمين الرجوع إلى [دليل التثبيت](docs/version3.x/installation.en.md) لإصدارات إطار PaddlePaddle المقابلة.**
    - **تدعم نماذج سلسلة PP-OCR الآن إرجاع إحداثيات الأحرف المفردة.**
    - إضافة مصادر تنزيل النماذج من AIStudio وModelScope وغيرها، مما يتيح للمستخدمين تحديد مصدر تنزيل النماذج.
    - إضافة دعم تحويل الرسوم البيانية إلى جداول عبر وحدة PP-Chart2Table.
    - تحسين أوصاف الوثائق لتعزيز قابلية الاستخدام.
</details>


[سجل التحديثات](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html)

</details>


## 🚀 بداية سريعة

### الخطوة 1: جرّب عبر الإنترنت
يوفر الموقع الرسمي لـ PaddleOCR **مركز تجربة** تفاعلي و**واجهات برمجة التطبيقات (APIs)** — لا حاجة لأي إعداد، فقط انقر لتجربة الخدمة.

👉 [زيارة الموقع الرسمي](https://www.paddleocr.com)

### الخطوة 2: النشر المحلي
للاستخدام المحلي، يُرجى الرجوع إلى الوثائق التالية بناءً على احتياجاتك:

- **سلسلة PP-OCR**: انظر [وثائق PP-OCR](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/OCR.html)
- **سلسلة PaddleOCR-VL**: انظر [وثائق PaddleOCR-VL](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- **PP-StructureV3**: انظر [وثائق PP-StructureV3](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html)
- **المزيد من القدرات**: انظر [وثائق المزيد من القدرات](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/pipeline_overview.html)


## 🧩 المزيد من الميزات

- تحويل النماذج إلى صيغة ONNX: [الحصول على نماذج ONNX](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- تسريع الاستدلال باستخدام محركات مثل OpenVINO وONNX Runtime وTensorRT، أو إجراء الاستدلال باستخدام نماذج بصيغة ONNX: [الاستدلال عالي الأداء](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- تسريع الاستدلال باستخدام وحدات GPU متعددة وعمليات متعددة: [الاستدلال المتوازي لخطوط الأنابيب](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- دمج PaddleOCR في تطبيقات مكتوبة بلغات C++ وC# وJava وغيرها: [الخدمة](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## 🔄 نظرة سريعة على نتائج التنفيذ

### PP-OCRv5

<div align="center">
  <p>
       <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-OCRv5_demo.gif" alt="عرض توضيحي لـ PP-OCRv5">
  </p>
</div>



### PP-StructureV3

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PP-StructureV3_demo.gif" alt="عرض توضيحي لـ PP-StructureV3">
  </p>
</div>

### PaddleOCR-VL

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/PaddleOCR-VL_demo.gif" alt="عرض توضيحي لـ PaddleOCR-VL">
  </p>
</div>


## ✨ تابعنا

⭐ **قم بتمييز هذا المستودع بنجمة لمتابعة التحديثات والإصدارات الجديدة المثيرة، بما في ذلك إمكانيات التعرف الضوئي على الحروف وتحليل المستندات القوية!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr2.en.gif" alt="تمييز المشروع بنجمة">
  </p>
</div>


## 👩‍👩‍👧‍👦 المجتمع

<div align="center">

| حساب PaddlePaddle الرسمي على WeChat | انضم إلى مجموعة النقاش التقني |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |
</div>


## 😃 مشاريع رائعة تستفيد من PaddleOCR
لم يكن PaddleOCR ليصل إلى ما هو عليه اليوم لولا مجتمعه المذهل! 💗 شكر جزيل لجميع شركائنا القدامى والمتعاونين الجدد وكل من بذل شغفه في PaddleOCR — سواء ذكرنا اسمه أم لا. دعمكم هو وقود حماسنا!

<div align="center">

| اسم المشروع | الوصف |
| ------------ | ----------- |
| [Dify](https://github.com/langgenius/dify) <a href="https://github.com/langgenius/dify"><img src="https://img.shields.io/github/stars/langgenius/dify"></a>|منصة جاهزة للإنتاج لتطوير سير العمل الوكيلي.|
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|محرك RAG قائم على الفهم العميق للمستندات.|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|إطار عمل Python ETL لمعالجة التدفقات والتحليلات الآنية وخطوط أنابيب النماذج اللغوية الكبيرة وRAG.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|أداة تحويل المستندات متعددة الأنواع إلى Markdown.|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|برنامج OCR مجاني، مفتوح المصدر، للمعالجة الدفعية دون اتصال بالإنترنت.|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|تطبيق سطح مكتب يدعم مزودي نماذج لغوية كبيرة متعددين.|
| [haystack](https://github.com/deepset-ai/haystack)<a href="https://github.com/deepset-ai/haystack"><img src="https://img.shields.io/github/stars/deepset-ai/haystack"></a> |إطار عمل لتنظيم الذكاء الاصطناعي لبناء تطبيقات نماذج لغوية كبيرة قابلة للتخصيص وجاهزة للإنتاج.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |OmniParser: أداة تحليل الشاشة لوكيل واجهة المستخدم الرسومية القائم على الرؤية البحتة.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |الأسئلة والأجوبة المبنية على أي شيء.|
| [تعرّف على المزيد من المشاريع](./awesome_projects.md) | [المزيد من المشاريع المبنية على PaddleOCR](./awesome_projects.md)|
</div>

## 👩‍👩‍👧‍👦 المساهمون

<div align="center">
<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>
</div>

## 🌟 النجوم

<div align="center">
  <p>
      <img width="800" src="https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date" alt="تاريخ النجوم">
  </p>
</div>


## 📄 الرخصة
هذا المشروع مُصدر بموجب [رخصة Apache 2.0](LICENSE).

## 🎓 الاستشهاد

```bibtex
@misc{cui2025paddleocr30technicalreport,
      title={PaddleOCR 3.0 Technical Report},
      author={Cheng Cui and Ting Sun and Manhui Lin and Tingquan Gao and Yubo Zhang and Jiaxuan Liu and Xueqing Wang and Zelun Zhang and Changda Zhou and Hongen Liu and Yue Zhang and Wenyu Lv and Kui Huang and Yichao Zhang and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2507.05595},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05595},
}

@misc{cui2025paddleocrvlboostingmultilingualdocument,
      title={PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model},
      author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Handong Zheng and Jing Zhang and Jun Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2025},
      eprint={2510.14528},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.14528},
}

@misc{cui2026paddleocrvl15multitask09bvlm,
      title={PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing},
      author={Cheng Cui and Ting Sun and Suyin Liang and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Xueqing Wang and Changda Zhou and Hongen Liu and Manhui Lin and Yue Zhang and Yubo Zhang and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2026},
      eprint={2601.21957},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.21957},
}
```
