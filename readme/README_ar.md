<div align="center">
    <p>
        <img width="100%" src="../docs/images/Banner.png" alt="PaddleOCR Banner">
    </p>

<!-- language -->
<p>

[English](../README.md) | [简体中文](./README_cn.md) | [繁體中文](./README_tcn.md) | [日本語](./README_ja.md) | [한국어](./README_ko.md) | [Français](./README_fr.md) | [Русский](./README_ru.md) | [Español](./README_es.md) | العربية

<!-- icon -->

[![stars](https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf)](https://github.com/PaddlePaddle/PaddleOCR)
[![Downloads](https://img.shields.io/pypi/dm/paddleocr)](https://pypi.org/project/PaddleOCR/)
![python](https://img.shields.io/badge/python-3.8～3.13-aff.svg)
![os](https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20gpu%2C%20xpu%2C%20npu-yellow.svg)


[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)

</p>
</div>

<div dir="rtl">

## 🚀 مقدمة
منذ إصداره الأولي، حظي PaddleOCR بتقدير واسع النطاق في الأوساط الأكاديمية والصناعية والبحثية، بفضل خوارزمياته المتطورة وأدائه المثبت في تطبيقات العالم الحقيقي. وهو يدعم بالفعل مشاريع مفتوحة المصدر شهيرة مثل Umi-OCR، و OmniParser، و MinerU، و RAGFlow، مما يجعله مجموعة أدوات التعرف الضوئي على الحروف المفضلة للمطورين في جميع أنحاء العالم.

في 20 مايو 2025، كشف فريق PaddlePaddle عن PaddleOCR 3.0، المتوافق تمامًا مع الإصدار الرسمي لإطار العمل **PaddlePaddle 3.0**. يعزز هذا التحديث **دقة التعرف على النصوص**، ويضيف دعمًا لـ **التعرف على أنواع نصوص متعددة** و **التعرف على الكتابة اليدوية**، ويلبي الطلب المتزايد من التطبيقات القائمة على النماذج الكبيرة على **التحليل عالي الدقة للمستندات المعقدة**. عند دمجه مع **ERNIE 4.5**، فإنه يعزز بشكل كبير دقة استخراج المعلومات الرئيسية. كما يقدم PaddleOCR 3.0 دعمًا لمسرعات الذكاء الاصطناعي الصينية غير المتجانسة مثل **Kunlunxin** و **Ascend**. للحصول على وثائق الاستخدام الكاملة، يرجى الرجوع إلى [وثائق PaddleOCR 3.0](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html).

##### ثلاث ميزات رئيسية جديدة في PaddleOCR 3.0:
نموذج التعرف على النصوص في جميع السيناريوهات [PP-OCRv5](../docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md): نموذج واحد يعالج خمسة أنواع مختلفة من النصوص بالإضافة إلى الكتابة اليدوية المعقدة. زادت دقة التعرف الإجمالية بمقدار 13 نقطة مئوية عن الجيل السابق. [تجربة مباشرة](https://aistudio.baidu.com/community/app/91660/webUI)

حل تحليل المستندات العام [PP-StructureV3](../docs/version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md): يقدم تحليلًا عالي الدقة لملفات PDF متعددة التخطيطات والسيناريوهات، متفوقًا على العديد من الحلول المفتوحة والمغلقة المصدر في المعايير العامة. [تجربة مباشرة](https://aistudio.baidu.com/community/app/518494/webUI)

حل فهم المستندات الذكي [PP-ChatOCRv4](../docs/version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md): مدعوم أصلاً بنموذج **ERNIE 4.5**، ويحقق دقة أعلى بنسبة 15 نقطة مئوية من سابقه. [تجربة مباشرة](https://aistudio.baidu.com/community/app/518493/webUI)

بالإضافة إلى توفير مكتبة نماذج متميزة، يقدم PaddleOCR 3.0 أيضًا أدوات سهلة الاستخدام تغطي تدريب النماذج والاستدلال ونشر الخدمات، حتى يتمكن المطورون من إدخال تطبيقات الذكاء الاصطناعي إلى الإنتاج بسرعة.

<div align="center">
  <p>
      <img width="100%" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/Arch.jpg" alt="PaddleOCR Architecture">
  </p>
</div>

**ملاحظة خاصة**: يقدم PaddleOCR 3.x العديد من التغييرات الكبيرة في الواجهات. **من المرجح أن الشيفرة القديمة المبنية على PaddleOCR 2.x غير متوافقة مع PaddleOCR 3.x**. يرجى التأكد من أن الوثائق التي تقرأها تتوافق مع إصدار PaddleOCR الذي تستخدمه. [تشرح هذه الوثيقة](https://paddlepaddle.github.io/PaddleOCR/latest/en/update/upgrade_notes.html) أسباب الترقية والتغييرات الرئيسية من PaddleOCR 2.x إلى 3.x.

## 📣 آخر التحديثات

<h4 dir="rtl"><strong>2025.08.21: إصدار <bdi dir="ltr">PaddleOCR 3.2.0</bdi>، يتضمن:</strong></h4>
<ul dir="rtl">
  <li><strong>تحديثات النماذج الرئيسية:</strong>
    <ul dir="rtl">
      <li>
        تمت إضافة ميزات التدريب والاستدلال والنشر لنماذج التعرف <bdi dir="ltr">PP-OCRv5</bdi> للغات الإنجليزية والتايلاندية واليونانية.
        <br>
        <bdi dir="ltr">النموذج الإنجليزي</bdi> حقق زيادة بنسبة 11% في الدقة مقارنة بالإصدار السابق من <bdi dir="ltr">PP-OCRv5</bdi> في سيناريوهات اللغة الإنجليزية. 
        <bdi dir="ltr">النموذج التايلاندي</bdi> حقق دقة بنسبة 82.68%،
        و<bdi dir="ltr">النموذج اليوناني</bdi> حقق دقة بنسبة 89.28%.
      </li>
    </ul>
  </li>
  <li><strong>تحسين إمكانيات النشر:</strong>
    <ul dir="rtl">
      <li>
        <bdi dir="ltr">دعم كامل لإصداري PaddlePaddle 3.1.0 و 3.1.1.</bdi>
      </li>
      <li>
        <bdi dir="ltr">إعادة هيكلة كاملة لحل النشر المحلي بلغة C++، متوافق مع Linux و Windows، ليحقق نفس الوظائف والدقة كما في إصدار Python.</bdi>
      </li>
      <li>
        <bdi dir="ltr">دعم CUDA 12</bdi> للاستدلال عالي الأداء، مع خيار استخدام <bdi dir="ltr">Paddle Inference</bdi> أو <bdi dir="ltr">ONNX Runtime</bdi>.
      </li>
      <li>
        <bdi dir="ltr">إتاحة الشيفرة المصدرية بالكامل</bdi> لحل النشر كخدمة عالية الاستقرار، مما يمكن المستخدمين من تخصيص صور Docker أو SDK حسب احتياجاتهم.
      </li>
      <li>
        يدعم حل النشر كخدمة عالية الاستقرار أيضاً استدعاءات HTTP يدوياً، مما يسمح للعملاء بالنشر بأي لغة.
      </li>
    </ul>
  </li>
  <li><strong>دعم مؤشرات الأداء:</strong>
    <ul dir="rtl">
      <li>
        <bdi dir="ltr">توفير وظيفة مؤشرات أداء مفصلة</bdi> عبر سلسلة الإنتاج بالكامل، لقياس زمن الاستدلال من البداية للنهاية وأزمنة تنفيذ الطبقات والوحدات المختلفة، لتسهيل تحليل الأداء.
      </li>
      <li>
        <bdi dir="ltr">توفر الوثائق القيم المرجعية (زمن الاستدلال، استهلاك الذاكرة، إلخ) على أهم منصات العتاد</bdi> لمساعدة المستخدمين في اتخاذ قرارات النشر.
      </li>
    </ul>
  </li>
  <li><strong>تصحيح الأخطاء:</strong>
    <ul dir="rtl">
      <li>
        تم حل مشكلة عدم حفظ السجلات أثناء تدريب النموذج.
      </li>
      <li>
        <bdi dir="ltr">تكييف جزء زيادة البيانات لنموذج المعادلات مع إصدار albumentations الجديد</bdi> وحل تحذير التعليق المحتمل عند استخدام tokenizers في تعدد العمليات.
      </li>
      <li>
        <bdi dir="ltr">تصحيح عدم تطابق بعض الإشارات مثل use_chart_parsing في ملف إعدادات PP-StructureV3 مقارنة بإصدارات أخرى.</bdi>
      </li>
    </ul>
  </li>
  <li><strong>تحديثات أخرى:</strong>
    <ul dir="rtl">
      <li>
        <bdi dir="ltr">فصل التبعيات الأساسية عن الاختيارية؛ وظائف التعرف الأساسية تتطلب فقط الحد الأدنى من التبعيات، بينما يمكن تثبيت ميزات إضافية مثل تحليل الوثائق أو استخراج المعلومات حسب الحاجة.</bdi>
      </li>
      <li>
        <bdi dir="ltr">دعم وحدات معالجة الرسومات NVIDIA السلسلة 50 في بيئة Windows، يرجى مراجعة <a href="../docs/version3.x/installation.en.md">دليل التثبيت</a> لاختيار إصدار Paddle المناسب.</bdi>
      </li>
      <li>
        <bdi dir="ltr">نماذج سلسلة PP-OCR تدعم الآن إرجاع إحداثيات كل حرف.</bdi>
      </li>
      <li>
        تمت إضافة مصادر تحميل النماذج مثل AIStudio وModelScope، مع إمكانية الاختيار بينها.
      </li>
      <li>
        دعم الاستدلال لوحدة تحويل الرسومات إلى جداول <bdi dir="ltr">PP-Chart2Table</bdi>.
      </li>
      <li>
        <bdi dir="ltr">تحسين بعض الأوصاف في الوثائق لتعزيز سهولة الاستخدام.</bdi>
      </li>
    </ul>
  </li>
</ul>

<h4 dir="rtl"><strong>2025.08.15: إصدار <bdi dir="ltr">PaddleOCR 3.1.1</bdi>، يتضمن:</strong></h4>
<ul dir="rtl">
  <li><strong>تصحيح الأخطاء:</strong>
    <ul dir="rtl">
      <li>
        تمت إضافة الطرق الناقصة <bdi dir="ltr">save_vector</bdi>، <bdi dir="ltr">save_visual_info_list</bdi>، <bdi dir="ltr">load_vector</bdi>، و<bdi dir="ltr">load_visual_info_list</bdi> إلى فئة <bdi dir="ltr">PP-ChatOCRv4</bdi>.
      </li>
      <li>
        تمت إضافة المعاملات الناقصة <bdi dir="ltr">glossary</bdi> و<bdi dir="ltr">llm_request_interval</bdi> إلى دالة <bdi dir="ltr">translate</bdi> في فئة <bdi dir="ltr">PPDocTranslation</bdi>.
      </li>
    </ul>
  </li>
  <li><strong>تحسين الوثائق:</strong>
    <ul dir="rtl">
      <li>تمت إضافة عرض توضيحي إلى وثائق <bdi dir="ltr">MCP</bdi>.</li>
      <li>تمت إضافة توضيحات حول إصدارات <bdi dir="ltr">PaddlePaddle</bdi> و<bdi dir="ltr">PaddleOCR</bdi> المستخدمة في اختبارات مؤشرات الأداء.</li>
      <li>تم تصحيح الأخطاء والنواقص في وثائق خط إنتاج ترجمة المستندات.</li>
    </ul>
  </li>
  <li><strong>أخرى:</strong>
    <ul dir="rtl">
      <li>
        تعديل تبعيات خادم <bdi dir="ltr">MCP</bdi>: تم استخدام مكتبة <bdi dir="ltr">puremagic</bdi> (بايثون فقط) بدلاً من <bdi dir="ltr">python-magic</bdi> لتقليل مشاكل التثبيت.
      </li>
      <li>
        إعادة اختبار مؤشرات أداء <bdi dir="ltr">PP-OCRv5</bdi> باستخدام إصدار <bdi dir="ltr">PaddleOCR 3.1.0</bdi> وتحديث الوثائق.
      </li>
    </ul>
  </li>
</ul>

<h4 dir="rtl">🔥🔥<strong>2025.06.29: إصدار <bdi dir="ltr">PaddleOCR 3.1.0</bdi>، يتضمن:</strong></h4>
<ul dir="rtl">
  <li><strong>النماذج وخطوط الأنابيب الرئيسية:</strong>
    <ul dir="rtl">
      <li>
        <strong>تمت إضافة نموذج التعرف على النصوص متعدد اللغات <bdi dir="ltr">PP-OCRv5</bdi></strong>، والذي يدعم تدريب واستدلال نماذج التعرف على النصوص في 37 لغة، بما في ذلك الفرنسية، الإسبانية، البرتغالية، الروسية، الكورية وغيرها. <strong>تحسنت الدقة المتوسطة بنسبة تزيد عن 30%.</strong>
        <a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.html">التفاصيل</a>
      </li>
      <li>
        تم ترقية نموذج <bdi dir="ltr">PP-Chart2Table</bdi> في <bdi dir="ltr">PP-StructureV3</bdi>، مما عزز أكثر من إمكانية تحويل المخططات إلى جداول. في مجموعات التقييم الداخلية، ارتفع المقياس (<bdi dir="ltr">RMS-F1</bdi>) بمقدار <strong>9.36 نقطة مئوية (71.24% → 80.60%)</strong>.
      </li>
      <li>
        تم إطلاق خط أنابيب ترجمة المستندات الجديد <bdi dir="ltr">PP-DocTranslation</bdi>، المبني على <bdi dir="ltr">PP-StructureV3</bdi> و <bdi dir="ltr">ERNIE 4.5</bdi>، ويدعم ترجمة مستندات <bdi dir="ltr">Markdown</bdi>، ومستندات <bdi dir="ltr">PDF</bdi> ذات التنسيقات المعقدة وصور المستندات، مع حفظ النتائج كمستندات <bdi dir="ltr">Markdown</bdi>.
        <a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-DocTranslation.html">التفاصيل</a>
      </li>
    </ul>
  </li>
  <li><strong>دعم MCP:</strong><a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/mcp_server.html">التفاصيل</a>
    <ul dir="rtl">
      <li>
        <strong>يدعم خطوط أنابيب OCR و PP-StructureV3.</strong>
      </li>
      <li>
        يدعم ثلاثة أوضاع عمل: مكتبة Python المحلية، خدمة السحابة المجتمعية AIStudio، وخدمة الاستضافة الذاتية.
      </li>
      <li>
        يدعم استدعاء الخدمات المحلية عبر stdio والخدمات البعيدة عبر Streamable HTTP.
      </li>
    </ul>
  </li>
  <li><strong>تحسين الوثائق:</strong>
    <ul dir="rtl">
      <li>تم تحسين الشروحات في بعض الأدلة للمستخدمين لتوفير تجربة قراءة أكثر سلاسة.</li>
    </ul>
  </li>
</ul>


<details>
    <summary dir="rtl"><strong>سجل التحديثات</strong></summary>


<h4 dir="rtl">🔥🔥<strong>2025.06.26: إصدار <bdi dir="ltr">PaddleOCR 3.0.3</bdi>، يتضمن:</strong></h4>
<ul dir="rtl">
  <li> تصحيح خلل: تم حل المشكلة التي لم يكن فيها معلمة <code>enable_mkldnn</code> فعّالة، واستعادة السلوك الافتراضي باستخدام MKL-DNN للاستدلال بوحدة المعالجة المركزية.</li>
</ul>

<h4 dir="rtl">🔥🔥<strong>2025.06.19: إصدار <bdi dir="ltr">PaddleOCR 3.0.2</bdi>، يتضمن:</strong></h4>
<ul dir="rtl">
  <li><strong>ميزات جديدة:</strong>
    <ul dir="rtl">
      <li>تم تغيير مصدر التنزيل الافتراضي من <bdi dir="ltr"><code>BOS</code></bdi> إلى <bdi dir="ltr"><code>HuggingFace</code></bdi>. يمكن للمستخدمين أيضًا تغيير متغير البيئة <bdi dir="ltr"><code>PADDLE_PDX_MODEL_SOURCE</code></bdi> إلى <bdi dir="ltr"><code>BOS</code></bdi> لإعادة تعيين مصدر تنزيل النموذج إلى <bdi dir="ltr">Baidu Object Storage (BOS)</bdi>.</li>
      <li>تمت إضافة أمثلة استدعاء الخدمة لست لغات — <bdi dir="ltr">C++</bdi>, <bdi dir="ltr">Java</bdi>, <bdi dir="ltr">Go</bdi>, <bdi dir="ltr">C#</bdi>, <bdi dir="ltr">Node.js</bdi>, و <bdi dir="ltr">PHP</bdi> — لخطوط الأنابيب مثل <bdi dir="ltr">PP-OCRv5</bdi>, <bdi dir="ltr">PP-StructureV3</bdi>, و <bdi dir="ltr">PP-ChatOCRv4</bdi>.</li>
      <li>تحسين خوارزمية فرز تقسيم التخطيط في خط أنابيب <bdi dir="ltr">PP-StructureV3</bdi>، مما يعزز منطق الفرز للتخطيطات العمودية المعقدة لتقديم نتائج أفضل.</li>
      <li>تحسين منطق اختيار النموذج: عند تحديد لغة وعدم تحديد إصدار النموذج، سيقوم النظام تلقائيًا بتحديد أحدث إصدار للنموذج يدعم تلك اللغة.</li>
      <li>تعيين حد أعلى افتراضي لحجم ذاكرة التخزين المؤقت لـ <bdi dir="ltr">MKL-DNN</bdi> لمنع النمو غير المحدود، مع السماح للمستخدمين أيضًا بتكوين سعة ذاكرة التخزين المؤقت.</li>
      <li>تحديث التكوينات الافتراضية للاستدلال عالي الأداء لدعم تسريع <bdi dir="ltr">Paddle MKL-DNN</bdi> وتحسين منطق الاختيار التلقائي للتكوين لخيارات أكثر ذكاءً.</li>
      <li>تعديل منطق الحصول على الجهاز الافتراضي لمراعاة الدعم الفعلي لأجهزة الحوسبة بواسطة إطار عمل <bdi dir="ltr">Paddle</bdi> المثبت، مما يجعل سلوك البرنامج أكثر بديهية.</li>
      <li>إضافة مثال <bdi dir="ltr">Android</bdi> لـ <bdi dir="ltr">PP-OCRv5</bdi>. <a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/on_device_deployment.html">التفاصيل</a>.</li>
    </ul>
  </li>
  <li><strong>إصلاحات الأخطاء:</strong>
    <ul dir="rtl">
      <li>إصلاح مشكلة عدم تفعيل بعض معلمات <bdi dir="ltr">CLI</bdi> في <bdi dir="ltr">PP-StructureV3</bdi>.</li>
      <li>حل مشكلة حيث لا تعمل <bdi dir="ltr"><code>export_paddlex_config_to_yaml</code></bdi> بشكل صحيح في بعض الحالات.</li>
      <li>تصحيح التناقض بين السلوك الفعلي لـ <bdi dir="ltr"><code>save_path</code></bdi> ووصفه في الوثائق.</li>
      <li>إصلاح أخطاء تعدد الخيوط المحتملة عند استخدام <bdi dir="ltr">MKL-DNN</bdi> في نشر الخدمة الأساسية.</li>
      <li>تصحيح أخطاء ترتيب القنوات في المعالجة المسبقة للصور لنموذج <bdi dir="ltr">Latex-OCR</bdi>.</li>
      <li>إصلاح أخطاء ترتيب القنوات في حفظ الصور المرئية داخل وحدة التعرف على النص.</li>
      <li>حل أخطاء ترتيب القنوات في نتائج الجداول المرئية داخل خط أنابيب <bdi dir="ltr">PP-StructureV3</bdi>.</li>
      <li>إصلاح مشكلة تجاوز السعة في حساب <bdi dir="ltr"><code>overlap_ratio</code></bdi> في ظروف خاصة للغاية في خط أنابيب <bdi dir="ltr">PP-StructureV3</bdi>.</li>
    </ul>
  </li>
  <li><strong>تحسينات على الوثائق:</strong>
    <ul dir="rtl">
      <li>تحديث وصف المعلمة <bdi dir="ltr"><code>enable_mkldnn</code></bdi> في الوثائق لتعكس بدقة السلوك الفعلي للبرنامج.</li>
      <li>إصلاح الأخطاء في الوثائق المتعلقة بمعلمات <bdi dir="ltr"><code>lang</code></bdi> و <bdi dir="ltr"><code>ocr_version</code></bdi>.</li>
      <li>إضافة تعليمات لتصدير ملفات تكوين خط الإنتاج عبر <bdi dir="ltr">CLI</bdi>.</li>
      <li>إصلاح الأعمدة المفقودة في جدول بيانات أداء <bdi dir="ltr">PP-OCRv5</bdi>.</li>
      <li>تحسين مقاييس الأداء لـ <bdi dir="ltr">PP-StructureV3</bdi> عبر تكوينات مختلفة.</li>
    </ul>
  </li>
  <li><strong>أخرى:</strong>
    <ul dir="rtl">
      <li>تخفيف قيود الإصدار على التبعيات مثل <bdi dir="ltr">numpy</bdi> و <bdi dir="ltr">pandas</bdi>، واستعادة الدعم لـ <bdi dir="ltr">Python 3.12</bdi>.</li>
    </ul>
  </li>
</ul>


<h4 dir="rtl"><strong>🔥🔥 2025.06.05: إصدار <bdi dir="ltr">PaddleOCR 3.0.1</bdi>، يتضمن:</strong></h4>
<ul dir="rtl">
 <li><strong>تحسين بعض النماذج وتكويناتها:</strong>
    <ol dir="rtl">
      <li>تحديث تكوين النموذج الافتراضي لـ <bdi dir="ltr">PP-OCRv5</bdi>، وتغيير كل من الكشف والتعرف من <bdi dir="ltr"><code>mobile</code></bdi> إلى <bdi dir="ltr"><code>server</code></bdi>. لتحسين الأداء الافتراضي في معظم السيناريوهات، تم تغيير المعلمة <bdi dir="ltr"><code>limit_side_len</code></bdi> في التكوين من 736 إلى 64.</li>
      <li>إضافة نموذج جديد لتصنيف اتجاه أسطر النص <bdi dir="ltr"><code>PP-LCNet_x1_0_textline_ori</code></bdi> بدقة 99.42%. تم تحديث مصنف اتجاه أسطر النص الافتراضي لخطوط أنابيب <bdi dir="ltr">OCR</bdi> و <bdi dir="ltr">PP-StructureV3</bdi> و <bdi dir="ltr">PP-ChatOCRv4</bdi> إلى هذا النموذج.</li>
      <li>تحسين نموذج تصنيف اتجاه أسطر النص <bdi dir="ltr"><code>PP-LCNet_x0_25_textline_ori</code></bdi>، مما أدى إلى تحسين الدقة بمقدار 3.3 نقطة مئوية لتصل إلى الدقة الحالية البالغة 98.85%.</li>
    </ol>
      <li><strong>تحسينات وإصلاحات لبعض المشكلات في الإصدار 3.0.0، <a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html">التفاصيل</a></strong></li>
</ul>

🔥🔥2025.05.20: الإصدار الرسمي لـ **PaddleOCR v3.0**، بما في ذلك:
<h4 dir="rtl"><bdi dir="ltr">PP-OCRv5</bdi>: نموذج التعرف على النصوص عالي الدقة لجميع السيناريوهات – نص فوري من الصور/PDF.</h4>
<ol dir="rtl">
  <li>🌐 دعم نموذج واحد **لخمسة** أنواع من النصوص - معالجة سلسة **للصينية المبسطة والصينية التقليدية وبينين الصينية المبسطة والإنجليزية** و**اليابانية** ضمن نموذج واحد.</li>
  <li>✍️ تحسين **التعرف على الكتابة اليدوية**: أداء أفضل بشكل ملحوظ في النصوص المتصلة المعقدة والكتابة اليدوية غير القياسية.</li>
  <li>🎯 **زيادة في الدقة بمقدار 13 نقطة** عن <bdi dir="ltr">PP-OCRv4</bdi>، مما يحقق أداءً على أحدث طراز في مجموعة متنوعة من سيناريوهات العالم الحقيقي.</li>
</ol>

####  <h4 dir="rtl"><bdi dir="ltr">PP-StructureV3</bdi>: تحليل المستندات للأغراض العامة – أطلق العنان لتحليل الصور/PDFs بأحدث التقنيات لسيناريوهات العالم الحقيقي!</h4>
<ol dir="rtl">
  <li>🧮 **تحليل PDF عالي الدقة متعدد السيناريوهات**، يتصدر كلاً من الحلول المفتوحة والمغلقة المصدر على معيار <bdi dir="ltr">OmniDocBench</bdi>.</li>
  <li>🧠 تشمل القدرات المتخصصة **التعرف على الأختام**، **تحويل المخططات إلى جداول**، **التعرف على الجداول التي تحتوي على صيغ/صور متداخلة**، **تحليل المستندات ذات النصوص العمودية**، و**تحليل هياكل الجداول المعقدة**.</li>
</ol>

#### <h4 dir="rtl"><bdi dir="ltr">PP-ChatOCRv4</bdi>: فهم المستندات الذكي – استخرج المعلومات الأساسية، وليس فقط النصوص من الصور/PDFs.</h4>
<ol dir="rtl">
  <li>🔥 **زيادة في الدقة بمقدار 15 نقطة** في استخراج المعلومات الأساسية من ملفات <bdi dir="ltr">PDF/PNG/JPG</bdi> مقارنة بالجيل السابق.</li>
  <li>💻 دعم أصلي لـ <bdi dir="ltr">ERNIE 4.5</bdi>، مع التوافق مع عمليات نشر النماذج الكبيرة عبر <bdi dir="ltr">PaddleNLP</bdi> و <bdi dir="ltr">Ollama</bdi> و <bdi dir="ltr">vLLM</bdi> والمزيد.</li>
  <li>🤝 دمج <a href="https://github.com/PaddlePaddle/PaddleMIX/tree/develop/paddlemix/examples/ppdocbee2" dir="ltr">PP-DocBee2</a>، مما يتيح استخراج وفهم النصوص المطبوعة والمخطوطة والأختام والجداول والمخططات والعناصر الشائعة الأخرى في المستندات المعقدة.</li>
</ol>

<p align="right">[<a href="https://paddlepaddle.github.io/PaddleOCR/latest/en/update/update.html">سجل التحديثات</a>]</p>

</details>

## ⚡ التشغيل السريع
### 1. تشغيل العرض التوضيحي عبر الإنترنت
[![AI Studio](https://img.shields.io/badge/PP_OCRv5-AI_Studio-green)](https://aistudio.baidu.com/community/app/91660/webUI)
[![AI Studio](https://img.shields.io/badge/PP_StructureV3-AI_Studio-green)](https://aistudio.baidu.com/community/app/518494/webUI)
[![AI Studio](https://img.shields.io/badge/PP_ChatOCRv4-AI_Studio-green)](https://aistudio.baidu.com/community/app/518493/webUI)



### 2. التثبيت

قم بتثبيت PaddlePaddle بالرجوع إلى [دليل التثبيت](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)، وبعد ذلك، قم بتثبيت مجموعة أدوات PaddleOCR.


<div style="text-align: left;">

```bash
# إذا كنت تريد فقط استخدام ميزة التعرف الأساسي على النصوص (ترجع إحداثيات النص ومحتواه)، بما في ذلك سلسلة PP-OCR
python -m pip install paddleocr
# إذا كنت تريد استخدام جميع الميزات مثل تحليل المستندات، فهم المستندات، ترجمة المستندات، استخراج المعلومات الرئيسية، وما إلى ذلك
# python -m pip install "paddleocr[all]"
```

بدءًا من الإصدار 3.2.0، بالإضافة إلى مجموعة التبعيات `all` المذكورة أعلاه، تدعم PaddleOCR أيضًا تثبيت بعض الميزات الاختيارية جزئيًا عن طريق تحديد مجموعات تبعيات أخرى. جميع مجموعات التبعيات التي توفرها PaddleOCR موضحة في الجدول التالي:

| اسم مجموعة التبعيات   | الوظيفة المقابلة |
| -                     | -               |
| `doc-parser`          | تحليل المستندات: يمكن استخدامه لاستخراج عناصر التخطيط مثل الجداول، الصيغ، الأختام، الصور، وغيرها من المستندات؛ ويشمل نماذج مثل PP-StructureV3 وPaddleOCR-VL. |
| `ie`                  | استخراج المعلومات: يمكن استخدامها لاستخراج المعلومات الرئيسية مثل الأسماء، التواريخ، العناوين، المبالغ، إلخ من المستندات؛ تشمل نماذج مثل PP-ChatOCRv4 |
| `trans`               | ترجمة المستندات: يمكن استخدامها لترجمة المستندات من لغة إلى أخرى؛ تشمل نماذج مثل PP-DocTranslation |
| `all`                 | جميع الميزات الكاملة      |

</div>

### 3. تشغيل الاستدلال عبر واجهة سطر الأوامر (CLI)

<div style="text-align: left !important;">

```bash
# Run PP-OCRv5 inference
paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False  

# Run PP-StructureV3 inference
paddleocr pp_structurev3 -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png --use_doc_orientation_classify False --use_doc_unwarping False

# Get the Qianfan API Key at first, and then run PP-ChatOCRv4 inference
paddleocr pp_chatocrv4_doc -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/vehicle_certificate-1.png -k 驾驶室准乘人数 --qianfan_api_key your_api_key --use_doc_orientation_classify False --use_doc_unwarping False 

# Get more information about "paddleocr ocr"
paddleocr ocr --help
```

</div>

### 4. تشغيل الاستدلال عبر واجهة برمجة التطبيقات (API)

<details dir="ltr" open>
  <summary dir="rtl"><strong>4.1 مثال PP-OCRv5</strong></summary>

```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

result = ocr.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png")

for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")
```

</details>

<details>
   <summary dir="rtl"><strong>4.2 مثال PP-StructureV3</strong></summary>

```python
from pathlib import Path
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False
)

# للصور
output = pipeline.predict(
    input="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/pp_structure_v3_demo.png",
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
# إذا تم استخدام نموذج كبير متعدد الوسائط، فيجب بدء خدمة mllm المحلية. يمكنك الرجوع إلى الوثائق: https://github.com/PaddlePaddle/PaddleX/blob/release/3.0/docs/pipeline_usage/tutorials/vlm_pipelines/doc_understanding.en.md لتنفيذ النشر وتحديث تكوين mllm_chat_bot_config.
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


## مسرّعات الذكاء الاصطناعي الصينية غير المتجانسة
<ul dir="rtl">
  <li><a href="https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_NPU.html">Huawei Ascend</a></li>
  <li><a href="https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/other_devices_support/paddlepaddle_install_XPU.html">Kunlunxin</a></li>
</ul>

## ⛰️ دروس متقدمة
- [درس PP-OCRv5 التعليمي](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/OCR.html)
- [درس PP-StructureV3 التعليمي](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-StructureV3.html)
- [درس PP-ChatOCRv4 التعليمي](https://paddlepaddle.github.io/PaddleOCR/latest/version3.x/pipeline_usage/PP-ChatOCRv4.html)

## 🔄 نظرة سريعة على نتائج التنفيذ


<p align="center">
    <img width="100%" src="../docs/images/demo.gif" alt="PP-OCRv5 Demo">
</p>



<p align="center">
    <img width="100%" src="../docs/images/blue_v3.gif" alt="PP-StructureV3 Demo">
</p>

## 🌟 لا تفوت أحدث الأخبار

⭐ **ضع نجمة لهذا المستودع لتكون على اطلاع دائم بأحدث التحديثات والإصدارات الجديدة المثيرة، بما في ذلك ميزات التعرف الضوئي على الحروف (OCR) وتحليل المستندات القوية!** ⭐

<div align="center">
  <p>
       <img width="1200" src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/paddleocr/README/star_paddleocr.en.gif" alt="Star-Project">
  </p>
</div>

## 🧩 المزيد من الميزات

- تحويل النماذج إلى صيغة ONNX: [الحصول على نماذج ONNX](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/obtaining_onnx_models.html).
- تسريع الاستدلال باستخدام محركات مثل OpenVINO و ONNX Runtime و TensorRT، أو إجراء الاستدلال باستخدام نماذج صيغة ONNX: [الاستدلال عالي الأداء](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/high_performance_inference.html).
- تسريع الاستدلال باستخدام تعدد وحدات معالجة الرسوميات والعمليات المتعددة: [الاستدلال المتوازي للخطوط الإنتاجية](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/instructions/parallel_inference.html).
- دمج PaddleOCR في التطبيقات المكتوبة بلغات مثل ++C و#C وJava وغيرها: [الخدمة](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/deployment/serving.html).

## 👩‍👩‍👧‍👦 المجتمع

| حساب PaddlePaddle الرسمي على WeChat | انضم إلى مجموعة النقاش التقني |
| :---: | :---: |
| <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qrcode_for_paddlepaddle_official_account.jpg" width="150"> | <img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/README/qr_code_for_the_questionnaire.jpg" width="150"> |


## 😃 مشاريع رائعة تستخدم PaddleOCR
لم يكن PaddleOCR ليصل إلى ما هو عليه اليوم بدون مجتمعه المذهل! 💗 شكرًا جزيلاً لجميع شركائنا القدامى، والمتعاونين الجدد، وكل من صب شغفه في PaddleOCR - سواء ذكرنا اسمك أم لا. دعمكم يشعل نارنا!

| اسم المشروع | الوصف |
| ------------ | ----------- |
| [RAGFlow](https://github.com/infiniflow/ragflow) <a href="https://github.com/infiniflow/ragflow"><img src="https://img.shields.io/github/stars/infiniflow/ragflow"></a>|محرك RAG يعتمد على فهم عميق للوثائق.|
| [pathway](https://github.com/pathwaycom/pathway) <a href="https://github.com/pathwaycom/pathway"><img src="https://img.shields.io/github/stars/pathwaycom/pathway"></a>|إطار عمل ETL بلغة بايثون لمعالجة التدفق، التحليل الفوري، خطوط أنابيب LLM وRAG.|
| [MinerU](https://github.com/opendatalab/MinerU) <a href="https://github.com/opendatalab/MinerU"><img src="https://img.shields.io/github/stars/opendatalab/MinerU"></a>|أداة تحويل المستندات متعددة الأنواع إلى Markdown|
| [Umi-OCR](https://github.com/hiroi-sora/Umi-OCR) <a href="https://github.com/hiroi-sora/Umi-OCR"><img src="https://img.shields.io/github/stars/hiroi-sora/Umi-OCR"></a>|برنامج OCR مجاني ومفتوح المصدر للعمل دفعة واحدة دون اتصال بالإنترنت.|
| [cherry-studio](https://github.com/CherryHQ/cherry-studio) <a href="https://github.com/CherryHQ/cherry-studio"><img src="https://img.shields.io/github/stars/CherryHQ/cherry-studio"></a>|عميل سطح مكتب يدعم عدة مزودي LLM.|
| [OmniParser](https://github.com/microsoft/OmniParser)<a href="https://github.com/microsoft/OmniParser"><img src="https://img.shields.io/github/stars/microsoft/OmniParser"></a> |أداة OmniParser: أداة تحليل الشاشة لوكيل واجهة المستخدم الرسومية المستند إلى الرؤية البحتة.|
| [QAnything](https://github.com/netease-youdao/QAnything)<a href="https://github.com/netease-youdao/QAnything"><img src="https://img.shields.io/github/stars/netease-youdao/QAnything"></a> |نظام سؤال وجواب يعتمد على أي شيء.|
| [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) <a href="https://github.com/opendatalab/PDF-Extract-Kit"><img src="https://img.shields.io/github/stars/opendatalab/PDF-Extract-Kit"></a>|مجموعة أدوات قوية مفتوحة المصدر مصممة لاستخراج محتوى عالي الجودة بكفاءة من مستندات PDF المعقدة والمتنوعة.|
| [Dango-Translator](https://github.com/PantsuDango/Dango-Translator)<a href="https://github.com/PantsuDango/Dango-Translator"><img src="https://img.shields.io/github/stars/PantsuDango/Dango-Translator"></a> |يتعرف على النص على الشاشة، ويترجمه ويعرض نتائج الترجمة في الوقت الفعلي.|
| [تعرف على المزيد من المشاريع](../awesome_projects.md) | [مشاريع أخرى تعتمد على PaddleOCR](../awesome_projects.md)|

## 👩‍👩‍👧‍👦 المساهمون

<a href="https://github.com/PaddlePaddle/PaddleOCR/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleOCR&max=400&columns=20"  width="800"/>
</a>


## 🌟 نجمة

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleOCR&type=Date)](https://star-history.com/#PaddlePaddle/PaddleOCR&Date)


## 📄 الترخيص
هذا المشروع مرخص بموجب [ترخيص Apache 2.0](LICENSE).

</div>

## 🎓 الاستشهاد الأكاديمي

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
```
