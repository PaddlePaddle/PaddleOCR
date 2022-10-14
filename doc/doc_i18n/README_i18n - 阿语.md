English| [简体中文](README_ch.md)

<p align="center">
 <img src="./doc/PaddleOCR_log.png" align="middle" width = "600"/>
<p align="center">
<p align="left">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleOCR?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/pypi/format/PaddleOCR?color=c77"></a>
    <a href="https://pypi.org/project/PaddleOCR/"><img src="https://img.shields.io/pypi/dm/PaddleOCR?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleOCR?color=ccf"></a>
</p>


## المقدمة

يهدف PaddleOCR إلى إنشاء أدوات OCR متعددة اللغات، رائعة، رائدة وعملية تساعد المستخدمين على التدرب على أفضل النماذج وتطبيقها في الحياة العملية.


<div align="center">
    <img src="https://user-images.githubusercontent.com/50011306/187821591-6cb09459-fdbf-4ad3-8c5a-26af611c211d.png" width="800">
</div>

<div align="center">
    <img src="./doc/imgs_results/PP-OCRv3/en/en_4.png" width="800">
</div>


<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/00006737.jpg" width="800">
</div>


## 📣 آخر التحديثات
- **🔥- **2022.8.24 الإصدار PaddleOCR [الإصدار/2.6](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)**
  -إصدار [PP-Structurev2](./ppstructure/)مع الوظائف والأداء الذي تمت تحسينه بالكامل، وتكييفه مع الجانب الصيني، ودعم جديد لـ [استعادة التخطيط(./ppstructure/recovery)[و ** أمر سطر واحد لتحويل PDF إلى .**Word 


] - تحليل التخطيط (./ppstructure/layout)[التحسين: تم  تقليل تخزين النموذج بنسبة 95%، بينما زادت السرعة بمقدار 11 مرة، ويبلغ متوسط تكلفة وقت وحدة المعالجة المركزية 41 مللي ثانية فقط.
 ]- التعرف على الجدول (./ppstructure/table)[التحسين: تم تصميم 3 استراتيجيات للتحسين، وتم تحسين دقة النموذج بنسبة 6% في ظل استهلاك الوقت المماثل .

 ] - استخراج المعلومات الأساسية (./ppstructure/kie)[التحسين - تم تصميم هيكل نموذج بصري مستقل، وزيادة دقة التعرف على الكيانات الدلالية بنسبة 2.8%، وزيادة دقة استخراج العلاقة بنسبة %9.1


-- ** 🔥  **2022.7 الإصدار [مجموعة تطبيقات OCR المشهد] (./applications/README_en.md)
  -الإصدار ** 9 نماذج رأسية ** مثل الأنبوب الرقمي وشاشة LCD ولوحة الترخيص ونموذج التعرف على خط اليد ونموذج SVTR عالي الدقة وما إلى ذلك ،OCR والتي تغطي التطبيقات الرأسية للتعرف المرئي على الحروف بشكل عام والتصنيع والتمويل والنقل.
2022.5.9 إصدار PaddleOCR [إصدار/2.5] 
- **- ** [الإصدار/2.5](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5)**

  -إصدار: [PP-OCRv3](./doc/doc_en/ppocr_introduction_en.md#pp-ocrv3) مع سرعة مماثلة، تم تحسين تأثير الجانب الصيني بنسبة 5% مقارنة بـ PP-OCRv2 ، وتم تحسين تأثير المشهد الإنجليزي بنسبة 11% وتم تحسين متوسط دقة التعرف على 80 نموذجًا متعدد اللغات بأكثر من %5.

  -إصدار: [PPOCRLabelv2](./PPOCRLabel) أضف وظيفة التعليق التوضيحي لمهمة التعرف على الجدول، مهمة استخراج المعلومات الأساسية وصورة نصية غير منتظمة.

    -إصدار كتاب إلكتروني تفاعلي [* "OCRالتعمق في التعرف المرئي على الحروف [* " (./doc/doc_en/ocr_book_en.md), يغطي أحدث النظريات وممارسات الكود لتقنية التعرف المرئي على الحروف الكاملة.

]- المزيد (./doc/doc_en/update_en.md)[


## 🌟 سمات  

يدعم PaddleOCR مجموعة متنوعة من الخوارزميات المتطورة المتعلقة باOCR لتعرف المرئي على الحروف،
 ونماذج/ حلول صناعية مميزة متطورة[PP-OCR](./doc/doc_en/ppocr_introduction_en.md)      و [PP-Structure](./ppstructure/README.md) 
على هذا الأساس، واجتياز العملية الكاملة لإنتاج البيانات، تدريب النموذج، الضغط، الاستدلال ، والنشر.

<div align="center">
    <img src="https://user-images.githubusercontent.com/50011306/195772420-1ff9fc48-5bf3-4715-98da-375d96f584d7.png">
</div>


## ⚡ تجربة سريعة


```bash
pip3 install paddlepaddle # for gpu user please install paddlepaddle-gpu
pip3 install paddleocr
paddleocr --image_dir ./doc/imgs_en/254.jpg --lang=en # change for i18n abbr
```



< إذا لم يكن لديك بيئة بايثون، فيرجى اتباع [تحضير البيئة (./doc/doc_en/environment_en.md).[
نوصيك بالبدء بـ [دروس تعليمية (#Tutorials)[


<a name="كتاب"></a>


##  📚 الكتاب الإلكتروني: * الغوص في OCR                                         
- التعمق في التعرف المرئي على الحروف [OCR](./doc/doc_en/ocr_book_en.md)


<a name="المجتمع"></a>


## 👫 المجتمع

للمطورين الدوليين، نحن نحترم [مناقشات [Paddle OCR (https://github.com/PaddlePaddle/PaddleOCR/discussions)  كمنصة للمجتمع الدولي.  يمكن مناقشة جميع الأفكار والأسئلة هنا باللغة الإنجليزية.


<a name="قائمة النموذج الصيني المدعومة"></a>


## 🛠️ PP-OCR سلسلة قائمة النماذج

مقدمة نموذجية | | اسم الموديل | المشهد الموصى بها | نموذج الكشف | مصنف الاتجاه || نموذج الاعتراف ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ || نماذج i18n | نموذج I18n | | المحمول والخادم  |  |  || الإنجليزية خفيفة الوزن للغاية PP-OCRv3 نموذج (13.4M) | en_PP-OCRv3_xx | | المحمول والخادم [نموذج الاستدلال] (https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar)  / [نموذج مدرب] (https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) |  [نموذج الاستدلال] https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar  / [نموذج مدرب] https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar |  [نموذج الاستدلال] (https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar)  / [نموذج مدرب] (https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) | 
| الصينية والإنجليزية خفيفة الوزن للغاية PP-OCRv3 نموذج (16.2M) | ch_PP-OCRv3_xx | | المحمول والخادم  [نموذج الاستدلال] (https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)  / [نموذج مدرب] (https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) |  [نموذج الاستدلال] https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar  / [نموذج مدرب] https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar |  [نموذج الاستدلال] (https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar)  / [نموذج مدرب] (https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar |



- لمزيد من تنزيلات النماذج (بما في ذلك لغات متعددة) ، يرجى الرجوع إلى [تنزيلات نموذج سلسلة PP-OCR]  (./doc/doc_en/models_list_en.md).
- للحصول على طلب لغة جديد، يرجى الرجوع إلى [مبادئ توجيهية لطلبات_اللغات الجديدة(#language_requests).
 - بالنسبة لنماذج تحليل المستندات الهيكلية، يرجى الرجوع إلى[PP-Structure] models](./ppstructure/docs/models_list_en.md).


<a name="التعليمية"></a>


## 📖 الدروس

- تحضيرالبيئة [PP-OCR](./doc/doc_en/environment_en.md)[ 

 ] - بداية سريعة(./doc/doc_en/quickstart_en.md)[ 

] - نموذج حديقة الحيوان(./doc/doc_en/models_en.md)[ 

] - تدريب نموذجي(./doc/doc_en/training_en.md)[ 

 ] - كشف النص[(./doc/doc_en/detection_en.md)[ 

 ] - التعرف على النص(./doc/doc_en/recognition_en.md)[ 

 ] - تصنيف اتجاه النص(./doc/doc_en/angle_class_en.md)[

    -ضغط النموذج

] - نموذج توضيحي(./deploy/slim/quantization/README_en.md)]
  - نموذج التهذيب(./deploy/slim/prune/README_en.md)] 

 ] - تقطيرالمعرفة(./doc/doc_en/knowledge_distillation_en.md)[
] - الاستدلال والنشر(./deploy/README.md)[

 ] - استدلال بايثون(./doc/doc_en/inference_ppocr_en.md)[

  ] - استدلال(./deploy/cpp_infer/readme.md)[ C++

 ] -خدمة  (./deploy/pdserving/README.md)[

  ] - هاتف محمول(./deploy/lite/readme.md)[

        - [Paddle2ONNX](./deploy/paddle2onnx/readme.md)
    
        - [PaddleCloud](./deploy/paddlecloud/README.md)
        - [Benchmark](./doc/doc_en/benchmark_en.md)  
- [PP-Structure 


 ] - بداية سريعة(./ppstructure/docs/quickstart_en.md)[ 

 ] - نموذج حديقة الحيوان(./ppstructure/docs/models_list_en.md)[

 ] - تدريب نموذجي(./doc/doc_en/training_en.md)[ 

 ] - تحليل التخطيط(./ppstructure/layout/README.md)[

 ] - التعرف على الجدول(./ppstructure/table/README.md)[

 ] - استخراج المعلومات الأساسية(./ppstructure/kie/README.md)[

 ] - الاستدلال والنشر(./deploy/README.md)[

 ] - استدلال بايثون(./ppstructure/docs/inference_en.md)[

 ] - استدلال(./deploy/cpp_infer/readme.md)[ C++ 

 ] - خدمة(./deploy/hubserving/readme_en.md)[

] - الخوارزميات الأكاديمية(./doc/doc_en/algorithm_overview_en.md)[

 ] - كشف النص(./doc/doc_en/algorithm_overview_en.md)[

 ] - التعرف على النص(./doc/doc_en/algorithm_overview_en.md)[

 ] - التعرف المرئي على الحروف من طرف إلى طرف[
(./doc/doc_en/algorithm_overview_en.md) 

] - التعرف على الجدول(./doc/doc_en/algorithm_overview_en.md)[

] - استخراج المعلومات الأساسية(./doc/doc_en/algorithm_overview_en.md)    [

] - أضف خوارزميات جديدة إلى [PaddleOCR (./doc/doc_en/add_new_algorithm_en.md)

-شرح البيانات والتأليف

] - أداة التعليقات التوضيحية شبه الآلية: [PPOCRLabel  (./PPOCRLabel/README.md)

] - أداة تجميع البيانات: نمط النص(./StyleText/README.md)[

] - أدوات شرح البيانات الأخرى(./doc/doc_en/data_annotation_en.md)[
]    - أدوات تجميع البيانات الأخرى(./doc/doc_en/data_synthesis_en.md)[
-مجموعات البيانات

 ] - مجموعات بيانات التعرف البصري على الحروف العامة OCR(الصينية/ الإنجليزية) [
(doc/doc_en/dataset/datasets_en.md)

] - مجموعات البيانات المكتوبة بخط اليد _التعرف البصري على الحروف OCR_ (الصينية(
(doc/doc_en/dataset/handwritten_datasets_en.md)

 ] - مجموعات بيانات التعرف البصري على الحروف المختلفة OCR (متعددة اللغات [(
(doc/doc_en/dataset/vertical_and_multilingual_datasets_en.md)

 ] - تحليل التخطيط(doc/doc_en/dataset/layout_datasets_en.md)[

] - التعرف على الجدول(doc/doc_en/dataset/table_datasets_en.md)[

] - استخراج المعلومات الأساسية(doc/doc_en/dataset/kie_datasets_en.md)[

]- بنيةالكود(./doc/doc_en/tree_en.md)[
]- المرئيات(#Visualization)[
]- المجتمع(#Community)[
 ]- طلبات لغة جديدة(#language_requests)[
]-  الأسئلة الشائعة(./doc/doc_en/FAQ_en.md)[
- ]المراجع(./doc/doc_en/reference_en.md)[ 
]- الترخيص(#LICENSE)[


<a name="language_requests"></a>


 un ## إرشادات لطلبات اللغات الجديدة

إذا كنت ترغب في **طلب نموذج لغة جديد**، يرجى التصويت في [التصويت لصالح ترقيات النماذج متعددة اللغات] (https://github.com/PaddlePaddle/PaddleOCR/discussions/7253).  سنقوم بترقية النموذج وفقا للنتيجة بانتظام.  ** قم بدعوة أصدقائك للتصويت معا!** 

إذا كنت بحاجة إلى  ** تدريب نموذج لغة جديد ** بناء على السيناريو الخاص بك ، فإن البرنامج التعليمي في [مشروع تدريب نموذجي متعدد اللغات] (https://github.com/PaddlePaddle/PaddleOCR/discussions/7252) سيساعدك على إعداد مجموعة البيانات ويوضح لك العملية برمتها خطوة بخطوة . 

 [خطة تطويرOCR التعرف الضوئي على الحروف متعددة اللغات] الأصلية (https://github.com/PaddlePaddle/PaddleOCR/issues/1048) تعرض لك الكثير من المتن والقواميس المفيدة





<a name="المرئيات"></a>


## 👀 التصور [المزيد] (./doc/doc_en/visualization_en.md)

<details open>
<summary> 3 النموذج متعدد اللغات</summary> PP-OCRv
<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/multi_lang/japan_2.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/multi_lang/korean_1.jpg" width="800">
</div>
</details>

<details open>
<summary> النموذج  الإنجليزي</summary> PP-OCRv3
<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/en/en_1.png" width="800">
    <img src="doc/imgs_results/PP-OCRv3/en/en_2.png" width="800">
</div>
</details>
<details open>
<summary> النموذج  الصيني</summary> PP-OCRv3
<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic001.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic002.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic003.jpg" width="800">
</div>
</details>


<details open>
<summary>PP-Structurev2</summary>
1.تحليل التخطيط + التعرف على الجدول
<div align="center">
    <img src="./ppstructure/docs/table/ppstructure.GIF" width="800">
</div>

2. SER التعرف على الكيان المتعلق بدلالات الألفاظ 
<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094456-01a1dd11-1433-4437-9ab2-6480ac94ec0a.png" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808.jpg" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a.png" width="600">
</div>

 RE.3 استخراج العلاقة
<div align="center">
    <img src="https://user-images.githubusercontent.com/25809855/186094813-3a8e16cc-42e5-4982-b9f4-0134dfb5688d.png" width="600">
</div>   
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185393805-c67ff571-cf7e-4217-a4b0-8b396c4f22bb.jpg" width="600">
</div>
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185540080-0431e006-9235-4b6d-b63d-0b3c6e1de48f.jpg" width="600">
</div>
</details>

<a name="ترخيص"></a>


## 📄 الترخيص 
تم تحرير هذا المشروع تحت <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">Apache 2.0 license</a>
