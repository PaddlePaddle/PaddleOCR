English | [简体中文](README_ch.md) | [हिन्दी](./doc/doc_i18n/README_हिन्द.md) | [日本語](./doc/doc_i18n/README_日本語.md) | [한국인](./doc/doc_i18n/README_한국어.md) | [Pу́сский язы́к](./doc/doc_i18n/README_Ру́сский_язы́к.md)

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

## مقدمه

هدف PaddleOCR ایجاد ابزارهای OCR چندزبانه، عالی، پیشرو و کاربردی است که به کاربران کمک می کند مدل های بهتری را آموزش دهند و آنها را در عمل به کار گیرند.

<div align="center">
    <img src="./doc/imgs_results/PP-OCRv3/en/en_4.png" width="800">
</div>

<div align="center">
    <img src="./doc/imgs_results/ch_ppocr_mobile_v2.0/00006737.jpg" width="800">
</div>


## 📣 به روز رسانی های اخیر

در تاریخ 2022.11 اجرای [4 cutting-edge algorithms](doc/doc_ch/algorithm_overview.md) ：تشخیص متن [DRRG](doc/doc_en/algorithm_det_drrg_en.md) ،شناسایی متن [RFL](./doc/doc_en/algorithm_rec_rfl_en.md) ،   وضوح تصویر فوق‌العاده  [[Text Telescope](doc/doc_en/algorithm_sr_telescope_en.md) ؛ تشخیص عبارات ریاضی دست‌نویس [CAN](doc/doc_en/algorithm_rec_can_en.md) 

نسخه 2022.10 [بهینه نسخه JS مدل PP-OCRv3](./deploy/paddlejs/README.md)** با اندازه مدل 4.3M، زمان استنتاج 8 برابر سریعتر، و نسخه نمایشی وب آماده برای استفاده

 پخش زنده: مقدمه ای بر استراتژی بهینه سازی PP-StructureV2**. [کد QR زیر] (#Community) را با استفاده از WeChat اسکن کنید، حساب رسمی PaddlePaddle را دنبال کنید و پرسشنامه را پر کنید تا به گروه WeChat بپیوندید، پیوند زنده و مواد آموزشی 20G OCR (از جمله برنامه PDF2Word، 10 مدل در سناریوهای عمودی را دریافت کنید، و غیره.)



   
 انتشار [PP-StructureV2](./ppstructure/)، با عملکردها و عملکرد کاملاً ارتقا یافته، سازگار با صحنه های چینی، و پشتیبانی جدید از [بازیابی طرح بندی] (./ppstructure/recovery) و **فرمان یک خطی برای تبدیل PDF به Word **;
   
بهینه سازی [تجزیه و تحلیل طرح بندی](./ppstructure/layout): فضای ذخیره سازی مدل تا 95٪ کاهش یافته است، در حالی که سرعت 11 برابر افزایش یافته است، و میانگین هزینه زمان CPU تنها 41 میلی ثانیه است.
  
 بهینه سازی [تشخیص جدول](./ppstructure/table): 3 استراتژی بهینه سازی طراحی شده است و دقت مدل با مصرف زمان قابل مقایسه تا 6 درصد بهبود می یابد.
  
 [استخراج اطلاعات کلیدی] (./ppstructure/kie) بهینه سازی: یک ساختار مدل مستقل از بصری طراحی شده است، دقت تشخیص موجودیت معنایی 2.8٪ افزایش می یابد و دقت استخراج رابطه 9.1٪ افزایش می یابد.

 در تاریخ 2022.8انتشار [مجموعه برنامه صحنه OCR](./applications/README_en.md)**
   
 انتشار **9 مدل عمودی ** مانند لوله دیجیتال، صفحه نمایش ال سی دی، پلاک، مدل تشخیص دست خط، مدل SVTR با دقت بالا و غیره که کاربردهای اصلی OCR عمودی را به طور کلی، صنایع تولید، مالی و حمل و نقل پوشش می دهد.

 در تاریخ 2022.8 پیاده سازی [8 cutting-edge algorithms](doc/doc_en/algorithm_overview_en.md)  را اضافه کنید
  
تشخیص متن: [FCENet](doc/doc_en/algorithm_det_fcenet_en.md)، [DB++](doc/doc_en/algorithm_det_db_en.md)
  
تشخیص متن: [ViTSTR](doc/doc_en/algorithm_rec_vitstr_en.md)، [ABINet](doc/doc_en/algorithm_rec_abinet_en.md)، [VisionLAN](doc/doc_en/algorithm_rec_visionlan_en.md) /algorithm_rec_spin_en.md)، [RobustScanner](doc/doc_en/algorithm_rec_robustscanner_en.md)
  
تشخیص جدول: [TableMaster](doc/doc_en/algorithm_table_master_en.md)

ریلیز PaddleOCR   در تاریخ 2022.5.9     [release/2.5](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.5) 
     

انتشار [PP-OCRv3](./doc/doc_en/ppocr_introduction_en.md#pp-ocrv3)  با سرعت قابل مقایسه، جلوه صحنه چینی 5 درصد در مقایسه با PP-OCRv2 بهبود می یابد، جلوه صحنه 
انگلیسی 11% بهبود یافته و میانگین دقت تشخیص 80 مدل چند زبانه زبان بیش از 5% بهبود یافته است.

انتشار [PPOCRLabelv2](./PPOCRLabel) : تابع حاشیه نویسی را برای کار تشخیص جدول، کار استخراج اطلاعات کلید و تصویر متن نامنظم اضافه کنید.    

انتشار کتاب الکترونیکی تعاملی [*"Dive into OCR"*](./doc/doc_en/ocr_book_en.md)  تئوری پیشرفته و  کد های عملی  فناوری OCR تمام stack  را پوشش می دهد.

 [بیشتر](./doc/doc_en/update_en.md)

## 🌟 ویژگی ها

فریم ورک PaddleOCR از انواع الگوریتم‌های پیشرفته مرتبط با OCR پشتیبانی می‌کند و مدل‌ها/راه‌حل‌های ویژه صنعتی [PP-OCR](./doc/doc_en/ppocr_introduction_en.md) و [PP-Structure] (./ppstructure/README.md) را توسعه داده است. ) بر این اساس، و کل فرآیند تولید داده، آموزش مدل، فشرده سازی، استنتاج و استقرار را طی کنید.

<div align="center">
     <img src="https://user-images.githubusercontent.com/25809855/186171245-40abc4d7-904f-4949-ade1-250f86ed3a90.png">
</div>




> توصیه می شود با "تجربه سریع" در آموزش سند شروع کنید.
> 

## ⚡ تجربه سریع

تجربه آنلاین وب برای OCR بسیار سبک یا  [Online Experience](https://www.paddlepaddle.org.cn/hub/scene/ocr)  

تجربه DEMO موبایل (بر اساس EasyEdge و Paddle-Lite، از سیستم‌های iOS و Android پشتیبانی می‌کند): [برای دریافت کد QR برای نصب برنامه، وارد وب‌سایت شوید] (https://ai.baidu.com/easyedge/app /openSource?from=paddlelite)

استفاده از یک خط سریع :  [Quick Start](./doc/doc_en/quickstart_en.md)

<a name="book"></a>
## 📚 کتاب الکترونیکی :شیرجه به OCR

[Dive Into OCR ](./doc/doc_en/ocr_book_en.md)

<a name="Community"></a>

## 👫 انجمن

برای توسعه دهندگان بین المللی، ما [PaddleOCR Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions)  را به عنوان پلت فرم جامعه بین المللی خود در نظر می گیریم. همه ایده ها و سوالات را می توان در اینجا به زبان انگلیسی مورد بحث قرار داد.

برای برنامه های چینی، کد QR زیر را با Wechat خود اسکن کنید، می توانید به گروه بحث فنی رسمی بپیوندید. برای محتوای غنی‌تر انجمن، لطفاً به [中文README](README_ch.md) مراجعه کنید، منتظر مشارکت شما هستیم.


<div align="center">

<img src="https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/dygraph/doc/joinus.PNG"  width = "150" height = "150" />
</div>

<a name="Supported-Chinese-model-list"></a>

## 🛠️ لیست مدل های سری PP-OCR (به روز رسانی در 8 سپتامبر)


| Model introduction                                           | Model name                   | Recommended scene | Detection model                                              | Direction classifier                                         | Recognition model                                            |
| ------------------------------------------------------------ | ---------------------------- | ----------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Chinese and English ultra-lightweight PP-OCRv3 model（16.2M）     | ch_PP-OCRv3_xx          | Mobile & Server | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar) |
| English ultra-lightweight PP-OCRv3 model（13.4M）     | en_PP-OCRv3_xx          | Mobile & Server | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) | [inference model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar) |
| Chinese and English ultra-lightweight PP-OCRv2 model（11.6M） |  ch_PP-OCRv2_xx |Mobile & Server|[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_distill_train.tar)| [inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |[inference model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_train.tar)|
| Chinese and English ultra-lightweight PP-OCR model (9.4M)       | ch_ppocr_mobile_v2.0_xx      | Mobile & server   |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_train.tar)|[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar) |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_train.tar)      |
| Chinese and English general PP-OCR model (143.4M)               | ch_ppocr_server_v2.0_xx      | Server            |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_train.tar)    |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar)    |[inference model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) / [trained model](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_train.tar)  |

- برای دانلود مدل های بیشتر (از جمله چندین زبان)، لطفاً به [PP-OCR series model downloads](./doc/doc_en/models_list_en.md) مراجعه کنید .
- برای درخواست زبان جدید، لطفاً به راهنما برای درخواست زبان های جدید [Guideline for new language_requests](#language_requests)  مراجعه کنید.
- برای مدل‌های تحلیل سند ساختاری، لطفاً به [PP-Structure models](./ppstructure/docs/models_list_en.md) مراجعه کنید.

## 📖 آموزش
- [آماده سازی محیط] (./doc/doc_en/environment_en.md)
- [PP-OCR 🔥](./doc/doc_en/ppocr_introduction_en.md)
     - [شروع سریع] (./doc/doc_en/quickstart_en.md)
     - [مدل باغ وحش](./doc/doc_en/models_en.md)
     - [آموزش مدل](./doc/doc_en/training_en.md)
         - [تشخیص متن](./doc/doc_en/detection_en.md)
         - [تشخیص متن](./doc/doc_en/recognition_en.md)
         - [طبقه بندی جهت متن] (./doc/doc_en/angle_class_en.md)
     - فشرده سازی مدل
         - [کوانتیزاسیون مدل](./deploy/slim/quantization/README_en.md)
         - [مدل هرس](./deploy/slim/prune/README_en.md)
         - [تقطیر دانش](./doc/doc_en/knowledge_distillation_en.md)
     - [استنتاج و استقرار] (./deploy/README.md)
         - [استنتاج پایتون](./doc/doc_en/inference_ppocr_en.md)
         - [C++ Inference] (./deploy/cpp_infer/readme.md)
         - [خدمت] (./deploy/pdserving/README.md)
         - [موبایل](./deploy/lite/readme.md)
         - [Paddle2ONNX](./deploy/paddle2onnx/readme.md)
         - [PaddleCloud] (./deploy/paddlecloud/README.md)
         - [معیار] (./doc/doc_en/benchmark_en.md)
- [PP-Structure 🔥](./ppstructure/README.md)
     - [شروع سریع] (./ppstructure/docs/quickstart_en.md)
     - [Model Zoo](./ppstructure/docs/models_list_en.md)
     - [آموزش مدل](./doc/doc_en/training_en.md)
         - [تحلیل طرح‌بندی] (./ppstructure/layout/README.md)
         - [تشخیص جدول] (./ppstructure/table/README.md)
         - [استخراج اطلاعات کلیدی] (./ppstructure/kie/README.md)
     - [استنتاج و استقرار] (./deploy/README.md)
         - [استنتاج پایتون](./ppstructure/docs/inference_en.md)
         - [C++ Inference] (./deploy/cpp_infer/readme.md)
         - [در حال ارائه] (./deploy/hubserving/readme_en.md)
- [الگوریتم‌های آکادمیک] (./doc/doc_en/algorithm_overview_en.md)
     - [تشخیص متن] (./doc/doc_en/algorithm_overview_en.md)
     - [تشخیص متن] (./doc/doc_en/algorithm_overview_en.md)
     - [OCR انتها به انتها](./doc/doc_en/algorithm_overview_en.md)
     - [تشخیص جدول] (./doc/doc_en/algorithm_overview_en.md)
     - [ استخراج اطلاعات کلیدی] (./doc/doc_en/algorithm_overview_en.md)
     - [افزودن الگوریتم‌های جدید به PaddleOCR](./doc/doc_en/add_new_algorithm_en.md)
- حاشیه نویسی و ترکیب داده ها
     - [ابزار حاشیه نویسی نیمه خودکار: PPOCRLabel] (./PPOCRLabel/README.md)
     - [ابزار ترکیب داده: Style-Text] (./StyleText/README.md)
     - [سایر ابزارهای حاشیه نویسی داده] (./doc/doc_en/data_annotation_en.md)
     - [سایر ابزارهای سنتز داده] (./doc/doc_en/data_synthesis_en.md)
- مجموعه داده ها
     - [General OCR Datasets (چینی/انگلیسی)](doc/doc_en/dataset/datasets_en.md)
     - [HandWritten_OCR_Datasets(چینی)](doc/doc_en/dataset/handwritten_datasets_en.md)
     - [مجموعه‌های مختلف OCR (چند زبانه)] (doc/doc_en/dataset/vertical_and_multilingual_datasets_en.md)
     - [تحلیل طرح‌بندی](doc/doc_en/dataset/layout_datasets_en.md)
     - [تشخیص جدول] (doc/doc_en/dataset/table_datasets_en.md)
     - [استخراج اطلاعات کلیدی] (doc/doc_en/dataset/kie_datasets_en.md)
- [ساختار کد] (./doc/doc_en/tree_en.md)
- [تجسم] (#تجسم)
- [Community] (#Community)
- [درخواست‌های زبان جدید] (#زبان_درخواست‌ها)
- [سؤالات متداول](./doc/doc_en/FAQ_en.md)
- [مرجع] (./doc/doc_en/reference_en.md)
- [مجوز] (#LICENSE)

<a name="Visualization"></a>

## 👀 Visualization [more](./doc/doc_en/visualization_en.md)

<details open>
<summary>PP-OCRv3 Chinese model</summary>
<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic001.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic002.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/ch/PP-OCRv3-pic003.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-OCRv3 English model</summary>
<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/en/en_1.png" width="800">
    <img src="doc/imgs_results/PP-OCRv3/en/en_2.png" width="800">
</div>
</details>

<details open>
<summary>PP-OCRv3 Multilingual model</summary>
<div align="center">
    <img src="doc/imgs_results/PP-OCRv3/multi_lang/japan_2.jpg" width="800">
    <img src="doc/imgs_results/PP-OCRv3/multi_lang/korean_1.jpg" width="800">
</div>
</details>

<details open>
<summary>PP-StructureV2</summary>

- layout analysis + table recognition  
<div align="center">
    <img src="./ppstructure/docs/table/ppstructure.GIF" width="800">
</div>

- SER (Semantic entity recognition)
<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/197464552-69de557f-edff-4c7f-acbf-069df1ba097f.png" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185310636-6ce02f7c-790d-479f-b163-ea97a5a04808.jpg" width="600">
</div>

<div align="center">
    <img src="https://user-images.githubusercontent.com/14270174/185539517-ccf2372a-f026-4a7c-ad28-c741c770f60a.png" width="600">
</div>

- RE (Relation Extraction)
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
<a name="language_requests"></a>

##  راهنمای درخواست های زبان جدید

اگر می خواهید یک پشتیبانی زبان جدید درخواست کنید، یک PR با 1 فایل زیر مورد نیاز است:

1. در پوشه [ppocr/utils/dict](./ppocr/utils/dict)،
لازم است متن dict را به این مسیر ارسال کنید و آن را با "{language}_dict.txt" نامگذاری کنید که حاوی لیستی از همه کاراکترها است. لطفاً نمونه قالب را از سایر فایل‌های موجود در آن پوشه ببینید.

اگر زبان شما دارای عناصر منحصربه‌فردی است، لطفاً از قبل به هر طریقی مانند پیوندهای مفید، ویکی‌پدیا و غیره به من بگویید.

جزئیات بیشتر، لطفاً به [طرح توسعه OCR چند زبانه] (https://github.com/PaddlePaddle/PaddleOCR/issues/1048) مراجعه کنید.


<a name="LICENSE"></a>
## 📄 مجوز
این پروژه تحت <a href="https://github.com/PaddlePaddle/PaddleOCR/blob/master/LICENSE">مجوز Apache 2.0</a> منتشر شده است

