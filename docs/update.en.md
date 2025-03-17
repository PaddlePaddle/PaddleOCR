---
comments: true
hide:
  - navigation
  - toc
---

### Recently Update

#### **🔥🔥2025.3.7 release PaddleOCR v2.10, including**:

  - **12 new self-developed single models:**
    - **[Layout Detection](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)** series with 3 models: PP-DocLayout-L, PP-DocLayout-M, PP-DocLayout-S, supporting prediction of 23 common layout categories. High-quality layout detection for various document types such as papers, reports, exams, books, magazines, contracts, newspapers in both English and Chinese. **mAP@0.5 reaches up to 90.4%, lightweight models can process over 100 pages of document images per second end-to-end.**
    - **[Formula Recognition](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/formula_recognition.html)** series with 2 models: PP-FormulaNet-L, PP-FormulaNet-S, supporting 50,000 common LaTeX vocabulary, capable of recognizing complex printed and handwritten formulas. **PP-FormulaNet-L has 6 percentage points higher accuracy than models of the same level, and PP-FormulaNet-S is 16 times faster than models with similar accuracy.**
    - **[Table Structure Recognition](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_structure_recognition.html)** series with 2 models: SLANeXt_wired, SLANeXt_wireless. A newly developed table structure recognition model, supporting structured prediction for both wired and wireless tables. Compared to SLANet_plus, SLANeXt shows significant improvement in table structure, **with 6 percentage points higher accuracy on internal high-difficulty table recognition evaluation sets.**
    - **[Table Classification](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_classification.html)** series with 1 model: PP-LCNet_x1_0_table_cls, an ultra-lightweight classification model for both wired and wireless tables.
    - **[Table Cell Detection](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/table_cells_detection.html)** series with 2 models: RT-DETR-L_wired_table_cell_det, RT-DETR-L_wireless_table_cell_det, supporting cell detection in both wired and wireless tables. These can be combined with SLANeXt_wired, SLANeXt_wireless, text detection, and text recognition modules for end-to-end table prediction. (See the newly added Table Recognition v2 pipeline)
    - **[Text Recognition](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/text_recognition.html)** series with 1 model: PP-OCRv4_server_rec_doc, **supports over 15,000 characters, with a broader text recognition range, additionally improving the recognition accuracy of certain texts. The accuracy is more than 3 percentage points higher than PP-OCRv4_server_rec on internal datasets.**
    - **[Text Line Orientation Classification](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html)** series with 1 model: PP-LCNet_x0_25_textline_ori, **an ultra-lightweight text line orientation classification model with only 0.3M storage.**

   - **4 high-value multi-model combination solutions:**
     - **[Document Image Preprocessing Pipeline](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html)**: Achieve correction of distortion and orientation in document images through the combination of ultra-lightweight models.
     - **[Layout Parsing v2 Pipeline](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/layout_parsing_v2.html)**: Combines multiple self-developed different types of OCR models to optimize complex layout reading order, achieving end-to-end conversion of various complex PDF files to Markdown and JSON files. The conversion effect is better than other open-source solutions in multiple document scenarios. It can provide high-quality data production capabilities for large model training and application.
     - **[Table Recognition v2 Pipeline](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html)**: **Provides better table recognition capabilities.** By combining table classification module, table cell detection module, table structure recognition module, text detection module, text recognition module, etc., it achieves prediction of various styles of tables. Users can customize and finetune any module to improve the effect of vertical tables.
     - **[PP-ChatOCRv4-doc Pipeline](https://paddlepaddle.github.io/PaddleX/latest/en/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.html)**: Based on PP-ChatOCRv3-doc, **integrating multi-modal large models, optimizing Prompt and multi-model combination post-processing logic. It effectively addresses common complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition, achieving 15 percentage points higher accuracy than PP-ChatOCRv3-doc. The large model upgrades local deployment capabilities, providing a standard OpenAI interface, supporting calls to locally deployed large models like DeepSeek-R1.**

#### **🔥 2024.10.18 release PaddleOCR v2.9, including**:
  * PaddleX, an All-in-One development tool based on PaddleOCR's advanced technology, supports low-code full-process development capabilities in the OCR field:
     * 🎨 [**Rich Model One-Click Call**](https://paddlepaddle.github.io/PaddleOCR/latest/en/paddlex/quick_start.html): Integrates **17 models** related to text image intelligent analysis, general OCR, general layout parsing, table recognition, formula recognition, and seal recognition into 6 pipelines, which can be quickly experienced through a simple **Python API one-click call**. In addition, the same set of APIs also supports a total of **200+ models** in image classification, object detection, image segmentation, and time series forecasting, forming 20+ single-function modules, making it convenient for developers to use **model combinations**.

     * 🚀 [**High Efficiency and Low barrier of entry**](https://paddlepaddle.github.io/PaddleOCR/latest/en/paddlex/overview.html): Provides two methods based on **unified commands** and **GUI** to achieve simple and efficient use, combination, and customization of models. Supports multiple deployment methods such as **high-performance inference, service-oriented deployment, and edge deployment**. Additionally, for various mainstream hardware such as **NVIDIA GPU, Kunlunxin XPU, Ascend NPU, Cambricon MLU, and Haiguang DCU**, models can be developed with **seamless switching**.

  * Supports [PP-ChatOCRv3-doc](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_en.md), [high-precision layout detection model based on RT-DETR](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/layout_detection_en.md) and [high-efficiency layout area detection model based on PicoDet](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/layout_detection_en.md), [high-precision table structure recognition model](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/table_structure_recognition_en.md), text image unwarping model [UVDoc](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/text_image_unwarping_en.md), formula recognition model [LatexOCR](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/formula_recognition_en.md), and [document image orientation classification model based on PP-LCNet](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/doc_img_orientation_classification_en.md).
  
#### 2022.5.9 release PaddleOCR v2.5, including

- [PP-OCRv3](./ppocr_introduction_en.md#pp-ocrv3): With comparable speed, the effect of Chinese scene is further improved by 5% compared with PP-OCRv2, the effect of English scene is improved by 11%, and the average recognition accuracy of 80 language multilingual models is improved by more than 5%.
- [PPOCRLabelv2](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md): Add the annotation function for table recognition task, key information extraction task and irregular text image.
- Interactive e-book [*"Dive into OCR"*](./ocr_book_en.md), covers the cutting-edge theory and code practice of OCR full stack technology.

#### 2022.5.7 Add support for metric and model logging during training to [Weights & Biases](https://docs.wandb.ai/)

#### 2021.12.21 OCR open source online course starts. The lesson starts at 8:30 every night and lasts for ten days. Free registration: <https://aistudio.baidu.com/aistudio/course/introduce/25207>

#### 2021.12.21 release PaddleOCR v2.4, release 1 text detection algorithm (PSENet), 3 text recognition algorithms (NRTR、SEED、SAR), 1 key information extraction algorithm (SDMGR) and 3 DocVQA algorithms (LayoutLM、LayoutLMv2，LayoutXLM)

#### 2021.9.7 release PaddleOCR v2.3, [PP-OCRv2](#PP-OCRv2) is proposed. The CPU inference speed of PP-OCRv2 is 220% higher than that of PP-OCR server. The F-score of PP-OCRv2 is 7% higher than that of PP-OCR mobile

#### 2021.8.3 released PaddleOCR v2.2, add a new structured documents analysis toolkit, i.e., [PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/ppstructure/README.md), support layout analysis and table recognition (One-key to export chart images to Excel files)

#### 2021.4.8 release end-to-end text recognition algorithm [PGNet](https://www.aaai.org/AAAI21Papers/AAAI-2885.WangP.pdf) which is published in AAAI 2021. Find tutorial [here](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/pgnet_en.md)；release multi language recognition [models](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/multi_languages_en.md), support more than 80 languages recognition; especially, the performance of [English recognition model](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/doc/doc_en/models_list_en.md#English) is Optimized

#### 2021.1.21 update more than 25+ multilingual recognition models [models list](./models_list_en.md), including：English, Chinese, German, French, Japanese，Spanish，Portuguese Russia Arabic and so on.  Models for more languages will continue to be updated [Develop Plan](https://github.com/PaddlePaddle/PaddleOCR/issues/1048)

#### 2020.12.15 update Data synthesis tool, i.e., [Style-Text](https://github.com/PFCCLab/StyleText/blob/main/README.md)，easy to synthesize a large number of images which are similar to the target scene image

#### 2020.11.25 Update a new data annotation tool, i.e., [PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel/blob/main/README.md), which is helpful to improve the labeling efficiency. Moreover, the labeling results can be used in training of the PP-OCR system directly

#### 2020.9.22 Update the PP-OCR technical article, <https://arxiv.org/abs/2009.09941>

#### 2020.9.19 Update the ultra lightweight compressed ppocr_mobile_slim series models, the overall model size is 3.5M, suitable for mobile deployment

#### 2020.9.17 update English recognition model and Multilingual recognition model, `English`, `Chinese`, `German`, `French`, `Japanese` and `Korean` have been supported. Models for more languages will continue to be updated

#### 2020.8.24 Support the use of PaddleOCR through whl package installation，please refer  [PaddleOCR Package](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_en/whl_en.md)

#### 2020.8.16 Release text detection algorithm [SAST](https://arxiv.org/abs/1908.05498) and text recognition algorithm [SRN](https://arxiv.org/abs/2003.12294)

#### 2020.7.23, Release the playback and PPT of live class on BiliBili station, PaddleOCR Introduction, [address](https://aistudio.baidu.com/aistudio/course/introduce/1519)

#### 2020.7.15, Add mobile App demo , support both iOS and  Android  (based on easyedge and Paddle Lite)

#### 2020.7.15, Improve the  deployment ability, add the C + +  inference , serving deployment. In addition, the benchmarks of the ultra-lightweight Chinese OCR model are provided

#### 2020.7.15, Add several related datasets, data annotation and synthesis tools

#### 2020.7.9 Add a new model to support recognize the  character "space"

#### 2020.7.9 Add the data argument and learning rate decay strategies during training

#### 2020.6.8 Add [datasets](dataset/datasets_en.md) and keep updating

#### 2020.6.5 Support exporting `attention` model to `inference_model`

#### 2020.6.5 Support separate prediction and recognition, output result score

#### 2020.5.30 Provide Lightweight Chinese OCR online experience

#### 2020.5.30 Model prediction and training support on Windows system

#### 2020.5.30 Open source general Chinese OCR model

#### 2020.5.14 Release [PaddleOCR Open Class](https://www.bilibili.com/video/BV1nf4y1U7RX?p=4)

#### 2020.5.14 Release [PaddleOCR Practice Notebook](https://aistudio.baidu.com/aistudio/projectdetail/467229)

#### 2020.5.14 Open source 8.6M lightweight Chinese OCR model
