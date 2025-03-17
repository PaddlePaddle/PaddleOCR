---
comments: true
hide:
  - navigation
  - toc
---

### 更新

#### **🔥🔥2025.3.7 PaddleOCR 2.10 版本，主要包含如下内容**：

  - **重磅新增 OCR 领域 12 个自研单模型：**
    - **[版面区域检测](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/layout_detection.html)** 系列 3 个模型：PP-DocLayout-L、PP-DocLayout-M、PP-DocLayout-S，支持预测 23 个常见版面类别，中英论文、研报、试卷、书籍、杂志、合同、报纸等丰富类型的文档实现高质量版面检测，**mAP@0.5 最高达 90.4%，轻量模型端到端每秒处理超百页文档图像。**
    - **[公式识别](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/formula_recognition.html)** 系列 2 个模型：PP-FormulaNet-L、PP-FormulaNet-S，支持 5 万种 LaTeX 常见词汇，支持识别高难度印刷公式和手写公式，其中 **PP-FormulaNet-L 较开源同等量级模型精度高 6 个百分点，PP-FormulaNet-S 较同等精度模型速度快 16 倍。**
    - **[表格结构识别](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/table_structure_recognition.html)** 系列 2 个模型：SLANeXt_wired、SLANeXt_wireless。飞桨自研新一代表格结构识别模型，分别支持有线表格和无线表格的结构预测。相比于SLANet_plus，SLANeXt在表格结构方面有较大提升，**在内部高难度表格识别评测集上精度高 6 个百分点。**
    - **[表格分类](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/table_classification.html)** 系列 1 个模型：PP-LCNet_x1_0_table_cls，超轻量级有线表格和无线表格的分类模型。
    - **[表格单元格检测](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/table_cells_detection.html)** 系列 2 个模型：RT-DETR-L_wired_table_cell_det、RT-DETR-L_wireless_table_cell_det，分别支持有线表格和无线表格的单元格检测，可配合SLANeXt_wired、SLANeXt_wireless、文本检测、文本识别模块完成对表格的端到端预测。（参见本次新增的表格识别v2产线）
    - **[文本识别](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html)** 系列 1 个模型： PP-OCRv4_server_rec_doc，**支持1.5万+字典，文字识别范围更广，与此同时提升了部分文字的识别精准度，在内部数据集上，精度较 PP-OCRv4_server_rec 高 3 个百分点以上。**
    - **[文本行方向分类](https://paddlepaddle.github.io/PaddleX/latest/module_usage/tutorials/ocr_modules/text_recognition.html)** 系列 1 个模型：PP-LCNet_x0_25_textline_ori，**存储只有 0.3M** 的超轻量级文本行方向分类模型。

   - **重磅推出 4 条高价值多模型组合方案：** 
     - **[文档图像预处理产线](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/doc_preprocessor.html)**：通过超轻量级模型组合使用，实现对文档图像的扭曲和方向的矫正。
     - **[版面解析v2产线](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/layout_parsing_v2.html)**：组合多个自研的不同类型的 OCR 类模型，优化复杂版面阅读顺序，实现多种复杂 PDF 文件端到端转换 Markdown 文件和 JSON 文件。在多个文档场景下，转换效果较其他开源方案更好。可以为大模型训练和应用提供高质量的数据生产能力。
     - **[表格识别v2产线](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/ocr_pipelines/table_recognition_v2.html)**：**提供更好的表格端到端识别能力。** 通过将表格分类模块、表格单元格检测模块、表格结构识别模块、文本检测模块、文本识别模块等组合使用，实现对多种样式的表格预测，用户可自定义微调其中任意模块以提升垂类表格的效果。
     - **[PP-ChatOCRv4-doc产线](https://paddlepaddle.github.io/PaddleX/latest/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction_v4.html)**：在 PP-ChatOCRv3-doc 的基础上，**融合了多模态大模型，优化了 Prompt 和多模型组合后处理逻辑，更好地解决了版面分析、生僻字、多页 pdf、表格、印章识别等常见的复杂文档信息抽取难点问题，准确率较 PP-ChatOCRv3-doc 高 15 个百分点。其中，大模型升级了本地部署的能力，提供了标准的 OpenAI 调用接口，支持对本地大模型如 DeepSeek-R1 部署的调用。**



#### **🔥2024.10.1 添加OCR领域低代码全流程开发能力**
  * 飞桨低代码开发工具PaddleX，依托于PaddleOCR的先进技术，支持了OCR领域的低代码全流程开发能力：
     * 🎨 [**模型丰富一键调用**](https://paddlepaddle.github.io/PaddleOCR/latest/paddlex/quick_start.html)：将文本图像智能分析、通用OCR、通用版面解析、通用表格识别、公式识别、印章文本识别涉及的**17个模型**整合为6条模型产线，通过极简的**Python API一键调用**，快速体验模型效果。此外，同一套API，也支持图像分类、目标检测、图像分割、时序预测等共计**200+模型**，形成20+单功能模块，方便开发者进行**模型组合**使用。
     * 🚀[**提高效率降低门槛**](https://paddlepaddle.github.io/PaddleOCR/latest/paddlex/overview.html)：提供基于**统一命令**和**图形界面**两种方式，实现模型简洁高效的使用、组合与定制。支持**高性能推理、服务化部署和端侧部署**等多种部署方式。此外，对于各种主流硬件如**英伟达GPU、昆仑芯、昇腾、寒武纪和海光**等，进行模型开发时，都可以**无缝切换**。

  * 支持文档场景信息抽取v3[PP-ChatOCRv3-doc](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/pipeline_usage/tutorials/information_extraction_pipelines/document_scene_information_extraction.md)、基于RT-DETR的[高精度版面区域检测模型](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/layout_detection.md)和PicoDet的[高效率版面区域检测模型](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/layout_detection.md)、高精度表格结构识别模型[SLANet_Plus](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/table_structure_recognition.md)、文本图像矫正模型[UVDoc](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/text_image_unwarping.md)、公式识别模型[LatexOCR](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/formula_recognition.md)、基于PP-LCNet的[文档图像方向分类模型](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/doc_img_orientation_classification.md)
  
#### 🔥 2024.7 添加 PaddleOCR 算法模型挑战赛冠军方案：
    - 赛题一：OCR 端到端识别任务冠军方案——[场景文本识别算法-SVTRv2](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/text_recognition/algorithm_rec_svtrv2.html)；
    - 赛题二：通用表格识别任务冠军方案——[表格识别算法-SLANet-LCNetV2](https://paddlepaddle.github.io/PaddleOCR/latest/algorithm/table_recognition/algorithm_table_slanet.html)。
    

#### **🔥2024.5.10 上线星河零代码产线(OCR 相关)**

全面覆盖了以下四大 OCR 核心任务，提供极便捷的 Badcase 分析和实用的在线体验

- [通用 OCR](https://aistudio.baidu.com/community/app/91660) (PP-OCRv4)。
- [通用表格识别](https://aistudio.baidu.com/community/app/91661) (SLANet)。
- [通用图像信息抽取](https://aistudio.baidu.com/community/app/91662) (PP-ChatOCRv2-common)。
- [文档场景信息抽取](https://aistudio.baidu.com/community/app/70303) (PP-ChatOCRv2-doc)。

  同时采用了 **[全新的场景任务开发范式](https://aistudio.baidu.com/pipeline/mine)** ,将模型统一汇聚，实现训练部署的零代码开发，并支持在线服务化部署和导出离线服务化部署包。

#### 🔥2023.8.7 发布 PaddleOCR [release/2.7](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.7)

- 发布[PP-OCRv4](./doc/doc_ch/PP-OCRv4_introduction.md)，提供 mobile 和 server 两种模型
    - PP-OCRv4-mobile：速度可比情况下，中文场景效果相比于 PP-OCRv3 再提升 4.5%，英文场景提升 10%，80 语种多语言模型平均识别准确率提升 8%以上
    - PP-OCRv4-server：发布了目前精度最高的 OCR 模型，中英文场景上检测模型精度提升 4.9%， 识别模型精度提升 2%
可参考[快速开始](./doc/doc_ch/quickstart.md) 一行命令快速使用，同时也可在飞桨 AI 套件(PaddleX)中的[通用 OCR 产业方案](https://aistudio.baidu.com/aistudio/modelsdetail?modelId=286)中低代码完成模型训练、推理、高性能部署全流程

#### 🔨**2022.11 新增实现[4 种前沿算法](doc/doc_ch/algorithm_overview.md)**：文本检测 [DRRG](doc/doc_ch/algorithm_det_drrg.md),  文本识别 [RFL](doc/doc_ch/algorithm_rec_rfl.md), 文本超分[Text Telescope](doc/doc_ch/algorithm_sr_telescope.md)，公式识别[CAN](doc/doc_ch/algorithm_rec_can.md)

#### **2022.10 优化[JS 版 PP-OCRv3 模型](./deploy/paddlejs/README_ch.md)**：模型大小仅 4.3M，预测速度提升 8 倍，配套 web demo 开箱即用

- **💥 直播回放：PaddleOCR 研发团队详解 PP-StructureV2 优化策略**。微信扫描[下方二维码](#开源社区)，关注公众号并填写问卷后进入官方交流群，获取直播回放链接与 20G 重磅 OCR 学习大礼包（内含 PDF 转 Word 应用程序、10 种垂类模型、《动手学 OCR》电子书等）

#### **🔥2022.8.24 发布 PaddleOCR [release/2.6](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.6)**

- 发布[PP-StructureV2](./ppstructure/README_ch.md)，系统功能性能全面升级，适配中文场景，新增支持[版面复原](./ppstructure/recovery/README_ch.md)，支持**一行命令完成 PDF 转 Word**；
- [版面分析](./ppstructure/layout/README_ch.md)模型优化：模型存储减少 95%，速度提升 11 倍，平均 CPU 耗时仅需 41ms；
- [表格识别](./ppstructure/table/README_ch.md)模型优化：设计 3 大优化策略，预测耗时不变情况下，模型精度提升 6%；
- [关键信息抽取](./ppstructure/kie/README_ch.md)模型优化：设计视觉无关模型结构，语义实体识别精度提升 2.8%，关系抽取精度提升 9.1%。

#### **2022.8 发布 [OCR 场景应用集合](./applications)**：包含数码管、液晶屏、车牌、高精度 SVTR 模型、手写体识别等**9 个垂类模型**，覆盖通用，制造、金融、交通行业的主要 OCR 垂类应用

#### 2022.5.9 发布PaddleOCR v2.5。发布内容包括

- [PP-OCRv3](./ppocr_introduction.md#pp-ocrv3)，速度可比情况下，中文场景效果相比于PP-OCRv2再提升5%，英文场景提升11%，80语种多语言模型平均识别准确率提升5%以上；
- 半自动标注工具[PPOCRLabelv2](https://github.com/PFCCLab/PPOCRLabel)：新增表格文字图像、图像关键信息抽取任务和不规则文字图像的标注功能；
- OCR产业落地工具集：打通22种训练部署软硬件环境与方式，覆盖企业90%的训练部署环境需求
- 交互式OCR开源电子书[《动手学OCR》](./ocr_book.md)，覆盖OCR全栈技术的前沿理论与代码实践，并配套教学视频。

#### 2022.5.7 添加对[Weights & Biases](https://docs.wandb.ai/)训练日志记录工具的支持

#### 2021.12.21 《OCR十讲》课程开讲，12月21日起每晚八点半线上授课！ 【免费】报名地址：<https://aistudio.baidu.com/aistudio/course/introduce/25207>

#### 2021.12.21 发布PaddleOCR v2.4。OCR算法新增1种文本检测算法（PSENet），3种文本识别算法（NRTR、SEED、SAR）；文档结构化算法新增1种关键信息提取算法（SDMGR），3种DocVQA算法（LayoutLM、LayoutLMv2，LayoutXLM）

#### 2021.9.7 发布PaddleOCR v2.3，发布[PP-OCRv2](#PP-OCRv2)，CPU推理速度相比于PP-OCR server提升220%；效果相比于PP-OCR mobile 提升7%

#### 2021.8.3 发布PaddleOCR v2.2，新增文档结构分析[PP-Structure](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/ppstructure/README_ch.md)工具包，支持版面分析与表格识别（含Excel导出）

#### 2021.6.29 [FAQ](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/doc/doc_ch/FAQ.md)新增5个高频问题，总数248个，每周一都会更新，欢迎大家持续关注

#### 2021.4.8 release 2.1版本，新增AAAI 2021论文[端到端识别算法PGNet](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/doc/doc_ch/pgnet.md)开源，[多语言模型](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.2/doc/doc_ch/multi_languages.md)支持种类增加到80+

#### 2020.12.15 更新数据合成工具[Style-Text](https://github.com/PFCCLab/StyleText/blob/main/README_ch.md)，可以批量合成大量与目标场景类似的图像，在多个场景验证，效果明显提升

#### 2020.12.07 [FAQ](../../doc/doc_ch/FAQ.md)新增5个高频问题，总数124个，并且计划以后每周一都会更新，欢迎大家持续关注

#### 2020.11.25 更新半自动标注工具[PPOCRLabel](https://github.com/PFCCLab/PPOCRLabel/blob/main/README_ch.md)，辅助开发者高效完成标注任务，输出格式与PP-OCR训练任务完美衔接

#### 2020.9.22 更新PP-OCR技术文章，<https://arxiv.org/abs/2009.09941>

#### 2020.9.19 更新超轻量压缩ppocr_mobile_slim系列模型，整体模型3.5M(详见PP-OCR Pipeline)，适合在移动端部署使用

#### 2020.9.17 更新超轻量ppocr_mobile系列和通用ppocr_server系列中英文ocr模型，媲美商业效果

#### 2020.9.17 更新[英文识别模型](./models_list.md#english-recognition-model)和[多语种识别模型](./models_list.md#english-recognition-model)，已支持`德语、法语、日语、韩语`，更多语种识别模型将持续更新

#### 2020.8.26 更新OCR相关的84个常见问题及解答，具体参考[FAQ](./FAQ.md)

#### 2020.8.24 支持通过whl包安装使用PaddleOCR，具体参考[Paddleocr Package使用说明](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/doc/doc_ch/whl.md)

#### 2020.8.21 更新8月18日B站直播课回放和PPT，课节2，易学易用的OCR工具大礼包，[获取地址](https://aistudio.baidu.com/aistudio/education/group/info/1519)

#### 2020.8.16 开源文本检测算法[SAST](https://arxiv.org/abs/1908.05498)和文本识别算法[SRN](https://arxiv.org/abs/2003.12294)

#### 2020.7.23 发布7月21日B站直播课回放和PPT，课节1，PaddleOCR开源大礼包全面解读，[获取地址](https://aistudio.baidu.com/aistudio/course/introduce/1519)

#### 2020.7.15 添加基于EasyEdge和Paddle-Lite的移动端DEMO，支持iOS和Android系统

#### 2020.7.15 完善预测部署，添加基于C++预测引擎推理、服务化部署和端侧部署方案，以及超轻量级中文OCR模型预测耗时Benchmark

#### 2020.7.15 整理OCR相关数据集、常用数据标注以及合成工具

#### 2020.7.9 添加支持空格的识别模型，识别效果，预测及训练方式请参考快速开始和文本识别训练相关文档

#### 2020.7.9 添加数据增强、学习率衰减策略,具体参考[配置文件](./config.md)

#### 2020.6.8 添加[数据集](dataset/datasets.md)，并保持持续更新

#### 2020.6.5 支持 `attetnion` 模型导出 `inference_model`

#### 2020.6.5 支持单独预测识别时，输出结果得分

#### 2020.5.30 提供超轻量级中文OCR在线体验

#### 2020.5.30 模型预测、训练支持Windows系统

#### 2020.5.30 开源通用中文OCR模型

#### 2020.5.14 发布[PaddleOCR公开课](https://www.bilibili.com/video/BV1nf4y1U7RX?p=4)

#### 2020.5.14 发布[PaddleOCR实战练习](https://aistudio.baidu.com/aistudio/projectdetail/467229)

#### 2020.5.14 开源8.6M超轻量级中文OCR模型
