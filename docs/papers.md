---
comments: true
---

# PaddleOCR 论文展示

从 2020 年的轻量文字识别，到面向复杂文档的结构化解析与视觉语言模型，本页梳理 PaddleOCR 六年发展中的论文与技术脉络。

**收录范围与核对日期：** 截至 2026-09-14，收录已核实的 PaddleOCR 系列论文、百度参与且在 PaddleOCR 提供实现的相关算法研究，并注明合作研究与飞桨生态关联工作的来源。仅使用 PaddleOCR 的下游应用论文不在此范围内；更多第三方复现算法见[算法总览](version2.x/algorithm/overview.md)。日期表示论文首次公开时间，不等同于模型发布日期；预印本与后续版本不重复计数。

## 阅读导览

先读总体报告，再按研究方向查阅。每个方向内优先保持同一系列连续，并按论文首次公开时间从早到晚排列；相关算法单独成组。

| 研究方向 | 推荐阅读顺序 |
| --- | --- |
| [总体报告](#research-overview) | [PaddleOCR 3.0](#paper-paddleocr-3-0) |
| [文字识别](#research-ocr) | [PP-OCR](#paper-pp-ocr) → [PP-OCRv2](#paper-pp-ocrv2) → [PP-OCRv3](#paper-pp-ocrv3) → [PP-OCRv5](#paper-pp-ocrv5) → [PP-OCRv6](#paper-pp-ocrv6)；[SVTR](#paper-svtr) → [CPPD](#paper-cppd) |
| [文档结构化与版面分析](#research-layout) | [PP-StructureV2](#paper-pp-structurev2) → [PP-DocLayout](#paper-pp-doclayout) → [RT-DocLayout / PP-DocLayoutV3](#paper-rt-doclayout) |
| [公式识别](#research-formula) | [PP-FormulaNet](#paper-pp-formulanet) |
| [视觉语言文档解析](#research-parsing) | [PaddleOCR-VL](#paper-paddleocr-vl) → [PaddleOCR-VL-1.5](#paper-paddleocr-vl-1-5) → [PaddleOCR-VL-1.6](#paper-paddleocr-vl-1-6) → [HPD-Parsing](#paper-hpd-parsing) |
| [评测基准](#research-benchmarks) | [Real5-OmniDocBench](#paper-real5-omnidocbench) |
| [文档理解与生态工作](#research-understanding) | [PP-DocBee](#paper-pp-docbee) |

## 如何引用

README 展示 PaddleOCR 3.0 和 PaddleOCR-VL 两篇总体工作的引用。若使用了具体模型或算法，请引用下方对应论文；每条记录均提供可复制的 BibTeX。

<a id="research-overview"></a>

## 总体报告

先了解 PaddleOCR 的整体能力和技术体系。

<a id="paper-paddleocr-3-0"></a>

### PaddleOCR 3.0

**[PaddleOCR 3.0 Technical Report](https://arxiv.org/abs/2507.05595)**

**首次公开:** 2025-07-08 · arXiv:2507.05595

**作者:** Cheng Cui; Ting Sun; Manhui Lin; Tingquan Gao; Yubo Zhang; Jiaxuan Liu; Xueqing Wang; Zelun Zhang; Changda Zhou; Hongen Liu; Yue Zhang; Wenyu Lv; Kui Huang; Yichao Zhang; Jing Zhang; Jun Zhang; Yi Liu; Dianhai Yu; Yanjun Ma

统一介绍 PP-OCRv5 多语种文字识别、PP-StructureV3 层次化文档解析和 PP-ChatOCRv4 关键信息抽取。除模型外，报告还覆盖训练、推理、部署与异构硬件支持，展示 PaddleOCR 从文字识别工具向文档智能工具链的演进。

[论文](https://arxiv.org/abs/2507.05595) · [PDF](https://arxiv.org/pdf/2507.05595) · [相关文档](index.md)

<details>
<summary>引用 BibTeX</summary>

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
```

</details>

<a id="research-ocr"></a>

## 文字识别

先看 PP-OCR 系列的系统演进，再看支撑识别能力的算法研究。

### PP-OCR 系列

<a id="paper-pp-ocr"></a>

#### PP-OCR

**[PP-OCR: A Practical Ultra Lightweight OCR System](https://arxiv.org/abs/2009.09941)**

**首次公开:** 2020-09-21 · arXiv:2009.09941

**作者:** Yuning Du; Chenxia Li; Ruoyu Guo; Xiaoting Yin; Weiwei Liu; Jun Zhou; Yifan Bai; Zilin Yu; Yehua Yang; Qingqing Dang; Haoshuang Wang

将文本检测、方向分类和文本识别组成可部署的超轻量 OCR 系统，通过骨干网络选择、数据增强、训练策略、剪枝和量化兼顾精度与模型体积。它奠定了 PP-OCR 系列面向中英文及多语种应用的工程基础。

[论文](https://arxiv.org/abs/2009.09941) · [PDF](https://arxiv.org/pdf/2009.09941) · [相关文档](version2.x/ppocr/overview.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{ppocr2020,
  title={PP-OCR: A Practical Ultra Lightweight OCR System},
  author={Du, Yuning and Li, Chenxia and Guo, Ruoyu and Yin, Xiaoting and Liu, Weiwei and Zhou, Jun and Bai, Yifan and Yu, Zilin and Yang, Yehua and Dang, Qingqing and Wang, Haoshuang},
  year={2020},
  eprint={2009.09941},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2009.09941},
}
```

</details>

<a id="paper-pp-ocrv2"></a>

#### PP-OCRv2

**[PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System](https://arxiv.org/abs/2109.03144)**

**首次公开:** 2021-09-07 · arXiv:2109.03144

**作者:** Yuning Du; Chenxia Li; Ruoyu Guo; Cheng Cui; Weiwei Liu; Jun Zhou; Bin Lu; Yehua Yang; Qiwen Liu; Xiaoguang Hu; Dianhai Yu; Yanjun Ma

在 PP-OCR 基础上引入协同互学习、CopyPaste 数据增强、PP-LCNet、统一深度互学习和改进的 CTC 损失等策略。工作重点是在保持轻量和高效部署的同时提高检测与识别精度。

[论文](https://arxiv.org/abs/2109.03144) · [PDF](https://arxiv.org/pdf/2109.03144) · [相关文档](version2.x/ppocr/overview.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{ppocrv22021,
  title={PP-OCRv2: Bag of Tricks for Ultra Lightweight OCR System},
  author={Du, Yuning and Li, Chenxia and Guo, Ruoyu and Cui, Cheng and Liu, Weiwei and Zhou, Jun and Lu, Bin and Yang, Yehua and Liu, Qiwen and Hu, Xiaoguang and Yu, Dianhai and Ma, Yanjun},
  year={2021},
  eprint={2109.03144},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2109.03144},
}
```

</details>

<a id="paper-pp-ocrv3"></a>

#### PP-OCRv3

**[PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System](https://arxiv.org/abs/2206.03001)**

**首次公开:** 2022-06-07 · arXiv:2206.03001

**作者:** Chenxia Li; Weiwei Liu; Ruoyu Guo; Xiaoting Yin; Kaitao Jiang; Yongkun Du; Yuning Du; Lingfeng Zhu; Baohua Lai; Xiaoguang Hu; Dianhai Yu; Yanjun Ma

从检测和识别两个环节提出九项优化，包括 LK-PAN、RSE-FPN、蒸馏以及 SVTR_LCNet。结合注意力引导的 CTC 训练、数据增强与预训练策略，进一步改善速度与精度的平衡，是轻量 OCR 向视觉 Transformer 演进的重要节点。

[论文](https://arxiv.org/abs/2206.03001) · [PDF](https://arxiv.org/pdf/2206.03001) · [相关文档](version2.x/ppocr/blog/PP-OCRv3_introduction.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{ppocrv32022,
  title={PP-OCRv3: More Attempts for the Improvement of Ultra Lightweight OCR System},
  author={Li, Chenxia and Liu, Weiwei and Guo, Ruoyu and Yin, Xiaoting and Jiang, Kaitao and Du, Yongkun and Du, Yuning and Zhu, Lingfeng and Lai, Baohua and Hu, Xiaoguang and Yu, Dianhai and Ma, Yanjun},
  year={2022},
  eprint={2206.03001},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2206.03001},
}
```

</details>

<a id="paper-pp-ocrv5"></a>

#### PP-OCRv5

**[PP-OCRv5: A Specialized 5M-Parameter Model Rivaling Billion-Parameter Vision-Language Models on OCR Tasks](https://arxiv.org/abs/2603.24373)**

**首次公开:** 2026-03-25 · arXiv:2603.24373

**作者:** Cheng Cui; Yubo Zhang; Ting Sun; Xueqing Wang; Hongen Liu; Manhui Lin; Yue Zhang; Tingquan Gao; Changda Zhou; Jiaxuan Liu; Zelun Zhang; Jing Zhang; Jun Zhang; Yi Liu

研究轻量专用 OCR 模型如何在文字识别任务上与大规模视觉语言模型竞争。围绕数据难度、标注准确性和数据多样性进行实验，展示约 5M 参数系统的潜力。这是 2026 年公开的独立论文，区别于 2025 年发布的 PP-OCRv5 模型及 PaddleOCR 3.0 总体报告。

[论文](https://arxiv.org/abs/2603.24373) · [PDF](https://arxiv.org/pdf/2603.24373) · [相关文档](version3.x/algorithm/PP-OCRv5/PP-OCRv5.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{ppocrv52026,
  title={PP-OCRv5: A Specialized 5M-Parameter Model Rivaling Billion-Parameter Vision-Language Models on OCR Tasks},
  author={Cui, Cheng and Zhang, Yubo and Sun, Ting and Wang, Xueqing and Liu, Hongen and Lin, Manhui and Zhang, Yue and Gao, Tingquan and Zhou, Changda and Liu, Jiaxuan and Zhang, Zelun and Zhang, Jing and Zhang, Jun and Liu, Yi},
  year={2026},
  eprint={2603.24373},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.24373},
}
```

</details>

<a id="paper-pp-ocrv6"></a>

#### PP-OCRv6

**[PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks](https://arxiv.org/abs/2606.13108)**

**首次公开:** 2026-06-11 · arXiv:2606.13108

**作者:** Yubo Zhang; Xueqing Wang; Manhui Lin; Yue Zhang; Penglongyi Deng; Ting Sun; Tingquan Gao; Zelun Zhang; Jiaxuan Liu; Changda Zhou; Hongen Liu; Suyin Liang; Cheng Cui; Yi Liu; Dianhai Yu; Yanjun Ma

以统一的 MetaFormer 风格模块和结构重参数化重构骨干网络、检测颈部与识别颈部，并结合数据优化提升专用 OCR 的效率和精度。提供 tiny、small、medium 三档模型，参数规模从 1.5M 到 34.5M，覆盖端侧到服务端部署；通过任务相关的步长配置兼顾文本检测与识别。

[论文](https://arxiv.org/abs/2606.13108) · [PDF](https://arxiv.org/pdf/2606.13108) · [相关文档](version3.x/algorithm/PP-OCRv6/PP-OCRv6.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{ppocrv62026,
  title={PP-OCRv6: From 1.5M to 34.5M Parameters, Surpassing Billion-Scale VLMs on OCR Tasks},
  author={Yubo Zhang and Xueqing Wang and Manhui Lin and Yue Zhang and Penglongyi Deng and Ting Sun and Tingquan Gao and Zelun Zhang and Jiaxuan Liu and Changda Zhou and Hongen Liu and Suyin Liang and Cheng Cui and Yi Liu and Dianhai Yu and Yanjun Ma},
  year={2026},
  eprint={2606.13108},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2606.13108},
}
```

</details>

### 识别算法

<a id="paper-svtr"></a>

#### SVTR

**[SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)**

**首次公开:** 2022-04-30 · arXiv:2205.00159

**作者:** Yongkun Du; Zhineng Chen; Caiyan Jia; Xiaoting Yin; Tianlun Zheng; Chenxia Li; Yuning Du; Yu-Gang Jiang

通过局部与全局混合模块建模字符内部及字符之间的视觉关系，使用单一视觉模型和简单预测层完成场景文字识别。该合作研究为 PP-OCRv3 的 SVTR_LCNet 提供基础方法，PaddleOCR 提供公开实现。

[论文](https://arxiv.org/abs/2205.00159) · [PDF](https://arxiv.org/pdf/2205.00159) · [相关文档](version2.x/algorithm/text_recognition/algorithm_rec_svtr.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{svtr2022,
  title={SVTR: Scene Text Recognition with a Single Visual Model},
  author={Du, Yongkun and Chen, Zhineng and Jia, Caiyan and Yin, Xiaoting and Zheng, Tianlun and Li, Chenxia and Du, Yuning and Jiang, Yu-Gang},
  year={2022},
  eprint={2205.00159},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2205.00159},
}
```

</details>

<a id="paper-cppd"></a>

#### CPPD

**[Context Perception Parallel Decoder for Scene Text Recognition](https://arxiv.org/abs/2307.12270)**

**首次公开:** 2023-07-23 · arXiv:2307.12270

**作者:** Yongkun Du; Zhineng Chen; Caiyan Jia; Xiaoting Yin; Chenxia Li; Yuning Du; Yu-Gang Jiang

通过字符计数和字符排序任务显式建模识别上下文，再以并行解码预测字符序列。该合作研究旨在保留上下文建模能力的同时提高解码效率，PaddleOCR 提供训练配置与复现文档。

[论文](https://arxiv.org/abs/2307.12270) · [PDF](https://arxiv.org/pdf/2307.12270) · [相关文档](version2.x/algorithm/text_recognition/algorithm_rec_cppd.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{cppd2023,
  title={Context Perception Parallel Decoder for Scene Text Recognition},
  author={Du, Yongkun and Chen, Zhineng and Jia, Caiyan and Yin, Xiaoting and Li, Chenxia and Du, Yuning and Jiang, Yu-Gang},
  year={2023},
  eprint={2307.12270},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2307.12270},
}
```

</details>

<a id="research-layout"></a>

## 文档结构化与版面分析

从文档分析系统到联合预测文档元素与阅读顺序的版面模型。

<a id="paper-pp-structurev2"></a>

### PP-StructureV2

**[PP-StructureV2: A Stronger Document Analysis System](https://arxiv.org/abs/2210.05391)**

**首次公开:** 2022-10-11 · arXiv:2210.05391

**作者:** Chenxia Li; Ruoyu Guo; Jun Zhou; Mengtao An; Yuning Du; Lingfeng Zhu; Yi Liu; Xiaoguang Hu; Dianhai Yu

将文档分析组织为版面信息抽取和关键信息抽取两类任务，并加入图像方向矫正与版面恢复。报告介绍轻量版面检测、SLANet 表格识别和 VI-LayoutXLM，打通从文档图像到结构化内容的处理流程。

[论文](https://arxiv.org/abs/2210.05391) · [PDF](https://arxiv.org/pdf/2210.05391) · [相关文档](version2.x/ppstructure/overview.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{ppstructurev22022,
  title={PP-StructureV2: A Stronger Document Analysis System},
  author={Li, Chenxia and Guo, Ruoyu and Zhou, Jun and An, Mengtao and Du, Yuning and Zhu, Lingfeng and Liu, Yi and Hu, Xiaoguang and Yu, Dianhai},
  year={2022},
  eprint={2210.05391},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2210.05391},
}
```

</details>

<a id="paper-pp-doclayout"></a>

### PP-DocLayout

**[PP-DocLayout: A Unified Document Layout Detection Model to Accelerate Large-Scale Data Construction](https://arxiv.org/abs/2503.17213)**

**首次公开:** 2025-03-21 · arXiv:2503.17213

**作者:** Ting Sun; Cheng Cui; Yuning Du; Yi Liu

面向多类型文档的版面区域检测，识别标题、正文、表格、公式等 23 类元素。提供 L、M、S 三档模型，分别侧重精度、精度与速度平衡以及资源受限场景，支持高效的大规模文档处理与训练数据构建。论文代码和模型发布于 PaddleX，PaddleOCR 提供版面区域检测模块。

[论文](https://arxiv.org/abs/2503.17213) · [PDF](https://arxiv.org/pdf/2503.17213) · [相关文档](version3.x/module_usage/layout_detection.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{sun2025ppdoclayoutunifieddocumentlayout,
  title={PP-DocLayout: A Unified Document Layout Detection Model to Accelerate Large-Scale Data Construction},
  author={Ting Sun and Cheng Cui and Yuning Du and Yi Liu},
  year={2025},
  eprint={2503.17213},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.17213},
}
```

</details>

<a id="paper-rt-doclayout"></a>

### RT-DocLayout / PP-DocLayoutV3

**[RT-DocLayout: Real-Time End-to-End Document Layout Analysis with Reading Order in the Wild](https://arxiv.org/abs/2606.23344)**

**首次公开:** 2026-06-22 · arXiv:2606.23344

**作者:** Cheng Cui; Tingquan Gao; Xueqing Wang; Changda Zhou; Hongen Liu; Ting Sun; Yubo Zhang; Zelun Zhang; Jiaxuan Liu; Manhui Lin; Yue Zhang; Suyin Liang; Yiqing Xiang; Yi Liu

基于 RT-DETR，在单个查询式解码器中统一文档元素分类、边界框检测、像素级分割与阅读顺序预测。通过联合学习几何形状和结构关系，改善弯折及透视变化等真实畸变下的版面分析，并为下游 OCR 与整页文档重建提供高效前端。

**版本对应：** [PP-DocLayoutV3 官方模型页](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)引用本篇 RT-DocLayout 论文。初代 PP-DocLayout 的 L/M/S 模型对应[上方 2025 年论文](#paper-pp-doclayout)，两篇工作分别收录。

[论文](https://arxiv.org/abs/2606.23344) · [PDF](https://arxiv.org/pdf/2606.23344) · [相关文档](version3.x/module_usage/layout_analysis.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{rtdoclayout2026,
  title={RT-DocLayout: Real-Time End-to-End Document Layout Analysis with Reading Order in the Wild},
  author={Cheng Cui and Tingquan Gao and Xueqing Wang and Changda Zhou and Hongen Liu and Ting Sun and Yubo Zhang and Zelun Zhang and Jiaxuan Liu and Manhui Lin and Yue Zhang and Suyin Liang and Yiqing Xiang and Yi Liu},
  year={2026},
  eprint={2606.23344},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2606.23344},
}
```

</details>

<a id="research-formula"></a>

## 公式识别

将文档中的公式图像转换为可处理的符号表达。

<a id="paper-pp-formulanet"></a>

### PP-FormulaNet

**[PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition](https://arxiv.org/abs/2503.18382)**

**首次公开:** 2025-03-24 · arXiv:2503.18382

**作者:** Hongen Liu; Cheng Cui; Yuning Du; Yi Liu; Gang Pan

面向公式图像到 LaTeX 的转换，提供侧重精度的 L 模型和侧重效率的 S 模型。通过公式数据挖掘系统扩充高质量训练数据，使公式识别能够适应不同算力预算与复杂文档处理需求。

[论文](https://arxiv.org/abs/2503.18382) · [PDF](https://arxiv.org/pdf/2503.18382) · [相关文档](version3.x/module_usage/formula_recognition.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{ppformulanet2025,
  title={PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition},
  author={Liu, Hongen and Cui, Cheng and Du, Yuning and Liu, Yi and Pan, Gang},
  year={2025},
  eprint={2503.18382},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.18382},
}
```

</details>

<a id="research-parsing"></a>

## 视觉语言文档解析

先看 PaddleOCR-VL 系列的连续迭代，再看层级并行解码方向的 HPD-Parsing。

<a id="paper-paddleocr-vl"></a>

### PaddleOCR-VL

**[PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model](https://arxiv.org/abs/2510.14528)**

**首次公开:** 2025-10-16 · arXiv:2510.14528

**作者:** Cheng Cui; Ting Sun; Suyin Liang; Tingquan Gao; Zelun Zhang; Jiaxuan Liu; Xueqing Wang; Changda Zhou; Hongen Liu; Manhui Lin; Yue Zhang; Yubo Zhang; Handong Zheng; Jing Zhang; Jun Zhang; Yi Liu; Dianhai Yu; Yanjun Ma

采用 NaViT 风格的动态分辨率视觉编码器与 ERNIE-4.5-0.3B 语言模型构建 0.9B 文档解析模型。支持 109 种语言以及文本、表格、公式和图表等元素，探索小规模视觉语言模型在复杂文档解析中的效率与能力。

[论文](https://arxiv.org/abs/2510.14528) · [PDF](https://arxiv.org/pdf/2510.14528) · [相关文档](version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
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

</details>

<a id="paper-paddleocr-vl-1-5"></a>

### PaddleOCR-VL-1.5

**[PaddleOCR-VL-1.5: Towards a Multi-Task 0.9B VLM for Robust In-the-Wild Document Parsing](https://arxiv.org/abs/2601.21957)**

**首次公开:** 2026-01-29 · arXiv:2601.21957

**作者:** Cheng Cui; Ting Sun; Suyin Liang; Tingquan Gao; Zelun Zhang; Jiaxuan Liu; Xueqing Wang; Changda Zhou; Hongen Liu; Manhui Lin; Yue Zhang; Yubo Zhang; Yi Liu; Dianhai Yu; Yanjun Ma

在维持 0.9B 规模的同时加强真实场景文档解析，并加入印章识别与文本检测识别任务。提出 Real5-OmniDocBench，系统评估扫描、倾斜、弯折、屏幕拍摄和光照变化带来的影响，将研究重点扩展到实际采集条件下的鲁棒性。

[论文](https://arxiv.org/abs/2601.21957) · [PDF](https://arxiv.org/pdf/2601.21957) · [相关文档](version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.5.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
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

</details>

<a id="paper-paddleocr-vl-1-6"></a>

### PaddleOCR-VL-1.6

**[PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training](https://arxiv.org/abs/2606.03264)**

**首次公开:** 2026-06-02 · arXiv:2606.03264

**作者:** Zelun Zhang; Hongen Liu; Suyin Liang; Yubo Zhang; Yiqing Xiang; Jiaxuan Liu; Ting Sun; Manhui Lin; Yue Zhang; Changda Zhou; Tingquan Gao; Cheng Cui; Yi Liu; Dianhai Yu; Yanjun Ma

针对前代模型不稳定、数据覆盖不足或监督不可靠的区域，提出区域感知的数据优化方法。结合精选数据与强化学习进行渐进式后训练，为紧凑文档解析模型提供有针对性的能力提升路径。

[论文](https://arxiv.org/abs/2606.03264) · [PDF](https://arxiv.org/pdf/2606.03264) · [相关文档](version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{zhang2026paddleocrvl16expandingfrontierdocument,
      title={PaddleOCR-VL-1.6: Expanding the Frontier of Document Parsing with Under-Optimized Region Refinement and Progressive Post-Training},
      author={Zelun Zhang and Hongen Liu and Suyin Liang and Yubo Zhang and Yiqing Xiang and Jiaxuan Liu and Ting Sun and Manhui Lin and Yue Zhang and Changda Zhou and Tingquan Gao and Cheng Cui and Yi Liu and Dianhai Yu and Yanjun Ma},
      year={2026},
      eprint={2606.03264},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.03264},
}
```

</details>

<a id="paper-hpd-parsing"></a>

### HPD-Parsing

**[HPD-Parsing: Hierarchical Parallel Document Parsing](https://arxiv.org/abs/2607.18839)**

**首次公开:** 2026-07-21 · arXiv:2607.18839

**作者:** Shu Wei; Jingjing Wu; Lingshu Zhang; Qunyi Xie; Hao Zou; Le Xiang; Xu Fan; Yangliu Xu; Manhui Lin; Xiaolong Ma; Cheng Cui; Tengyu Du; YY

提出层级并行解码：主布局分支协调全局文档结构，动态派生多个局部内容分支并发解析文档块。结合渐进式多 token 预测（P-MTP）进一步缩短各分支的串行解码路径，在保持解析精度的同时提高长文档与批量处理的吞吐能力。

[论文](https://arxiv.org/abs/2607.18839) · [PDF](https://arxiv.org/pdf/2607.18839) · [相关文档](version3.x/pipeline_usage/HPD-Parsing.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{hpdparsing2026,
  title={HPD-Parsing: Hierarchical Parallel Document Parsing},
  author={Shu Wei and Jingjing Wu and Lingshu Zhang and Qunyi Xie and Hao Zou and Le Xiang and Xu Fan and Yangliu Xu and Manhui Lin and Xiaolong Ma and Cheng Cui and Tengyu Du and YY},
  year={2026},
  eprint={2607.18839},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2607.18839},
}
```

</details>

<a id="research-benchmarks"></a>

## 评测基准

了解如何评估真实采集条件下的文档解析鲁棒性。

<a id="paper-real5-omnidocbench"></a>

### Real5-OmniDocBench

**[Real5-OmniDocBench: A Full-Scale Physical Reconstruction Benchmark for Robust Document Parsing in the Wild](https://arxiv.org/abs/2603.04205)**

**首次公开:** 2026-03-04 · arXiv:2603.04205

**作者:** Changda Zhou; Ziyue Gao; Xueqing Wang; Tingquan Gao; Cheng Cui; Jing Tang; Yi Liu

将 OmniDocBench v1.5 的 1,355 张文档图像在扫描、弯折、屏幕拍摄、光照变化和倾斜五类真实条件下进行一一对应的物理重建。通过保留原始数字文档与采集图像的标注映射，帮助区分几何畸变、成像伪影和模型能力造成的解析退化，为真实场景文档解析提供可控的鲁棒性评测。

[论文](https://arxiv.org/abs/2603.04205) · [PDF](https://arxiv.org/pdf/2603.04205) · [数据集](https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{real5omnidocbench2026,
  title={Real5-OmniDocBench: A Full-Scale Physical Reconstruction Benchmark for Robust Document Parsing in the Wild},
  author={Changda Zhou and Ziyue Gao and Xueqing Wang and Tingquan Gao and Cheng Cui and Jing Tang and Yi Liu},
  year={2026},
  eprint={2603.04205},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.04205},
}
```

</details>

<a id="research-understanding"></a>

## 文档理解与生态工作

进一步探索文档内容理解；此处保留工作所属项目的说明。

<a id="paper-pp-docbee"></a>

### PP-DocBee

**[PP-DocBee: Improving Multimodal Document Understanding Through a Bag of Tricks](https://arxiv.org/abs/2503.04065)**

**首次公开:** 2025-03-06 · arXiv:2503.04065

**作者:** Feng Ni; Kui Huang; Yao Lu; Wenyu Lv; Guanzhong Wang; Zeyu Chen; Yi Liu

通过文档场景数据合成、动态比例采样以及预处理和 OCR 后处理策略增强多模态文档理解。训练代码与模型发布于 PaddleMIX，PaddleOCR 提供相关文档理解产线入口，因此作为飞桨生态关联工作单列。

[论文](https://arxiv.org/abs/2503.04065) · [PDF](https://arxiv.org/pdf/2503.04065) · [相关文档](version3.x/pipeline_usage/doc_understanding.md)

<details>
<summary>引用 BibTeX</summary>

```bibtex
@misc{ppdocbee2025,
  title={PP-DocBee: Improving Multimodal Document Understanding Through a Bag of Tricks},
  author={Ni, Feng and Huang, Kui and Lu, Yao and Lv, Wenyu and Wang, Guanzhong and Chen, Zeyu and Liu, Yi},
  year={2025},
  eprint={2503.04065},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.04065},
}
```

</details>

## 其他版本与技术文档

以下版本或组件已有官方技术说明；在本次核对中未找到可确认的独立论文条目，或其方法已由上方总体报告介绍，因此保留文档入口，不另造论文引用。

| 版本 / 组件 | 研究资料与对应关系 |
| --- | --- |
| [PP-OCRv4](version2.x/ppocr/blog/PP-OCRv4_introduction.md) | 2023 年技术报告（仓库文档） |
| [PP-Structure / SLANet / VI-LayoutXLM](version2.x/ppstructure/overview.md) | 参见 PP-StructureV2 论文与文档 |
| [PP-StructureV3](version3.x/algorithm/PP-StructureV3/PP-StructureV3.md) | 参见 PaddleOCR 3.0 论文 |
| [PP-ChatOCRv4](version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.md) | 参见 PaddleOCR 3.0 论文 |
| [PP-FormulaNet Plus / SLANeXt](version3.x/module_usage/formula_recognition.md) | 公式识别及[表格结构识别](version3.x/module_usage/table_structure_recognition.md)模块文档 |

版面分析系列的相关方法可参阅 [RT-DocLayout](#paper-rt-doclayout)，模型版本的使用方法以[版面分析模块文档](version3.x/module_usage/layout_analysis.md)为准。

如需补充论文，请同时更新本页中英文版本，附上作者论文页或正式出版链接，并注明与 PaddleOCR 的关系。
