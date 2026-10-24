---
comments: true
---

# PaddleOCR Research Papers

From lightweight OCR in 2020 to structured document parsing and vision-language models, this page traces six years of PaddleOCR research and the ideas behind its development.

**Scope and verification date:** As of 2026-09-14, this catalog includes verified PaddleOCR series papers and related Baidu-associated algorithm research implemented in PaddleOCR, with the origins of collaborative and PaddlePaddle ecosystem work identified. Downstream papers that only use PaddleOCR are outside this scope; see the [algorithm overview](version2.x/algorithm/overview.en.md) for other third-party implementations. Dates indicate the first public paper release, which can differ from the model release date; later versions of the same preprint are not counted again.

## Reading guide

Start with the toolkit overview, then browse by research area. Within each area, papers in the same family stay together in order of first publication; related algorithms form a separate subgroup.

| Research area | Suggested reading order |
| --- | --- |
| [Toolkit overview](#research-overview) | [PaddleOCR 3.0](#paper-paddleocr-3-0) |
| [Text recognition](#research-ocr) | [PP-OCR](#paper-pp-ocr) → [PP-OCRv2](#paper-pp-ocrv2) → [PP-OCRv3](#paper-pp-ocrv3) → [PP-OCRv5](#paper-pp-ocrv5) → [PP-OCRv6](#paper-pp-ocrv6); [SVTR](#paper-svtr) → [CPPD](#paper-cppd) |
| [Document structure and layout analysis](#research-layout) | [PP-StructureV2](#paper-pp-structurev2) → [PP-DocLayout](#paper-pp-doclayout) → [RT-DocLayout / PP-DocLayoutV3](#paper-rt-doclayout) |
| [Formula recognition](#research-formula) | [PP-FormulaNet](#paper-pp-formulanet) |
| [Vision-language document parsing](#research-parsing) | [PaddleOCR-VL](#paper-paddleocr-vl) → [PaddleOCR-VL-1.5](#paper-paddleocr-vl-1-5) → [PaddleOCR-VL-1.6](#paper-paddleocr-vl-1-6) → [HPD-Parsing](#paper-hpd-parsing) |
| [Evaluation benchmarks](#research-benchmarks) | [Real5-OmniDocBench](#paper-real5-omnidocbench) |
| [Document understanding and ecosystem research](#research-understanding) | [PP-DocBee](#paper-pp-docbee) |

## How to cite

The README highlights PaddleOCR 3.0 and PaddleOCR-VL as the two overarching works. When using a specific model or algorithm, cite its corresponding paper below. Each entry includes copyable BibTeX.

<a id="research-overview"></a>

## Toolkit overview

Start with the toolkit’s capabilities and technical architecture.

<a id="paper-paddleocr-3-0"></a>

### PaddleOCR 3.0

**[PaddleOCR 3.0 Technical Report](https://arxiv.org/abs/2507.05595)**

**First public release:** 2025-07-08 · arXiv:2507.05595

**Authors:** Cheng Cui; Ting Sun; Manhui Lin; Tingquan Gao; Yubo Zhang; Jiaxuan Liu; Xueqing Wang; Zelun Zhang; Changda Zhou; Hongen Liu; Yue Zhang; Wenyu Lv; Kui Huang; Yichao Zhang; Jing Zhang; Jun Zhang; Yi Liu; Dianhai Yu; Yanjun Ma

Presents PP-OCRv5 multilingual recognition, PP-StructureV3 hierarchical document parsing, and PP-ChatOCRv4 key information extraction. It also covers training, inference, deployment, and heterogeneous hardware support, documenting the transition from an OCR toolkit to a document intelligence toolchain.

[Paper](https://arxiv.org/abs/2507.05595) · [PDF](https://arxiv.org/pdf/2507.05595) · [Documentation](index.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

## Text recognition

Follow the PP-OCR system family, then explore research on recognition algorithms.

### PP-OCR family

<a id="paper-pp-ocr"></a>

#### PP-OCR

**[PP-OCR: A Practical Ultra Lightweight OCR System](https://arxiv.org/abs/2009.09941)**

**First public release:** 2020-09-21 · arXiv:2009.09941

**Authors:** Yuning Du; Chenxia Li; Ruoyu Guo; Xiaoting Yin; Weiwei Liu; Jun Zhou; Yifan Bai; Zilin Yu; Yehua Yang; Qingqing Dang; Haoshuang Wang

Combines text detection, orientation classification, and recognition into a deployable lightweight OCR system. Backbone selection, augmentation, training strategies, pruning, and quantization balance accuracy and model size. It establishes the practical foundation of the PP-OCR family for Chinese, English, and other languages.

[Paper](https://arxiv.org/abs/2009.09941) · [PDF](https://arxiv.org/pdf/2009.09941) · [Documentation](version2.x/ppocr/overview.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2021-09-07 · arXiv:2109.03144

**Authors:** Yuning Du; Chenxia Li; Ruoyu Guo; Cheng Cui; Weiwei Liu; Jun Zhou; Bin Lu; Yehua Yang; Qiwen Liu; Xiaoguang Hu; Dianhai Yu; Yanjun Ma

Improves PP-OCR through collaborative mutual learning, CopyPaste augmentation, PP-LCNet, unified deep mutual learning, and an improved CTC loss. The system focuses on increasing detection and recognition accuracy while retaining efficient deployment.

[Paper](https://arxiv.org/abs/2109.03144) · [PDF](https://arxiv.org/pdf/2109.03144) · [Documentation](version2.x/ppocr/overview.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2022-06-07 · arXiv:2206.03001

**Authors:** Chenxia Li; Weiwei Liu; Ruoyu Guo; Xiaoting Yin; Kaitao Jiang; Yongkun Du; Yuning Du; Lingfeng Zhu; Baohua Lai; Xiaoguang Hu; Dianhai Yu; Yanjun Ma

Introduces nine detection and recognition improvements, including LK-PAN, RSE-FPN, distillation, and SVTR_LCNet. Attention-guided CTC training, augmentation, and pretraining improve the accuracy–speed balance, bringing visual Transformer ideas into practical lightweight OCR.

[Paper](https://arxiv.org/abs/2206.03001) · [PDF](https://arxiv.org/pdf/2206.03001) · [Documentation](version2.x/ppocr/blog/PP-OCRv3_introduction.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2026-03-25 · arXiv:2603.24373

**Authors:** Cheng Cui; Yubo Zhang; Ting Sun; Xueqing Wang; Hongen Liu; Manhui Lin; Yue Zhang; Tingquan Gao; Changda Zhou; Jiaxuan Liu; Zelun Zhang; Jing Zhang; Jun Zhang; Yi Liu

Examines how a lightweight specialized OCR model can compete with large vision-language models on OCR tasks. Experiments study data difficulty, annotation accuracy, and diversity, demonstrating the potential of a roughly 5M-parameter system. This separate paper appeared in 2026, following the 2025 model release and PaddleOCR 3.0 report.

[Paper](https://arxiv.org/abs/2603.24373) · [PDF](https://arxiv.org/pdf/2603.24373) · [Documentation](version3.x/algorithm/PP-OCRv5/PP-OCRv5.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2026-06-11 · arXiv:2606.13108

**Authors:** Yubo Zhang; Xueqing Wang; Manhui Lin; Yue Zhang; Penglongyi Deng; Ting Sun; Tingquan Gao; Zelun Zhang; Jiaxuan Liu; Changda Zhou; Hongen Liu; Suyin Liang; Cheng Cui; Yi Liu; Dianhai Yu; Yanjun Ma

Redesigns the backbone and detection and recognition necks using unified MetaFormer-style blocks and structural reparameterization, combined with data-centric optimization. Tiny, small, and medium tiers span 1.5M to 34.5M parameters for edge-to-server deployment, using task-specific stride configurations to support both detection and recognition.

[Paper](https://arxiv.org/abs/2606.13108) · [PDF](https://arxiv.org/pdf/2606.13108) · [Documentation](version3.x/algorithm/PP-OCRv6/PP-OCRv6.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

### Recognition algorithms

<a id="paper-svtr"></a>

#### SVTR

**[SVTR: Scene Text Recognition with a Single Visual Model](https://arxiv.org/abs/2205.00159)**

**First public release:** 2022-04-30 · arXiv:2205.00159

**Authors:** Yongkun Du; Zhineng Chen; Caiyan Jia; Xiaoting Yin; Tianlun Zheng; Chenxia Li; Yuning Du; Yu-Gang Jiang

Uses local and global mixing to model visual relationships within and between characters, recognizing text with a single visual model and a simple prediction layer. This collaborative research underpins SVTR_LCNet in PP-OCRv3 and has a public PaddleOCR implementation.

[Paper](https://arxiv.org/abs/2205.00159) · [PDF](https://arxiv.org/pdf/2205.00159) · [Documentation](version2.x/algorithm/text_recognition/algorithm_rec_svtr.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2023-07-23 · arXiv:2307.12270

**Authors:** Yongkun Du; Zhineng Chen; Caiyan Jia; Xiaoting Yin; Chenxia Li; Yuning Du; Yu-Gang Jiang

Uses character counting and character ordering tasks to model recognition context, then predicts the sequence in parallel. This collaborative research aims to retain contextual modeling while improving decoding efficiency; PaddleOCR provides training configurations and reproduction documentation.

[Paper](https://arxiv.org/abs/2307.12270) · [PDF](https://arxiv.org/pdf/2307.12270) · [Documentation](version2.x/algorithm/text_recognition/algorithm_rec_cppd.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

## Document structure and layout analysis

From document analysis systems to layout models that jointly predict elements and reading order.

<a id="paper-pp-structurev2"></a>

### PP-StructureV2

**[PP-StructureV2: A Stronger Document Analysis System](https://arxiv.org/abs/2210.05391)**

**First public release:** 2022-10-11 · arXiv:2210.05391

**Authors:** Chenxia Li; Ruoyu Guo; Jun Zhou; Mengtao An; Yuning Du; Lingfeng Zhu; Yi Liu; Xiaoguang Hu; Dianhai Yu

Organizes document analysis into layout information extraction and key information extraction, with image orientation correction and layout recovery. It introduces lightweight layout detection, SLANet table recognition, and VI-LayoutXLM to turn document images into structured content.

[Paper](https://arxiv.org/abs/2210.05391) · [PDF](https://arxiv.org/pdf/2210.05391) · [Documentation](version2.x/ppstructure/overview.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2025-03-21 · arXiv:2503.17213

**Authors:** Ting Sun; Cheng Cui; Yuning Du; Yi Liu

Detects 23 types of layout regions, including titles, text, tables, and formulas, across diverse document formats. L, M, and S variants target high accuracy, balanced performance, and resource-constrained deployment, supporting large-scale document processing and training-data construction. Code and models are released in PaddleX, with layout detection available through PaddleOCR.

[Paper](https://arxiv.org/abs/2503.17213) · [PDF](https://arxiv.org/pdf/2503.17213) · [Documentation](version3.x/module_usage/layout_detection.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2026-06-22 · arXiv:2606.23344

**Authors:** Cheng Cui; Tingquan Gao; Xueqing Wang; Changda Zhou; Hongen Liu; Ting Sun; Yubo Zhang; Zelun Zhang; Jiaxuan Liu; Manhui Lin; Yue Zhang; Suyin Liang; Yiqing Xiang; Yi Liu

Builds on RT-DETR to unify element classification, bounding-box detection, pixel-level segmentation, and reading-order prediction in a single query-based decoder. Joint geometric and structural learning improves layout analysis under warping and perspective changes, providing an efficient front end for downstream OCR and full-document reconstruction.

**Version mapping:** The [official PP-DocLayoutV3 model card](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) cites this RT-DocLayout paper. The original PP-DocLayout L/M/S models are covered by the [2025 paper above](#paper-pp-doclayout); the two works have separate entries.

[Paper](https://arxiv.org/abs/2606.23344) · [PDF](https://arxiv.org/pdf/2606.23344) · [Documentation](version3.x/module_usage/layout_analysis.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

## Formula recognition

Convert formula images into machine-readable symbolic expressions.

<a id="paper-pp-formulanet"></a>

### PP-FormulaNet

**[PP-FormulaNet: Bridging Accuracy and Efficiency in Advanced Formula Recognition](https://arxiv.org/abs/2503.18382)**

**First public release:** 2025-03-24 · arXiv:2503.18382

**Authors:** Hongen Liu; Cheng Cui; Yuning Du; Yi Liu; Gang Pan

Converts formula images into LaTeX with an accuracy-oriented L model and an efficiency-oriented S model. A formula mining system expands high-quality training data, supporting different compute budgets and mathematical document processing needs.

[Paper](https://arxiv.org/abs/2503.18382) · [PDF](https://arxiv.org/pdf/2503.18382) · [Documentation](version3.x/module_usage/formula_recognition.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

## Vision-language document parsing

Follow successive PaddleOCR-VL releases, then explore hierarchical parallel decoding in HPD-Parsing.

<a id="paper-paddleocr-vl"></a>

### PaddleOCR-VL

**[PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model](https://arxiv.org/abs/2510.14528)**

**First public release:** 2025-10-16 · arXiv:2510.14528

**Authors:** Cheng Cui; Ting Sun; Suyin Liang; Tingquan Gao; Zelun Zhang; Jiaxuan Liu; Xueqing Wang; Changda Zhou; Hongen Liu; Manhui Lin; Yue Zhang; Yubo Zhang; Handong Zheng; Jing Zhang; Jun Zhang; Yi Liu; Dianhai Yu; Yanjun Ma

Builds a 0.9B document parsing model from a NaViT-style dynamic-resolution visual encoder and ERNIE-4.5-0.3B. It supports 109 languages and elements including text, tables, formulas, and charts, exploring efficient document parsing with a compact vision-language model.

[Paper](https://arxiv.org/abs/2510.14528) · [PDF](https://arxiv.org/pdf/2510.14528) · [Documentation](version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2026-01-29 · arXiv:2601.21957

**Authors:** Cheng Cui; Ting Sun; Suyin Liang; Tingquan Gao; Zelun Zhang; Jiaxuan Liu; Xueqing Wang; Changda Zhou; Hongen Liu; Manhui Lin; Yue Zhang; Yubo Zhang; Yi Liu; Dianhai Yu; Yanjun Ma

Improves real-world document parsing while retaining a 0.9B model and adding seal recognition and text spotting. It introduces Real5-OmniDocBench to evaluate scanning, skew, warping, screen photography, and illumination, extending evaluation to robustness under practical capture conditions.

[Paper](https://arxiv.org/abs/2601.21957) · [PDF](https://arxiv.org/pdf/2601.21957) · [Documentation](version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.5.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2026-06-02 · arXiv:2606.03264

**Authors:** Zelun Zhang; Hongen Liu; Suyin Liang; Yubo Zhang; Yiqing Xiang; Jiaxuan Liu; Ting Sun; Manhui Lin; Yue Zhang; Changda Zhou; Tingquan Gao; Cheng Cui; Yi Liu; Dianhai Yu; Yanjun Ma

Targets regions where the previous model is unstable, data coverage is sparse, or supervision is unreliable. Region-aware data optimization and progressive post-training with curated data and reinforcement learning provide a targeted route to improve compact document parsing models.

[Paper](https://arxiv.org/abs/2606.03264) · [PDF](https://arxiv.org/pdf/2606.03264) · [Documentation](version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

**First public release:** 2026-07-21 · arXiv:2607.18839

**Authors:** Shu Wei; Jingjing Wu; Lingshu Zhang; Qunyi Xie; Hao Zou; Le Xiang; Xu Fan; Yangliu Xu; Manhui Lin; Xiaolong Ma; Cheng Cui; Tengyu Du; YY

Introduces hierarchical parallel decoding: a main layout branch coordinates document structure and dynamically spawns concurrent branches for block content. Progressive multi-token prediction (P-MTP) further shortens each branch’s sequential decoding path, improving throughput for long documents and batched parsing while retaining competitive accuracy.

[Paper](https://arxiv.org/abs/2607.18839) · [PDF](https://arxiv.org/pdf/2607.18839) · [Documentation](version3.x/pipeline_usage/HPD-Parsing.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

## Evaluation benchmarks

Evaluate document parsing robustness under real-world capture conditions.

<a id="paper-real5-omnidocbench"></a>

### Real5-OmniDocBench

**[Real5-OmniDocBench: A Full-Scale Physical Reconstruction Benchmark for Robust Document Parsing in the Wild](https://arxiv.org/abs/2603.04205)**

**First public release:** 2026-03-04 · arXiv:2603.04205

**Authors:** Changda Zhou; Ziyue Gao; Xueqing Wang; Tingquan Gao; Cheng Cui; Jing Tang; Yi Liu

Physically reconstructs all 1,355 OmniDocBench v1.5 images under scanning, warping, screen photography, illumination, and skew. One-to-one ground-truth correspondence with the digital originals supports controlled analysis of parsing degradation from geometry, imaging artifacts, and model limitations, enabling systematic robustness evaluation in real-world conditions.

[Paper](https://arxiv.org/abs/2603.04205) · [PDF](https://arxiv.org/pdf/2603.04205) · [Dataset](https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench)

<details>
<summary>Cite with BibTeX</summary>

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

## Document understanding and ecosystem research

Explore document content understanding, with the originating project identified for each work.

<a id="paper-pp-docbee"></a>

### PP-DocBee

**[PP-DocBee: Improving Multimodal Document Understanding Through a Bag of Tricks](https://arxiv.org/abs/2503.04065)**

**First public release:** 2025-03-06 · arXiv:2503.04065

**Authors:** Feng Ni; Kui Huang; Yao Lu; Wenyu Lv; Guanzhong Wang; Zeyu Chen; Yi Liu

Improves multimodal document understanding with document-specific data synthesis, dynamic proportional sampling, preprocessing, and OCR postprocessing. Training code and models are released in PaddleMIX, with a related document understanding pipeline in PaddleOCR, so it is listed separately as ecosystem research.

[Paper](https://arxiv.org/abs/2503.04065) · [PDF](https://arxiv.org/pdf/2503.04065) · [Documentation](version3.x/pipeline_usage/doc_understanding.en.md)

<details>
<summary>Cite with BibTeX</summary>

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

## Other releases and technical documentation

These releases or components have official technical documentation. This survey did not identify a confirmed standalone paper for them, or their methods are covered by an overarching report above. Documentation links are retained without inventing separate paper citations.

| Release / component | Research material and relationship |
| --- | --- |
| [PP-OCRv4](version2.x/ppocr/blog/PP-OCRv4_introduction.md) | 2023 technical report in repository documentation (Chinese) |
| [PP-Structure / SLANet / VI-LayoutXLM](version2.x/ppstructure/overview.en.md) | Covered by the PP-StructureV2 paper and documentation |
| [PP-StructureV3](version3.x/algorithm/PP-StructureV3/PP-StructureV3.en.md) | Covered by the PaddleOCR 3.0 paper |
| [PP-ChatOCRv4](version3.x/algorithm/PP-ChatOCRv4/PP-ChatOCRv4.en.md) | Covered by the PaddleOCR 3.0 paper |
| [PP-FormulaNet Plus / SLANeXt](version3.x/module_usage/formula_recognition.en.md) | Formula recognition and [table structure recognition](version3.x/module_usage/table_structure_recognition.en.md) module documentation |

For related layout analysis methods, see [RT-DocLayout](#paper-rt-doclayout); consult the [layout analysis documentation](version3.x/module_usage/layout_analysis.en.md) for model-specific usage.

To add a paper, update both language versions, include an author paper page or publication link, and state its relationship to PaddleOCR.
