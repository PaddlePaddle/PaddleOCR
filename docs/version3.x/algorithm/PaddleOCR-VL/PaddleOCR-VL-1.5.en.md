## 1. PaddleOCR-VL-1.5 Introduction

**PaddleOCR-VL-1.5** is an advanced next-generation model of PaddleOCR-VL, achieving a new state-of-the-art accuracy of 94.5% on OmniDocBench v1.5. To rigorously evaluate robustness against real-world physical distortions—including scanning artifacts, skew, warping, screen photography, and illumination—we propose the Real5-OmniDocBench benchmark. Experimental results demonstrate that this enhanced model attains SOTA performance on the newly curated benchmark. Furthermore, we extend the model’s capabilities by incorporating seal recognition and text spotting tasks, while remaining a 0.9B ultra-compact VLM with high efficiency.

### Key Metrics:

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/paddleocr-vl-1.5_metrics.png" width="800"/>
</div>


### **Core Features**

1. With a **parameter size of 0.9B**, PaddleOCR-VL-1.5 **achieves xxx% accuracy on OmniDocBench v1.5**, surpassing the previous SOTA model PaddleOCR-VL. Significant improvements are observed in **table, formula, and text understanding.**

2. **It introduces an innovative approach to document parsing by supporting irregular-shaped localization**, enabling accurate polygonal detection under skewed and curved document conditions. Evaluations across five real-world scenarios—scanning, curving, skewing, screen-photo capture, and light variation—demonstrate superior performance over mainstream open-source and proprietary models.

3. The model introduces **text spotting (text-line localization and recognition)**, along with **seal recognition**, with all corresponding metrics **setting new SOTA results** in their respective tasks.

4. PaddleOCR-VL-1.5 further strengthens its capability in **specialized scenarios and multilingual recognition.** Recognition performance is improved for **rare characters, ancient texts, multilingual tables, underlines, and checkboxes,** and language coverage is extended to include **China's Tibetan script and Bengali.**

5. The model supports **automatic cross-page table merging** and **cross-page paragraph heading recognition**, effectively mitigating content fragmentation issues in **long-document parsing.**


## 2. Model Architecture

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/PaddleOCR-VL-1.5.png" width="800"/>
</div>


## 3. Model Performance

### 1. OmniDocBench v1.5

#### PaddleOCR-VL achieves SOTA performance for overall, text, formula, tables and reading order on OmniDocBench v1.5.


<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/omnidocbenchv1.5_metrics.png" width="800"/>
</div>

> **Notes:** 
> - Performance metrics are cited from the [OmniDocBench official leaderboard](https://opendatalab.com/omnidocbench), except for Gemini-3 Pro, Qwen3-VL-235B-A22B-Instruct and our model, which were evaluated independently.


###  2. Real5-OmniDocBench

#### Across all five diverse and challenging scenarios—scanning, warping, screen-photography, illumination, and skew—PaddleOCR-VL-1.5 consistently sets new SOTA records


<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/real5-omnidocbench_metrics.png" width="800"/>
</div>

> **Notes:** 
> - Real5-OmniDocBench is a brand-new benchmark oriented toward real-world scenarios, which we constructed based on the OmniDocBench v1.5 dataset. The dataset comprises five distinct scenarios: Scanning, Warping, Screen-photography, Illumination, and Skew. For further details, please refer to [Real5-OmniDocBench](https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench).



## 4、Inference and deployment Performance


<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/inference_performance.png" width="600"/>
</div>

> **Notes:** 
> - End-to-End Inference Performance Comparison on OmniDocBench v1.5. PDF documents were processed in batches of 512 on a single NVIDIA A100 GPU. The reported end-to-end runtime includes both PDF rendering and Markdown generation. All methods rely on their built-in PDF parsing modules and default DPI settings to reflect out-of-the-box performance. 


## 5. Visualization

### Real-word Document Parsing


#### Illumination

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/light.jpg" width="800"/>
</div>


#### Skew

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/skew.jpg" width="800"/>
</div>


#### Screen Photography

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/screen.jpg" width="800"/>
</div>


#### Scanning

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/scaning.jpg" width="800"/>
</div>

#### Warping

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/curving.jpg" width="800"/>
</div>


### Text Spotting


<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/spotting.jpg" width="800"/>
</div>


### Seal Recognition


<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_5/seal.jpg" width="800"/>
</div>
