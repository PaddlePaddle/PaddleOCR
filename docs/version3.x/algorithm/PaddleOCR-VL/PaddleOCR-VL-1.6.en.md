## 1. Introduction to PaddleOCR-VL-1.6

**PaddleOCR-VL-1.6** further optimizes PaddleOCR-VL-1.5 by systematically analyzing under-optimized areas in the current model, applying targeted data optimization, and adopting refined post-training strategies. It achieves a new state-of-the-art (SOTA) result of 96.33% on the OmniDocBench v1.6 document parsing benchmark. PaddleOCR-VL-1.6 also reaches SOTA performance across all scenarios on Real5-OmniDocBench, a benchmark designed to evaluate robustness against real-world physical distortions. In addition, PaddleOCR-VL-1.6 outperforms PaddleOCR-VL-1.5 on three subtasks: seal recognition, text detection and recognition, and chart recognition, while still maintaining an ultra-compact 0.9B-parameter VLM and high efficiency.

### **Key Metrics:**

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/paddleocr-vl-1.6_metrics.png" width="800"/>
</div>


### **Core Features:**

1. **SOTA performance in document parsing:** With only 0.9B parameters, PaddleOCR-VL-1.6 achieves 96.33% accuracy on OmniDocBench v1.6, surpassing the previous SOTA model, PaddleOCR-VL-1.5. Significant improvements are observed in table, formula, and text recognition.

2. **SOTA performance for document parsing across five real-world scenarios:** PaddleOCR-VL-1.6 offers stronger robustness and practicality in real-world use cases. In evaluations across five real-world distortion scenarios—scanning, warping, skew, screen photography, and illumination variation—it outperforms mainstream open-source and closed-source models.

3. **Enhanced multi-element recognition capabilities:** Beyond improved layout parsing, PaddleOCR-VL-1.6 substantially strengthens recognition of complex tables, ancient books, and rare Chinese characters, while further improving three existing capabilities: chart parsing, seal recognition, and text detection and recognition.

4. **Compact 0.9B architecture:** PaddleOCR-VL-1.6 follows the compact 0.9B architecture of the PaddleOCR-VL series, enabling zero-cost adaptation and drop-in replacement.


## 2. Technical Architecture

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/overall.png" width="800"/>
</div> 

1. **Data engine:** Starting from PaddleOCR-VL-1.5, the data engine systematically identifies under-optimized areas in PaddleOCR-VL-1.5, designs strategies for obtaining high-quality labels, and performs targeted data optimization.
2. **Progressive post-training strategy:** Data is carefully categorized from three perspectives: quality, difficulty, and improvement value. The training weights of PaddleOCR-VL-1.5 are loaded, and a three-stage post-training strategy—continued pre-training, supervised fine-tuning, and reinforcement learning—is applied according to different data quality levels to steadily improve model performance.


## 3. Model Performance

### 1. OmniDocBench v1.6

#### PaddleOCR-VL-1.6 achieves state-of-the-art performance on OmniDocBench v1.6 in overall metrics, text, formulas, and tables. It also delivers leading results in reading order.


<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/table2.png" width="800"/>
</div>

> **Note:** 
> - Performance metrics are cited from the [official OmniDocBench leaderboard](https://opendatalab.com/omnidocbench).


###  2. Real5-OmniDocBench

#### PaddleOCR-VL-1.6 sets new SOTA records across five diverse and challenging scenarios: scanning, warping, screen photography, illumination, and skew.


<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/table3.png" width="800"/>
</div>

> **Note:** 
> - Real5-OmniDocBench is a new real-world benchmark built by the PaddleOCR team based on the OmniDocBench v1.5 dataset. It contains five scenarios: Scanning, Warping, Screen-photography, Illumination, and Skew. For more details, see [Real5-OmniDocBench](https://huggingface.co/datasets/PaddlePaddle/Real5-OmniDocBench).



## 4. Inference and Deployment Performance

PaddleOCR-VL-1.6 and PaddleOCR-VL-1.5 use exactly the same model architecture design, so they have identical inference speeds. For details about the inference speed of PaddleOCR-VL-1.5, refer to [PaddleOCR-VL-1.5 inference speed](./PaddleOCR-VL-1.5.md#4-inference-and-deployment-performance).


## 5. Visualization

### Comparison with PaddleOCR-VL-1.5


#### Ancient Book Recognition

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/ancient1.png" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/ancient2.png" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/ancient3.png" width="800"/>
</div>


#### Chart Parsing

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/chart1.png" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/chart2.png" width="800"/>
</div>


#### Formula Recognition

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/fomula1.png" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/fomula2.png" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/fomula3.png" width="800"/>
</div>

#### Rare Chinese Character Recognition

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/rare-cha1.png" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/rare-cha2.png" width="800"/>
</div>

#### Seal Recognition

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/seal1.png" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/seal2.png" width="800"/>
</div>


### Table Recognition


<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/table1.png" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/table2.png" width="800"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl_1_6/comparison/table3.png" width="800"/>
</div>
