# Introduction to PP-ChatOCRV4
**PP-ChatOCRv4** is a unique document and image intelligent analysis solution from PaddlePaddle, combining LLM, MLLM, and OCR technologies to address complex document information extraction challenges such as layout analysis, rare characters, multi-page PDFs, tables, and seal recognition. Integrated with ERNIE Bot, it fuses massive data and knowledge, achieving high accuracy and wide applicability. This pipeline also provides flexible service deployment options, supporting deployment on various hardware. Furthermore, it offers custom development capabilities, allowing you to train and fine-tune models on your own datasets, with seamless integration of trained models.

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-ChatOCRv4/algorithm_ppchatocrv4.png" width="600"/>
</div>

# Key Metrics

<div align="center">
<table>
 <thead>
  <tr > 
   <th class>Solution</td> 
   <th class>Avg Recall</td> 
  </tr> 
<thead>
 <tbody>
  <tr> 
   <td>GPT-4o</td> 
   <td>63.47%</td> 
  </tr>
  <tr> 
   <td>PP-ChatOCRv3</td> 
   <td class>70.08%</td> 
  </tr> 
  <tr> 
   <td>Qwen2.5-VL-72B</td> 
   <td>80.26%</td> 
  </tr> 
  <tr> 
   <td><b>PP-ChatOCRv4</b></td> 
   <td><b>85.55%</b></td> 
  </tr> 
 </tbody>
</table>
</div>

# Demo

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-ChatOCRv4/algorithm_ppchatocrv4_demo1.png" width="350"/>
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-ChatOCRv4/algorithm_ppchatocrv4_demo2.png" width="350"/>
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-ChatOCRv4/algorithm_ppchatocrv4_demo3.png" width="350"/>
<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr/PP-ChatOCRv4/algorithm_ppchatocrv4_demo4.png" width="350"/>
</div>


# FAQ

1. Does support other multimodal models?

Yes, only set on pipeline configuration.

2. How to reduce latency and improve throughput?

Use the High-performance inference plugin, and deploy multi instances.

3. How to further improve accuracy?

Firstly, it is necessary to check whether the extracted visual information is correct. If the visual information is incorrect, it is necessary to visualize the visual prediction results to determine which model performs poorly, and then fine-tune train the model with more data. If the visual information is correct but cannot extract the correct information, the prompt needs to be adjusted according to the analysing about the question and answer.
