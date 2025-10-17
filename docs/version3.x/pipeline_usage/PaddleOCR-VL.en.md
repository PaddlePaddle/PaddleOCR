---
comments: true
---

# PaddleOCR-VL Introduction

PaddleOCR-VL is a SOTA and resource-efficient model tailored for document parsing. Its core component is PaddleOCR-VL-0.9B, a compact yet powerful vision-language model (VLM) that integrates a NaViT-style dynamic resolution visual encoder with the ERNIE-4.5-0.3B language model to enable accurate element recognition. This innovative model efficiently supports 109 languages and excels in recognizing complex elements (e.g., text, tables, formulas, and charts), while maintaining minimal resource consumption. Through comprehensive evaluations on widely used public benchmarks and in-house benchmarks, PaddleOCR-VL achieves SOTA performance in both page-level document parsing and element-level recognition. It significantly outperforms existing solutions, exhibits strong competitiveness against top-tier VLMs, and delivers fast inference speeds. These strengths make it highly suitable for practical deployment in real-world scenarios.

## 1. Environment Preparation

Install PaddlePaddle and PaddleOCR:

```shell
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl
```
> For Windows users, please use WSL or a Docker container.

## 2. Quick Start

PaddleOCR-VL supports two usage methods: CLI command line and Python API. The CLI command line method is simpler and suitable for quickly verifying functionality, while the Python API method is more flexible and suitable for integration into existing projects.

### 2.1 Command Line Usage

Run a single command to quickly test the PaddleOCR-VL ：

```bash
paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png

# Use --use_doc_orientation_classify to enable document orientation classification
paddleocr doc_parser -i ./paddleocr_vl_demo.png --use_doc_orientation_classify True

# Use --use_doc_unwarping to enable document unwarping module
paddleocr doc_parser -i ./paddleocr_vl_demo.png --use_doc_unwarping True

# Use --use_layout_detection to enable layout detection
paddleocr doc_parser -i ./paddleocr_vl_demo.png --use_layout_detection False
```

<details><summary><b>Command line supports more parameters. Click to expand for detailed parameter descriptions</b></summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Description</th>
<th>Type</th>
<th>Default</th>
</tr>
</thead>
<tbody>
<tr>
<td> <code> input</code></td>
<td>Data to be predicted, required.
For example, the local path of an image file or PDF file: <code> /root/data/img.jpg</code>;<b>Such as a URL link</b>, for example, the network URL of an image file or PDF file:<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>;<b>Such as a local directory</b>, which should contain the images to be predicted, for example, the local path: <code> /root/data/</code>(Currently, prediction for directories containing PDF files is not supported. PDF files need to be specified with a specific file path).</td>
<td> <code> str</code></td>
<td></td>
</tr>
<tr>
<td> <code> save_path</code></td>
<td>Specify the path where the inference result file will be saved. If not set, the inference results will not be saved locally.</td>
<td> <code> str</code></td>
<td></td>
</tr>
<tr>
<td> <code> layout_detection_model_name</code></td>
<td>Name of the layout area detection and ranking model. If not set, the default model of the production line will be used.</td>
<td> <code> str</code></td>
<td></td>
</tr>
<tr>
<td> <code> layout_detection_model_dir</code></td>
<td>Directory path of the layout area detection and ranking model. If not set, the official model will be downloaded.</td>
<td> <code> str</code></td>
<td></td>
</tr>
<tr>
<td> <code> layout_threshold</code></td>
<td>Score threshold for the layout model. <code> Any floating-point number between 0-1. If not set, the parameter value initialized by the production line will be used.</code>float</td>
<td> <code> layout_nms</code></td>
<td> <code> layout_threshold</code></td>
<td>Score threshold for the layout model. Any value between  <code> 0-1</code>. If not set, the default value is used, which is  <code> 0.5</code>.
</td>


<td></td>
</tr>
<tr>
<td> <code> Whether to use post-processing NMS for layout detection. If not set, the parameter value initialized by the production line will be used, with a default initialization of</code></td>
<td>True <code> .</code>bool</td>
<td> <code> layout_unclip_ratio</code></td>
<td></td>
</tr>
<tr>
<td> <code> Expansion coefficient for the detection boxes of the layout area detection model.
Any floating-point number greater than</code></td>
<td>0 <code> . If not set, the parameter value initialized by the production line will be used.</code>float</td>
<td> <code> layout_merge_bboxes_mode</code></td>
<td></td>
</tr>
<tr>
<td> <code> Merging mode for the detection boxes output by the model in layout detection.</code></td>
<td>large<ul>
<li><b>, when set to large, it means that among the detection boxes output by the model, for overlapping and contained boxes, only the outermost largest box is retained, and the overlapping inner boxes are deleted;</b>small</li>
<li><b>, when set to small, it means that among the detection boxes output by the model, for overlapping and contained boxes, only the innermost small box is retained, and the overlapping outer boxes are deleted;</b>union</li>
<li><b>, no filtering of boxes is performed, and both inner and outer boxes are retained;</b>If not set, the parameter value initialized by the production line will be used.</li>
</ul>str</td>
<td> <code> vl_rec_model_name</code></td>
<td></td>
</tr>
<tr>
<td> <code> Name of the multimodal recognition model. If not set, the default model of the production line will be used.</code></td>
<td>str</td>
<td> <code> vl_rec_model_dir</code></td>
<td></td>
</tr>
<tr>
<td> <code> Directory path of the multimodal recognition model. If not set, the official model will be downloaded.</code></td>
<td>str</td>
<td> <code> vl_rec_backend</code></td>
<td></td>
</tr>
<tr>
<td> <code> Inference backend used by the multimodal recognition model.</code></td>
<td>str</td>
<td> <code> vl_rec_server_url</code></td>
<td></td>
</tr>
<tr>
<td> <code> If the multimodal recognition model uses an inference service, this parameter is used to specify the server URL.</code></td>
<td>str</td>
<td> <code> vl_rec_max_concurrency</code></td>
<td></td>
</tr>
<tr>
<td> <code> If the multimodal recognition model uses an inference service, this parameter is used to specify the maximum number of concurrent requests.</code></td>
<td>str</td>
<td> <code> doc_orientation_classify_model_name</code></td>
<td></td>
</tr>
<tr>
<td> <code> Name of the document orientation classification model. If not set, the default model of the production line will be used.</code></td>
<td>str</td>
<td> <code> doc_orientation_classify_model_dir</code></td>
<td></td>
</tr>
<tr>
<td> <code> Directory path of the document orientation classification model. If not set, the official model will be downloaded.</code></td>
<td>str</td>
<td> <code> doc_unwarping_model_name</code></td>
<td></td>
</tr>
<tr>
<td> <code> Name of the text image rectification model. If not set, the default model of the production line will be used.</code></td>
<td>str</td>
<td> <code> doc_unwarping_model_dir</code></td>
<td></td>
</tr>
<tr>
<td> <code> Directory path of the text image rectification model. If not set, the official model will be downloaded.</code></td>
<td>str</td>
<td> <code> use_doc_orientation_classify</code></td>
<td></td>
</tr>
<tr>
<td> <code> Whether to load and use the document orientation classification module. If not set, the parameter value initialized by the production line will be used, with a default initialization of</code></td>
<td>False <code> .</code>bool</td>
<td> <code> use_doc_unwarping</code></td>
<td></td>
</tr>
<tr>
<td> <code> Whether to load and use the text image rectification module. If not set, the parameter value initialized by the production line will be used, with a default initialization of</code></td>
<td>False <code> .</code>bool</td>
<td> <code> use_layout_detection</code></td>
<td></td>
</tr>
<tr>
<td> <code> Whether to load and use the layout area detection and ranking module. If not set, the parameter value initialized by the production line will be used, with a default initialization of</code></td>
<td>True <code> .</code>bool</td>
<td> <code> use_chart_recognition</code></td>
<td></td>
</tr>
<tr>
<td> <code> Whether to load and use the chart parsing module. If not set, the parameter value initialized by the production line will be used, with a default initialization of</code></td>
<td>False <code> .</code>bool</td>
<td> <code> bool</code></td>
<td></td>
</tr>
<tr>
<td> <code> format_block_content</code></td>
<td>Controls whether to format the content in <code> block_content</code>as Markdown. If not set, the parameter value initialized by the production line will be used, which is initially set to <code> False</code>by default.</td>
<td> <code> bool</code></td>
<td></td>
</tr>
<tr>
<td> <code> use_queues</code></td>
<td>Used to control whether to enable internal queues. When set to <code> True</code>, data loading (such as rendering PDF pages as images), layout detection model processing, and VLM inference will be executed asynchronously in separate threads, with data passed through queues, thereby improving efficiency. This approach is particularly efficient for PDF documents with a large number of pages or directories containing a large number of images or PDF files.</td>
<td> <code> bool</code></td>
<td></td>
</tr>
<tr>
<td> <code> prompt_label</code></td>
<td>The prompt type setting for the VL model, which takes effect only when <code> use_layout_detection=False</code>.</td>
<td> <code> str</code></td>
<td></td>
</tr>
<tr>
<td> <code> repetition_penalty</code></td>
<td>The repetition penalty parameter used for VL model sampling.</td>
<td> <code> float</code></td>
<td></td>
</tr>
<tr>
<td> <code> temperature</code></td>
<td>The temperature parameter used for VL model sampling.</td>
<td> <code> float</code></td>
<td></td>
</tr>
<tr>
<td> <code> top_p</code></td>
<td>The top-p parameter used for VL model sampling.</td>
<td> <code> float</code></td>
<td></td>
</tr>
<tr>
<td> <code> min_pixels</code></td>
<td>The minimum number of pixels allowed when the VL model preprocesses images.</td>
<td> <code> int</code></td>
<td></td>
</tr>
<tr>
<td> <code> max_pixels</code></td>
<td>The maximum number of pixels allowed when the VL model preprocesses images.</td>
<td> <code> int</code></td>
<td></td>
</tr>
<tr>
<td> <code> device</code></td>
<td>The device used for inference. Supports specifying specific card numbers:<ul>
<li><b>CPU</b>: For example, <code> cpu</code>indicates using the CPU for inference;</li>
<li><b>GPU</b>: For example, <code> gpu:0</code>indicates using the first GPU for inference;</li>
<li><b>NPU</b>: For example, <code> npu:0</code>indicates using the first NPU for inference;</li>
<li><b>XPU</b>: For example, <code> xpu:0</code>indicates using the first XPU for inference;</li>
<li><b>MLU</b>: For example, <code> mlu:0</code>indicates using the first MLU for inference;</li>
<li><b>DCU</b>: For example, <code> dcu:0</code>indicates using the first DCU for inference;</li>
</ul>If not set, the parameter value initialized by the production line will be used by default. During initialization, the local GPU device 0 will be used preferentially. If it is not available, the CPU device will be used.</td>
<td> <code> str</code></td>
<td></td>
</tr>
<tr>
<td> <code> enable_hpi</code></td>
<td>Whether to enable high-performance inference.</td>
<td> <code> bool</code></td>
<td> <code> False</code></td>
</tr>
<tr>
<td> <code> use_tensorrt</code></td>
<td>Whether to enable the TensorRT subgraph engine of Paddle Inference. If the model does not support acceleration via TensorRT, acceleration will not be used even if this flag is set.<br/>For PaddlePaddle with CUDA 11.8, the compatible TensorRT version is 8.x (x&amp;gt;=6). It is recommended to install TensorRT 8.6.1.6.<br/>
</td>
<td> <code> bool</code></td>
<td> <code> False</code></td>
</tr>
<tr>
<td> <code> precision</code></td>
<td>Computational precision, such as fp32, fp16.</td>
<td> <code> str</code></td>
<td> <code> fp32</code></td>
</tr>
<tr>
<td> <code> enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN accelerated inference. If MKL-DNN is not available or the model does not support acceleration via MKL-DNN, acceleration will not be used even if this flag is set.</td>
<td> <code> bool</code></td>
<td> <code> True</code></td>
</tr>
<tr>
<td> <code> mkldnn_cache_capacity</code></td>
<td>MKL-DNN cache capacity.</td>
<td> <code> int</code></td>
<td> <code> 10</code></td>
</tr>
<tr>
<td> <code> cpu_threads</code></td>
<td>The number of threads used for inference on the CPU.</td>
<td> <code> int</code></td>
<td> <code> 8</code></td>
</tr>
<tr>
<td> <code> paddlex_config</code></td>
<td>The file path of the PaddleX production line configuration.</td>
<td> <code> str</code></td>
<td></td>
</tr>
</tbody>
</table>
</details>
<br />

The inference result will be printed in the terminal. The default output of the PP-StructureV3 pipeline is as follows:

<details><summary> 👉Click to expand</summary>
<pre>
 <code> 
{'res': {'input_path': 'paddleocr_vl_demo.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': False, 'use_layout_detection': True, 'use_chart_recognition': False, 'format_block_content': False}, 'layout_det_res': {'input_path': None, 'page_index': None, 'boxes': [{'cls_id': 6, 'label': 'doc_title', 'score': 0.9636914134025574, 'coordinate': [np.float32(131.31366), np.float32(36.450516), np.float32(1384.522), np.float32(127.984665)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9281806349754333, 'coordinate': [np.float32(585.39465), np.float32(158.438), np.float32(930.2184), np.float32(182.57469)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840355515480042, 'coordinate': [np.float32(9.023666), np.float32(200.86115), np.float32(361.41583), np.float32(343.8828)]}, {'cls_id': 14, 'label': 'image', 'score': 0.9871416091918945, 'coordinate': [np.float32(775.50574), np.float32(200.66502), np.float32(1503.3807), np.float32(684.9304)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9801855087280273, 'coordinate': [np.float32(9.532196), np.float32(344.90594), np.float32(361.4413), np.float32(440.8244)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9708921313285828, 'coordinate': [np.float32(28.040405), np.float32(455.87976), np.float32(341.7215), np.float32(520.7117)]}, {'cls_id': 24, 'label': 'vision_footnote', 'score': 0.9002962708473206, 'coordinate': [np.float32(809.0692), np.float32(703.70044), np.float32(1488.3016), np.float32(750.5238)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9825374484062195, 'coordinate': [np.float32(8.896561), np.float32(536.54895), np.float32(361.05237), np.float32(655.8058)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9822263717651367, 'coordinate': [np.float32(8.971573), np.float32(657.4949), np.float32(362.01715), np.float32(774.625)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9767460823059082, 'coordinate': [np.float32(9.407074), np.float32(776.5216), np.float32(361.31067), np.float32(846.82874)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9868153929710388, 'coordinate': [np.float32(8.669495), np.float32(848.2543), np.float32(361.64703), np.float32(1062.8568)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9826608300209045, 'coordinate': [np.float32(8.8025055), np.float32(1063.8615), np.float32(361.46588), np.float32(1182.8524)]}, {'cls_id': 22, 'label': 'text', 'score': 0.982555627822876, 'coordinate': [np.float32(8.820602), np.float32(1184.4663), np.float32(361.66394), np.float32(1302.4507)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9584776759147644, 'coordinate': [np.float32(9.170288), np.float32(1304.2161), np.float32(361.48898), np.float32(1351.7483)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9782056212425232, 'coordinate': [np.float32(389.1618), np.float32(200.38202), np.float32(742.7591), np.float32(295.65146)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9844875931739807, 'coordinate': [np.float32(388.73303), np.float32(297.18463), np.float32(744.00024), np.float32(441.3034)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9680547714233398, 'coordinate': [np.float32(409.39468), np.float32(455.89386), np.float32(721.7174), np.float32(520.9387)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9741666913032532, 'coordinate': [np.float32(389.71606), np.float32(536.8138), np.float32(742.7112), np.float32(608.00165)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9840384721755981, 'coordinate': [np.float32(389.30988), np.float32(609.39636), np.float32(743.09247), np.float32(750.3231)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9845995306968689, 'coordinate': [np.float32(389.13272), np.float32(751.7772), np.float32(743.058), np.float32(894.8815)]}, {'cls_id': 22, 'label': 'text', 'score': 0.984852135181427, 'coordinate': [np.float32(388.83267), np.float32(896.0371), np.float32(743.58215), np.float32(1038.7345)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9804865717887878, 'coordinate': [np.float32(389.08478), np.float32(1039.9119), np.float32(742.7585), np.float32(1134.4897)]}, {'cls_id': 22, 'label': 'text', 'score': 0.986461341381073, 'coordinate': [np.float32(388.52643), np.float32(1135.8137), np.float32(743.451), np.float32(1352.0085)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9869391918182373, 'coordinate': [np.float32(769.8341), np.float32(775.66235), np.float32(1124.9813), np.float32(1063.207)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9822869896888733, 'coordinate': [np.float32(770.30383), np.float32(1063.938), np.float32(1124.8295), np.float32(1184.2192)]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9689218997955322, 'coordinate': [np.float32(791.3042), np.float32(1199.3169), np.float32(1104.4521), np.float32(1264.6985)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9713128209114075, 'coordinate': [np.float32(770.4253), np.float32(1279.6072), np.float32(1124.6917), np.float32(1351.8672)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9236552119255066, 'coordinate': [np.float32(1153.9058), np.float32(775.5814), np.float32(1334.0654), np.float32(798.1581)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9857938885688782, 'coordinate': [np.float32(1151.5197), np.float32(799.28015), np.float32(1506.3619), np.float32(991.1156)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9820687174797058, 'coordinate': [np.float32(1151.5686), np.float32(991.91095), np.float32(1506.6023), np.float32(1110.8875)]}, {'cls_id': 22, 'label': 'text', 'score': 0.9866049885749817, 'coordinate': [np.float32(1151.6919), np.float32(1112.1301), np.float32(1507.1611), np.float32(1351.9504)]}]}}}
</code></pre></details>

For explanation of the result parameters, refer to [2.2 Python Script Integration](#222-python-script-integration).

<b>Note: </b> The default model for the production line is relatively large, which may result in slower inference speed. It is recommended to use inference acceleration frameworks to enhance VLM inference performance for faster inference.

### 2.2 Python Script Integration

The command line method is for quick testing and visualization. In actual projects, you usually need to integrate the model via code. You can perform pipeline inference with just a few lines of code as shown below:

```python
from paddleocr import PaddleOCRVL

pipeline = PaddleOCRVL()
# pipeline = PaddleOCRVL(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# pipeline = PaddleOCRVL(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# pipeline = PaddleOCRVL(use_layout_detection=False) # Use use_layout_detection to enable/disable layout detection module
output = pipeline.predict("./paddleocr_vl_demo.png")
for res in output:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the current image's structured result in JSON format
    res.save_to_markdown(save_path="output") ## Save the current image's result in Markdown format
```

For PDF files, each page will be processed individually and generate a separate Markdown file. If you want to convert the entire PDF to a single Markdown file, use the following method:

```python
from pathlib import Path
from paddleocr import PaddleOCRVL

input_file = "./your_pdf_file.pdf"
output_path = Path("./output")

pipeline = PaddleOCRVL()
output = pipeline.predict(input=input_file)

markdown_list = []
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)
```

**Note:**

- In the example code, the parameters `use_doc_orientation_classify` and  `use_doc_unwarping` are all set to `False` by default. These indicate that document orientation classification and document image unwarping are disabled. You can manually set them to `True` if needed.

The above Python script performs the following steps:

<details><summary>(1) Instantiate the production line object. Specific parameter descriptions are as follows:</summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tbody>
<tr>
<td> <code> layout_detection_model_name</code></td>
<td>Name of the layout area detection and ranking model. If set to <code> None</code>, the default model of the production line will be used.</td>
<td> <code> str|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> layout_detection_model_dir</code></td>
<td>Directory path of the layout area detection and ranking model. If set to <code> None</code>, the official model will be downloaded.</td>
<td> <code> str|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> layout_threshold</code></td>
<td>Score threshold for the layout model.<ul>
<li><b>float</b>: <code> Any floating-point number between</code>0-1</li>
<li><b>;</b>dict <code> :</code>{0:0.1}</li>
<li><b>The key is the class ID, and the value is the threshold for that class;</b>None <code> : If set to</code>None</li>
</ul>
</td>
<td> <code> , the parameter value initialized by the production line will be used.</code></td>
<td> <code> float|dict|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>layout_nms <code> Whether to use post-processing NMS for layout detection. If set to</code>None</td>
<td> <code> , the parameter value initialized by the production line will be used.</code></td>
<td> <code> bool|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>layout_unclip_ratio<ul>
<li><b>Expansion coefficient for the detection box of the layout area detection model.</b>float <code> : Any floating-point number greater than</code>0</li>
<li><b>;</b>Tuple[float,float]</li>
<li><b>: The respective expansion coefficients in the horizontal and vertical directions;</b>dict<b>, where the key of the dict is of</b>int <code> type, representing</code>cls_id<b>, and the value is of</b>tuple <code> type, such as</code>{0: (1.1, 2.0)}</li>
<li><b>, indicating that the center of the detection box for class 0 output by the model remains unchanged, with the width expanded by 1.1 times and the height expanded by 2.0 times;</b>None <code> : If set to</code>None</li>
</ul>
</td>
<td> <code> , the parameter value initialized by the production line will be used.</code></td>
<td> <code> float|Tuple[float,float]|dict|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>layout_merge_bboxes_mode<ul>
<li><b>Method for filtering overlapping boxes in layout area detection.</b>str <code> :</code>large <code> ,</code>small <code> ,</code>union</li>
<li><b>, indicating whether to retain the large box, small box, or both during overlapping box filtering;</b>dict<b>: The key of the dict is of</b>int <code> type, representing</code>cls_id<b>, and the value is of</b>str <code> type, such as</code>{0: "large", 2: "small"}</li>
<li><b>, indicating that the large mode is used for the detection box of class 0, and the small mode is used for the detection box of class 2;</b>None <code> : If set to</code>None</li>
</ul>
</td>
<td> <code> , the parameter value initialized by the production line will be used.</code></td>
<td> <code> str|dict|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>vl_rec_model_name <code> Name of the multimodal recognition model. If set to</code>None</td>
<td> <code> , the default model of the production line will be used.</code></td>
<td> <code> str|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>vl_rec_model_dir <code> Directory path of the multimodal recognition model. If set to</code>None</td>
<td> <code> , the official model will be downloaded.</code></td>
<td> <code> str|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>vl_rec_backend</td>
<td> <code> Inference backend used by the multimodal recognition model.</code></td>
<td> <code> int|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>vl_rec_server_url</td>
<td> <code> If the multimodal recognition model uses an inference service, this parameter is used to specify the server URL.</code></td>
<td> <code> str|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>vl_rec_max_concurrency</td>
<td> <code> If the multimodal recognition model uses an inference service, this parameter is used to specify the maximum number of concurrent requests.</code></td>
<td> <code> str|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>doc_orientation_classify_model_name <code> Name of the document orientation classification model. If set to</code>None</td>
<td> <code> , the default model of the production line will be used.</code></td>
<td> <code> str|None</code></td>
</tr>
<tr>
<td> <code> None</code></td>
<td>doc_orientation_classify_model_dir <code> Directory path of the document orientation classification model. If set to</code>None</td>
<td> <code> , the official model will be downloaded.</code></td>
<td> <code> str|None</code></td>
</tr>
<tr>
<td> <code> doc_unwarping_model_name</code></td>
<td>Name of the text image rectification model. If set to <code> None</code>, the default model of the production line will be used.</td>
<td> <code> str|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> doc_unwarping_model_dir</code></td>
<td>Directory path of the text image rectification model. If set to <code> None</code>, the official model will be downloaded.</td>
<td> <code> str|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> use_doc_orientation_classify</code></td>
<td>Whether to load and use the document orientation classification module. If set to <code> None</code>, the parameter value initialized by the production line will be used, and it is initialized to <code> False</code> by default.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> use_doc_unwarping</code></td>
<td>Whether to load and use the text image rectification module. If set to <code> None</code>, the parameter value initialized by the production line will be used, and it is initialized to <code> False</code> by default.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> use_layout_detection</code></td>
<td>Whether to load and use the layout area detection and sorting module. If set to <code> None</code>, the parameter value initialized by the production line will be used, and it is initialized to <code> True</code> by default.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> use_chart_recognition</code></td>
<td>Whether to load and use the chart parsing module. If set to <code> None</code>, the parameter value initialized by the production line will be used, and it is initialized to <code> False</code> by default.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> format_block_content</code></td>
<td>Controls whether to format the content in <code> block_content</code> into Markdown format. If set to <code> None</code>, the parameter value initialized by the production line will be used, and it is initialized to <code> False</code> by default.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> device</code></td>
<td>Device used for inference. Supports specifying specific card numbers:<ul>
<li><b>CPU</b>: For example, <code> cpu</code> indicates using the CPU for inference;</li>
<li><b>GPU</b>: For example, <code> gpu:0</code> indicates using the first GPU for inference;</li>
<li><b>NPU</b>: For example, <code> npu:0</code> indicates using the first NPU for inference;</li>
<li><b>XPU</b>: For example, <code> xpu:0</code> indicates using the first XPU for inference;</li>
<li><b>MLU</b>: For example, <code> mlu:0</code> indicates using the first MLU for inference;</li>
<li><b>DCU</b>: For example, <code> dcu:0</code> indicates using the first DCU for inference;</li>
<li><b>None</b>: If set to <code> None</code>, during initialization, the local GPU device 0 will be used preferentially. If not available, the CPU device will be used.</li>
</ul>
</td>
<td> <code> str|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> enable_hpi</code></td>
<td>Whether to enable high-performance inference.</td>
<td> <code> bool</code></td>
<td> <code> False</code></td>
</tr>
<tr>
<td> <code> use_tensorrt</code></td>
<td>Whether to enable the TensorRT subgraph engine of Paddle Inference. If the model does not support acceleration via TensorRT, acceleration will not be used even if this flag is set.<br/>For PaddlePaddle with CUDA 11.8, the compatible TensorRT version is 8.x (x&amp;gt;=6), and it is recommended to install TensorRT 8.6.1.6.<br/>
</td>
<td> <code> bool</code></td>
<td> <code> False</code></td>
</tr>
<tr>
<td> <code> precision</code></td>
<td>Computational precision, such as fp32, fp16.</td>
<td> <code> str</code></td>
<td> <code> "fp32"</code></td>
</tr>
<tr>
<td> <code> enable_mkldnn</code></td>
<td>Whether to enable MKL-DNN accelerated inference. If MKL-DNN is not available or the model does not support acceleration via MKL-DNN, acceleration will not be used even if this flag is set.</td>
<td> <code> bool</code></td>
<td> <code> True</code></td>
</tr>
<tr>
<td> <code> mkldnn_cache_capacity</code></td>
<td>MKL-DNN cache capacity.</td>
<td> <code> int</code></td>
<td> <code> 10</code></td>
</tr>
<tr>
<td> <code> cpu_threads</code></td>
<td>Number of threads used for inference on the CPU.</td>
<td> <code> int</code></td>
<td> <code> 8</code></td>
</tr>
<tr>
<td> <code> paddlex_config</code></td>
<td>Path to the PaddleX production line configuration file.</td>
<td> <code> str|None</code></td>
<td> <code> None</code></td>
</tr>
</tbody>
</table>
</details>
<details><summary>(2) Call the <code> predict()</code>method of the PaddleOCR-VL production line object for inference prediction. This method will return a list of results. Additionally, the production line also provides the <code> predict_iter()</code>Method. The two are completely consistent in terms of parameter acceptance and result return. The difference lies in that <code> predict_iter()</code>returns a <code> generator</code>, which can process and obtain prediction results step by step. It is suitable for scenarios involving large datasets or where memory conservation is desired. You can choose either of these two methods based on actual needs. Below are the parameters of the <code> predict()</code>method and their descriptions:</summary>
<table>
<thead>
<tr>
<th>Parameter</th>
<th>Parameter Description</th>
<th>Parameter Type</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td> <code> input</code></td>
<td>Data to be predicted, supporting multiple input types. Required.<ul>
<li><b>Python Var</b>: such as <code> numpy.ndarray</code>representing image data</li>
<li><b>str</b>: such as the local path of an image file or PDF file: <code> /root/data/img.jpg</code>;<b>such as a URL link</b>, such as the network URL of an image file or PDF file:<a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/demo_paper.png">Example</a>;<b>such as a local directory</b>, which should contain the images to be predicted, such as the local path: <code> /root/data/</code>(Currently, prediction for directories containing PDF files is not supported. PDF files need to be specified with a specific file path)</li>
<li><b>list</b>: List elements should be of the aforementioned data types, such as <code> [numpy.ndarray, numpy.ndarray]</code>, <code> ["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code> ["/root/data1", "/root/data2"].</code></li>
</ul>
</td>
<td> <code> Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td> <code> use_doc_orientation_classify</code></td>
<td>Whether to use the document orientation classification module during inference. Setting it to <code> None</code>means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> use_doc_unwarping</code></td>
<td>Whether to use the text image rectification module during inference. Setting it to <code> None</code>means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> use_layout_detection</code></td>
<td>Whether to use the layout region detection and sorting module during inference. Setting it to <code> None</code>means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> use_chart_recognition</code></td>
<td>Whether to use the chart parsing module during inference. Setting it to <code> None</code>means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> layout_threshold</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code> None</code>means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td> <code> float|dict|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> layout_nms</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code> None</code>means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> layout_unclip_ratio</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code> None</code>means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td> <code> float|Tuple[float,float]|dict|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> layout_merge_bboxes_mode</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code> None</code>means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td> <code> str|dict|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> use_queues</code></td>
<td>Used to control whether to enable internal queues. When set to <code> True</code>, data loading (such as rendering PDF pages as images), layout detection model processing, and VLM inference will be executed asynchronously in separate threads, with data passed through queues, thereby improving efficiency. This approach is particularly efficient for PDF documents with many pages or directories containing a large number of images or PDF files.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> prompt_label</code></td>
<td>The prompt type setting for the VL model, which takes effect only when <code> use_layout_detection=False</code>.</td>
<td> <code> str|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> format_block_content</code></td>
<td>The parameter meaning is basically the same as the instantiation parameter. Setting it to <code> None</code>means using the instantiation parameter; otherwise, this parameter takes precedence.</td>
<td> <code> bool|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> repetition_penalty</code></td>
<td>The repetition penalty parameter used for VL model sampling.</td>
<td> <code> float|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> temperature</code></td>
<td>Temperature parameter used for VL model sampling.</td>
<td> <code> float|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> top_p</code></td>
<td>Top-p parameter used for VL model sampling.</td>
<td> <code> float|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> min_pixels</code></td>
<td>The minimum number of pixels allowed when the VL model preprocesses images.</td>
<td> <code> int|None</code></td>
<td> <code> None</code></td>
</tr>
<tr>
<td> <code> max_pixels</code></td>
<td>The maximum number of pixels allowed when the VL model preprocesses images.</td>
<td> <code> int|None</code></td>
<td> <code> None</code></td>
</tr>
</table>
</details>
<details><summary>(3) Process the prediction results: The prediction result for each sample is a corresponding Result object, supporting operations such as printing, saving as an image, and saving as a <code> json</code>file:</summary>
<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameter</th>
<th>Parameter Type</th>
<th>Parameter Description</th>
<th>Default Value</th>
</tr>
</thead>
<tr>
<td rowspan="3"> <code> print()</code></td>
<td rowspan="3">Print results to the terminal</td>
<td> <code> format_json</code></td>
<td> <code> bool</code></td>
<td>Whether to format the output content using <code> JSON</code>indentation.</td>
<td> <code> True</code></td>
</tr>
<tr>
<td> <code> indent</code></td>
<td> <code> int</code></td>
<td>Specify the indentation level to beautify the output <code> JSON</code>data, making it more readable. Only valid when <code> format_json</code>is <code> True</code>.</td>
<td>4</td>
</tr>
<tr>
<td> <code> ensure_ascii</code></td>
<td> <code> bool</code></td>
<td>Control whether non- <code> ASCII</code>characters are escaped as <code> Unicode</code>. When set to <code> True</code>, all non- <code> ASCII</code>characters will be escaped; <code> False</code>retains the original characters. Only valid when <code> format_json</code>is <code> True</code>.</td>
<td> <code> False</code></td>
</tr>
<tr>
<td rowspan="3"> <code> save_to_json()</code></td>
<td rowspan="3">Save the results as a json format file</td>
<td> <code> save_path</code></td>
<td> <code> str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file type naming.</td>
<td>None</td>
</tr>
<tr>
<td> <code> indent</code></td>
<td> <code> int</code></td>
<td>Specify the indentation level to beautify the output <code> JSON</code>data, making it more readable. Only valid when <code> format_json</code>is <code> True</code>.</td>
<td>4</td>
</tr>
<tr>
<td> <code> ensure_ascii</code></td>
<td> <code> bool</code></td>
<td>Control whether non- <code> ASCII</code>characters are escaped as <code> Unicode</code>. When set to <code> True</code>, all non- <code> ASCII</code>characters will be escaped; <code> False</code>retains the original characters. Only valid when <code> format_json</code>is <code> True</code>.</td>
<td> <code> False</code></td>
</tr>
<tr>
<td> <code> save_to_img()</code></td>
<td>Save the visualized images of each intermediate module in png format</td>
<td> <code> save_path</code></td>
<td> <code> str</code></td>
<td>The file path for saving, supporting directory or file paths.</td>
<td>None</td>
</tr>
<tr>
<td rowspan="3"> <code> save_to_markdown()</code></td>
<td rowspan="3">Save each page in an image or PDF file as a markdown format file separately</td>
<td> <code> save_path</code></td>
<td> <code> str</code></td>
<td>The file path for saving. When it is a directory, the saved file name will be consistent with the input file type naming</td>
<td>None</td>
</tr>
<tr>
<td> <code> pretty</code></td>
<td> <code> bool</code></td>
<td>Whether to beautify the <code> markdown</code>output results, centering charts, etc., to make the <code> markdown</code>rendering more aesthetically pleasing.</td>
<td>True</td>
</tr>
<tr>
<td> <code> show_formula_number</code></td>
<td> <code> bool</code></td>
<td>Control whether to retain formula numbers in <code> markdown</code>. When set to <code> True</code>, all formula numbers are retained; <code> False</code>retains only the formulas</td>
<td> <code> False</code></td>
</tr>
<tr>
<tr>
<td> <code> save_to_html()</code></td>
<td>Save the tables in the file as html format files</td>
<td> <code> save_path</code></td>
<td> <code> str</code></td>
<td>The file path for saving, supporting directory or file paths.</td>
<td>None</td>
</tr>
<tr>
<td> <code> save_to_xlsx()</code></td>
<td>Save the tables in the file as xlsx format files</td>
<td> <code> save_path</code></td>
<td> <code> str</code></td>
<td>The file path for saving, supporting directory or file paths.</td>
<td>None</td>
</tr>
</tr></table>- Calling the `print()` method will print the results to the terminal. The content printed to the terminal is explained as follows:
    - `input_path`: `(str)` The input path of the image or PDF to be predicted.

    - `page_index`: `(Union[int, None])` If the input is a PDF file, it indicates the current page number of the PDF; otherwise, it is `None`.

    - `model_settings`: `(Dict[str, bool])` Model parameters required for configuring the production line.

        - `use_doc_preprocessor`: `(bool)` Controls whether to enable the document preprocessing sub-production line.
        - `use_seal_recognition`: `(bool)` Controls whether to enable the seal text recognition sub-production line.
        - `use_table_recognition`: `(bool)` Controls whether to enable the table recognition sub-production line.
        - `use_formula_recognition`: `(bool)` Controls whether to enable the formula recognition sub-production line.

    - `doc_preprocessor_res`: `(Dict[str, Union[List[float], str]])` A dictionary of document preprocessing results, which exists only when `use_doc_preprocessor=True`.
        - `input_path`: `(str)` The image path accepted by the document preprocessing sub-production line. When the input is a `numpy.ndarray`, it is saved as `None`. Here, it is `None`.
        - `page_index`: `None`. Since the input here is a `numpy.ndarray`, the value is `None`.
        - `model_settings`: `(Dict[str, bool])` Model configuration parameters for the document preprocessing sub-production line.
          - `use_doc_orientation_classify`: `(bool)` Controls whether to enable the document image orientation classification sub-module.
          - `use_doc_unwarping`: `(bool)` Controls whether to enable the text image distortion correction sub-module.
        - `angle`: `(int)` The prediction result of the document image orientation classification sub-module. When enabled, the actual angle value is returned.

    - `parsing_res_list`: `(List[Dict])` A list of parsing results, where each element is a dictionary. The order of the list is the reading order after parsing.
        - `block_bbox`: `(np.ndarray)` The bounding box of the layout area.
        - `block_label`: `(str)` The label of the layout area, such as `text`, `table`, etc.
        - `block_content`: `(str)` The content within the layout area.
        - `block_id`: `(int)` The index of the layout area, used to display the layout sorting results.
        - `block_order` `(int)` The order of the layout area, used to display the layout reading order. For non-sorted parts, the default value is `None`.

    - `overall_ocr_res`: `(Dict[str, Union[List[str], List[float], numpy.ndarray]])` A dictionary of global OCR results.
      - `input_path`: `(Union[str, None])` The image path accepted by the image OCR sub-production line. When the input is a `numpy.ndarray`, it is saved as `None`.
      - `page_index`: `None`. Since the input here is a `numpy.ndarray`, the value is `None`.
      - `model_settings`: `(Dict)` Model configuration parameters for the OCR sub-production line.
      - `dt_polys`: `(List[numpy.ndarray])` A list of polygonal bounding boxes for text detection. Each detection box is represented by a numpy array consisting of the coordinates of 4 vertices, with an array shape of (4, 2) and a data type of int16.
      - `dt_scores`: `(List[float])` A list of confidence scores for text detection boxes.
      - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the text detection module.
        - `limit_side_len`: `(int)` The side length limit value during image preprocessing.
        - `limit_type`: `(str)` The processing method for the side length limit.
        - `thresh`: `(float)` The confidence threshold for text pixel classification.
        - `box_thresh`: `(float)` The confidence threshold for text detection boxes.
        - `unclip_ratio`: `(float)` The expansion coefficient for text detection boxes.
        - `text_type`: `(str)` The type of text detection, currently fixed as "general".

      - `text_type`: `(str)` The type of text detection, currently fixed as "general".
      - `textline_orientation_angles`: `(List[int])` The prediction results of text line orientation classification. When enabled, the actual angle values are returned (e.g., [0,0,1]).
      - `text_rec_score_thresh`: `(float)` The filtering threshold for text recognition results.
      - `rec_texts`: `(List[str])` A list of text recognition results, containing only texts with confidence scores exceeding `text_rec_score_thresh`.
      - `rec_scores`: `(List[float])` List of confidence scores for text recognition, filtered by `text_rec_score_thresh`
- `rec_polys`: `(List[numpy.ndarray])` List of text detection boxes filtered by confidence, with the same format as `dt_polys`

- `formula_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of formula recognition results, with each element being a dict
    - `rec_formula`: `(str)` Formula recognition result
    - `rec_polys`: `(numpy.ndarray)` Formula detection box, with shape (4, 2) and dtype int16
    - `formula_region_id`: `(int)` Region number where the formula is located

- `seal_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of seal text recognition results, with each element being a dict
    - `input_path`: `(str)` Input path of the seal image
    - `page_index`: `None`, as the input here is `numpy.ndarray`, so the value is `None`
    - `model_settings`: `(Dict)` Model configuration parameters for the seal text recognition sub-pipeline
    - `dt_polys`: `(List[numpy.ndarray])` List of seal detection boxes, with the same format as `dt_polys`
    - `text_det_params`: `(Dict[str, Dict[str, int, float]])` Configuration parameters for the seal detection module, with specific parameter meanings as above
    - `text_type`: `(str)` Type of seal detection, currently fixed as "seal"
    - `text_rec_score_thresh`: `(float)` Filtering threshold for seal text recognition results
    - `rec_texts`: `(List[str])` List of seal text recognition results, containing only texts with confidence exceeding `text_rec_score_thresh`
    - `rec_scores`: `(List[float])` List of confidence scores for seal text recognition, filtered by `text_rec_score_thresh`
    - `rec_polys`: `(List[numpy.ndarray])` List of seal detection boxes filtered by confidence, with the same format as `dt_polys`
    - `rec_boxes`: `(numpy.ndarray)` Array of rectangular bounding boxes for detection boxes, with shape (n, 4) and dtype int16. Each row represents a rectangle

- `table_res_list`: `(List[Dict[str, Union[numpy.ndarray, List[float], str]]])` List of table recognition results, with each element being a dict
    - `cell_box_list`: `(List[numpy.ndarray])` List of bounding boxes for table cells
    - `pred_html`: `(str)` HTML format string of the table
    - `table_ocr_pred`: `(dict)` OCR recognition results for the table
        - `rec_polys`: `(List[numpy.ndarray])` List of detection boxes for cells
        - `rec_texts`: `(List[str])` Recognition results for cells
        - `rec_scores`: `(List[float])` Recognition confidence scores for cells
        - `rec_boxes`: `(numpy.ndarray)` Array of rectangular bounding boxes for detection boxes, with shape (n, 4) and dtype int16. Each row represents a rectangle

- Calling the `save_to_json()` method will save the above content to the specified `save_path`. If a directory is specified, the saved path will be `save_path/{your_img_basename}_res.json`. If a file is specified, it will be saved directly to that file. Since json files do not support saving numpy arrays, the `numpy.array` types within will be converted to list form.
- Calling the `save_to_img()` method will save the visualization results to the specified `save_path`. If a directory is specified, visualized images for layout region detection, global OCR, layout reading order, etc., will be saved. If a file is specified, it will be saved directly to that file. (Production lines typically contain many result images, so it is not recommended to directly specify a specific file path, as multiple images will be overwritten, retaining only the last one.)
- Calling the `save_to_markdown()` method will save the converted Markdown file to the specified `save_path`. The saved file path will be `save_path/{your_img_basename}.md`. If the input is a PDF file, it is recommended to directly specify a directory; otherwise, multiple markdown files will be overwritten.

Additionally, it also supports obtaining visualized images and prediction results with results through attributes, as follows:<table>
<thead>
<tr>
<th>Attribute</th>
<th>Attribute Description</th>
</tr>
</thead>
<tbody>
<tr>
<td> <code> json</code></td>
<td>Obtain the prediction <code> json</code>result in the format</td>
</tr>
<tr>
<td rowspan="2"> <code> img</code></td>
<td rowspan="2">obtain in the format of <code> dict</code>visualized image</td>
</tr>
<tr>
</tr>
<tr>
<td rowspan="3"> <code> markdown</code></td>
<td rowspan="3">obtain in the format of <code> dict</code>markdown result</td>
</tr>
<tr>
</tr>
<tr>
</tr>
</tbody>
</table>- The prediction result obtained through the `json` attribute is data of dict type, with relevant content consistent with that saved by calling the `save_to_json()` method.
- The prediction result returned by the `img` attribute is data of dict type. The keys are `layout_det_res`, `overall_ocr_res`, `text_paragraphs_ocr_res`, `formula_res_region1`, `table_cell_img`, and `seal_res_region1`, with corresponding values being `Image.Image` objects: used to display visualized images of layout region detection, OCR, OCR text paragraphs, formulas, tables, and seal results, respectively. If optional modules are not used, the dict only contains `layout_det_res`.
- The prediction result returned by the `markdown` attribute is data of dict type. The keys are `markdown_texts`, `markdown_images`, and `page_continuation_flags`, with corresponding values being markdown text, images displayed in Markdown (`Image.Image` objects), and a bool tuple used to identify whether the first element on the current page is the start of a paragraph and whether the last element is the end of a paragraph, respectively.</details>

## 3. Enhancing VLM Inference Performance Using Inference Acceleration Frameworks

The inference performance under the default configuration has not been fully optimized and may not meet actual production requirements. PaddleOCR supports enhancing the inference performance of VLM through inference acceleration frameworks such as vLLM and SGLang, thereby accelerating the inference speed in production lines. The usage process mainly consists of two steps:

1. Start the VLM inference service;
2. Configure the PaddleOCR production line to invoke the VLM inference service as a client.

### 3.1 Starting the VLM Inference Service

#### 3.1.1 Using Docker Images

PaddleOCR provides Docker images for quickly starting the vLLM inference service. The service can be started using the following command:

```bash
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server
```

The service listens on port **8080** by default.

When starting the container, you can pass in parameters to override the default configuration. The parameters are consistent with the `paddleocr genai_server` command (see the next subsection for details). For example:

```bash
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server \
    paddlex_genai_server --model_name PaddleOCR-VL-0.9B --host 0.0.0.0 --port 8118 --backend vllm
```

If you are using an NVIDIA 50 series graphics card (Compute Capacity >= 12), you need to install a specific version of FlashAttention before launching the service.

```
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddlex-genai-vllm-server \
    /bin/bash
python -m pip install flash-attn==2.8.3
paddlex_genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8118
```

#### 3.1.2 Installation and Usage via PaddleOCR CLI

Since the inference acceleration framework may have dependency conflicts with the PaddlePaddle framework, it is recommended to install it in a virtual environment. Taking vLLM as an example:

```bash
# Create a virtual environment
python -m venv .venv
# Activate the environment
source .venv/bin/activate
# Install PaddleOCR
python -m pip install "paddleocr[doc-parser]"
# Install dependencies for inference acceleration service
paddleocr install_genai_server_deps vllm
```

Usage of the `paddleocr install_genai_server_deps` command:

```bash
paddleocr install_genai_server_deps <name of the inference acceleration framework>
```

The currently supported frameworks are named `vllm` and `sglang`, corresponding to vLLM and SGLang, respectively.

If you are using an NVIDIA 50 series graphics card (Compute Capacity >= 12), you need to install a specific version of FlashAttention before launching the service.

```
python -m pip install flash-attn==2.8.3
```

After installation, you can start the service using the `paddleocr genai_server` command:

```bash
paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --port 8118
```

The parameters supported by this command are as follows:

| Parameter          | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `--model_name`     | Model name                                                    |
| `--model_dir`      | Model directory                                               |
| `--host`           | Server hostname                                               |
| `--port`           | Server port number                                            |
| `--backend`        | Backend name, i.e., the name of the inference acceleration framework used. Options are `vllm` or `sglang`. |
| `--backend_config` | A YAML file can be specified, which contains backend configurations. |

### 3.2 How to Use the Client

After starting the VLM inference service, the client can invoke the service through PaddleOCR.

#### 3.2.1 CLI Invocation

The backend type (`vllm-server` or `sglang-server`) can be specified via `--vl_rec_backend`, and the service address can be specified via `--vl_rec_server_url`. For example:

```bash
paddleocr doc_parser --input paddleocr_vl_demo.png --vl_rec_backend vllm-server --vl_rec_server_url http://127.0.0.1:8118/v1
```

#### 3.2.2 Python API Invocation

Pass the `vl_rec_backend` and `vl_rec_server_url` parameters when creating the `PaddleOCRVL` object:

```python
pipeline = PaddleOCRVL(vl_rec_backend="vllm-server", vl_rec_server_url="http://127.0.0.1:8118/v1")
```

#### 3.2.3 Service-Oriented Deployment

The fields `VLRecognition.genai_config.backend` and `VLRecognition.genai_config.server_url` can be modified in the configuration file, for example:

```yaml
VLRecognition:
  ...
  genai_config:
    backend: vllm-server
    server_url: http://127.0.0.1:8118/v1
```

### 3.3 Performance Tuning

The default configuration is tuned on a single NVIDIA A100 and assumes exclusive client service, so it may not be suitable for other environments. If users encounter performance issues during actual use, they can try the following optimization methods.

#### 3.3.1 Server-side Parameter Adjustment

Different inference acceleration frameworks support different parameters. Refer to their respective official documentation to learn about available parameters and when to adjust them:

- [vLLM Official Parameter Tuning Guide](https://docs.vllm.ai/en/latest/configuration/optimization.html)
- [SGLang Hyperparameter Tuning Documentation](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html)

The PaddleOCR VLM inference service supports parameter tuning through configuration files. The following example demonstrates how to adjust the `gpu-memory-utilization` and `max-num-seqs` parameters of the vLLM server:

1. Create a YAML file named `vllm_config.yaml` with the following content:

```yaml
gpu-memory-utilization: 0.3
   max-num-seqs: 128


2. Specify the configuration file path when starting the service, for example, using the `paddleocr genai_server` command:

```bash
paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --backend_config vllm_config.yaml


If you are using a shell that supports process substitution (such as Bash), you can also pass configuration items directly when starting the service without creating a configuration file:

```bash
paddleocr genai_server --model_name PaddleOCR-VL-0.9B --backend vllm --backend_config <(echo -e 'gpu-memory-utilization: 0.3\nmax-num-seqs: 128')
```

#### 3.3.2 Client-Side Parameter Adjustment

PaddleOCR groups sub-images from single or multiple input images and initiates concurrent requests to the server. Therefore, the number of concurrent requests significantly impacts performance.

- For the CLI and Python API, the maximum number of concurrent requests can be adjusted using the `vl_rec_max_concurrency` parameter.
- For service-based deployment, modify the `VLRecognition.genai_config.max_concurrency` field in the configuration file.

When there is a one-to-one correspondence between the client and the VLM inference service, and the server-side resources are sufficient, the number of concurrent requests can be appropriately increased to enhance performance. If the server needs to support multiple clients or has limited computational resources, the number of concurrent requests should be reduced to prevent service abnormalities caused by resource overload.

#### 3.3.3 Recommendations for Performance Tuning on Common Hardware

The following configurations are tailored for scenarios with a one-to-one correspondence between the client and the VLM inference service.

**NVIDIA RTX 3060**

- **Server-Side**
  - vLLM: `gpu-memory-utilization=0.8`
