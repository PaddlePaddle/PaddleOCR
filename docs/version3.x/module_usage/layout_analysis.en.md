---

comments: true

---



# Layout Analysis Module User Guide



## 1. Overview



Layout analysis is a crucial component in document parsing systems. Its goal is to parse the overall layout of a document, accurately detecting various elements it contains (such as text paragraphs, headings, images, tables, formulas, etc.), and restoring the correct reading order of these elements. The performance of this module directly impacts the overall accuracy and usability of the document parsing system.



## 2. Supported Model List



Currently, this module only supports the PP-DocLayoutV2 model. Structurally, PP-DocLayoutV2 is based on the layout detection model [PP-DocLayout_plus-L](./layout_detection.en.md) (built upon the RT-DETR-L model) and cascades a lightweight pointer network with 6 Transformer layers. The PP-DocLayout_plus-L component continues to perform layout detection, identifying different elements in document images (such as text, charts, images, formulas, paragraphs, abstracts, references, etc.), classifying them into predefined categories, and determining their positions within the document. The detected bounding boxes and class labels are then fed into the subsequent pointer network to sort the layout elements and obtain the correct reading order.



<div align="center">

<img src="https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/refs/heads/main/images/paddleocr_vl/methods/PP-DocLayoutV2.png" width="600"/>

</div>



As shown in the figure above, PP-DocLayoutV2 embeds the targets detected by RT-DETR using absolute 2D positional encoding and class labels. Additionally, the pointer network’s attention mechanism incorporates the geometric bias mechanism from Relation-DETR to explicitly model pairwise geometric relationships between elements. The pairwise relation head linearly projects element representations into query and key vectors, then computes bilinear similarities to generate pairwise logits, ultimately producing an N×N matrix representing the relative order between every pair of elements. Finally, a deterministic "win-accumulation" decoding algorithm restores a topologically consistent reading order for the detected layout elements.



The following table only presents the layout detection accuracy of PP-DocLayoutV2. The evaluation dataset is a self-built layout region detection dataset, containing 1,000 images of various document types such as Chinese and English papers, magazines, newspapers, research reports, PPTs, exam papers, textbooks, etc., and covering 25 common layout element categories: document title, section header, text, vertical text, page number, abstract, table of contents, references, footnote, image caption, header, footer, header image, footer image, algorithm, inline formula, display formula, formula number, image, table, figure title (figure title, table title, chart title), seal, chart, aside text, and reference content.



<table>

<thead>

<tr>

<th>Model</th><th>Model Download Link</th>

<th>mAP(0.5) (%)</th>

<th>GPU Inference Time (ms)<br/>[Standard / High-Performance Mode]</th>

<th>CPU Inference Time (ms)<br/>[Standard / High-Performance Mode]</th>

<th>Model Size (MB)</th>

<th>Description</th>

</tr>

</thead>

<tbody>

<tr>

<td>PP-DocLayoutV2</td>

<td><a href="https://huggingface.co/PaddlePaddle/PP-DocLayoutV2">Inference Model</a></td>

<td>81.4</td>

<td> - </td>

<td> - </td>

<td>203.8</td>

<td>In-house layout analysis model trained on a diverse self-built dataset including Chinese and English academic papers, multi-column magazines, newspapers, PPTs, contracts, books, exam papers, research reports, ancient books, Japanese documents, and vertical text documents. Provides high-precision layout region localization and reading order recovery.</td>

</tr>

<tr>

</tbody>

</table>

## III. Quick Integration  <a id="quick"> </a>

> ❗ Before quick integration, please install the PaddleOCR wheel package. For detailed instructions, refer to [PaddleOCR Local Installation Tutorial](../installation.en.md)。



<b>Note: </b>The official models would be download from HuggingFace by default. If can't access to HuggingFace, please set the environment variable `PADDLE_PDX_MODEL_SOURCE="BOS"` to change the model source to BOS. In the future, more model sources will be supported.

You can also integrate the model inference from the layout area detection module into your project. Before running the following code, please download [Example Image](https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/layout.jpg) Go to the local area.

```python
from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayoutV2")
output = model.predict("layout.jpg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the result obtained is:

```bash
{'res': {'input_path': 'layout.jpg', 'page_index': None, 'boxes': [{'cls_id': 7, 'label': 'figure_title', 'score': 0.9764086604118347, 'coordinate': [34.227077, 19.217348, 362.48834, 80.243195]}, {'cls_id': 21, 'label': 'table', 'score': 0.9881672263145447, 'coordinate': [73.759026, 105.72034, 322.67398, 298.82642]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9405199885368347, 'coordinate': [35.089825, 330.67575, 144.06467, 346.7406]}, {'cls_id': 22, 'label': 'text', 'score': 0.9889925122261047, 'coordinate': [33.540825, 349.37204, 363.5848, 615.04504]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.9430180788040161, 'coordinate': [35.032074, 627.2185, 188.24695, 643.66693]}, {'cls_id': 22, 'label': 'text', 'score': 0.988863468170166, 'coordinate': [33.72388, 646.5885, 363.05005, 852.60236]}, {'cls_id': 7, 'label': 'figure_title', 'score': 0.9749509692192078, 'coordinate': [385.14465, 19.341825, 715.4594, 78.739395]}, {'cls_id': 21, 'label': 'table', 'score': 0.9876241683959961, 'coordinate': [436.68512, 105.67539, 664.1041, 314.0018]}, {'cls_id': 22, 'label': 'text', 'score': 0.9878363013267517, 'coordinate': [385.06494, 345.74847, 715.173, 463.18677]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.951842725276947, 'coordinate': [386.3095, 476.34277, 703.0732, 493.00302]}, {'cls_id': 22, 'label': 'text', 'score': 0.9884033799171448, 'coordinate': [385.1237, 496.70123, 716.09717, 702.33386]}, {'cls_id': 17, 'label': 'paragraph_title', 'score': 0.940105676651001, 'coordinate': [386.66754, 715.5732, 527.5737, 731.77826]}, {'cls_id': 22, 'label': 'text', 'score': 0.9876392483711243, 'coordinate': [384.9688, 734.457, 715.63086, 853.71]}]}}
```

The meanings of the parameters are as follows:
- `input_path`: The path to the input image for prediction.
- `page_index`: If the input is a PDF file, it indicates which page of the PDF it is; otherwise, it is `None`.
- `boxes`: Predicted layout element information sorted in reading order, represented as a list of dictionaries. According to the reading order, each dictionary represents a detected layout element and contains the following information:
  - `cls_id`: Class ID, an integer.
  - `label`: Class label, a string.
  - `score`: Confidence score of the bounding box, a float.
  - `coordinate`: Coordinates of the bounding box, a list of floats in the format <code>[xmin, ymin, xmax, ymax]</code>.



Relevant methods, parameters, and explanations are as follows:

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
<td><code>model_name</code></td>
<td>Model name. If set to <code>None</code>, <code>PP-DocLayout-L</code> will be used.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>model_dir</code></td>
<td>Model storage path.</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>device</code></td>
<td>Device for inference.<br/>
<b>For example:</b> <code>"cpu"</code>, <code>"gpu"</code>, <code>"npu"</code>, <code>"gpu:0"</code>, <code>"gpu:0,1"</code>.<br/>
If multiple devices are specified, parallel inference will be performed.<br/>
By default, GPU 0 is used if available; otherwise, CPU is used.
</td>
<td><code>str|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>enable_hpi</code></td>
<td>Whether to enable high-performance inference.</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>use_tensorrt</code></td>
<td>Whether to use the Paddle Inference TensorRT subgraph engine. If the model does not support acceleration through TensorRT, setting this flag will not enable acceleration.<br/>
For Paddle with CUDA version 11.8, the compatible TensorRT version is 8.x (x>=6), and it is recommended to install TensorRT 8.6.1.6.<br/>

</td>
<td><code>bool</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>precision</code></td>
<td>Computation precision when using the TensorRT subgraph engine in Paddle Inference.<br/><b>Options:</b> <code>"fp32"</code>, <code>"fp16"</code>.</td>
<td><code>str</code></td>
<td><code>"fp32"</code></td>
</tr>
<tr>
<td><code>enable_mkldnn</code></td>
<td>
Whether to enable MKL-DNN acceleration for inference. If MKL-DNN is unavailable or the model does not support it, acceleration will not be used even if this flag is set.
</td>
<td><code>bool</code></td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>mkldnn_cache_capacity</code></td>
<td>
MKL-DNN cache capacity.
</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>cpu_threads</code></td>
<td>Number of threads to use for inference on CPUs.</td>
<td><code>int</code></td>
<td><code>10</code></td>
</tr>
<tr>
<td><code>img_size</code></td>
<td>Input image size.<ul>
<li><b>int</b>: e.g. <code>640</code>, resizes input image to 640x640.</li>
<li><b>list</b>: e.g. <code>[640, 512]</code>, resizes input image to width 640 and height 512.</li>
</ul>
</td>
<td><code>int|list|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Threshold for filtering low-confidence predictions.<ul>
<li><b>float</b>: e.g. <code>0.2</code>, filters out all boxes with confidence below 0.2.</li>
<li><b>dict</b>: The key is <code>int</code> (class id), the value is <code>float</code> (threshold). For example, <code>{0: 0.45, 2: 0.48, 7: 0.4}</code> means class 0 uses threshold 0.45, class 2 uses 0.48, class 7 uses 0.4.</li>
<li><b>None</b>: uses the model's default configuration.</li>
</ul>
</td>
<td><code>float|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Whether to use NMS post-processing to filter overlapping boxes.
<ul>
<li><b>bool</b>: whether to use NMS for post-processing to filter overlapping boxes.</li>
<li><b>None</b>: uses the model's default configuration.</li>
</ul>
</td>
<td><code>bool|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Scaling factor for the side length of the detection box.<ul>
<li><b>float</b>: A float greater than 0, e.g. <code>1.1</code>, expands width and height by 1.1 times.</li>
<li><b>list</b>: e.g. <code>[1.2, 1.5]</code>, expands width by 1.2x and height by 1.5x.</li>
<li><b>dict</b>: The key is <code>int</code> (class id), the value is <code>tuple</code> of two floats (width ratio, height ratio). For example, <code>{0: (1.1, 2.0)}</code> means for class 0, width is expanded by 1.1x and height by 2.0x.</li>
<li><b>None</b>: uses the model's default configuration.</li>
</ul>
</td>
<td><code>float|list|dict|None</code></td>
<td><code>None</code></td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Merge mode for model output bounding boxes.<ul>
<li><b>"large"</b>: Only keep the largest outer box among overlapping boxes, remove inner boxes.</li>
<li><b>"small"</b>: Only keep the smallest inner box among overlapping boxes, remove outer boxes.</li>
<li><b>"union"</b>: Keep all boxes, no filtering.</li>
<li><b>dict</b>: The key is <code>int</code> (class id), the value is <code>str</code> (mode). For example, <code>{0: "large", 2: "small"}</code> means class 0 uses "large" mode, class 2 uses "small" mode.</li>
<li><b>None</b>: Use the model's default configuration.</li>
</ul>
</td>
<td><code>str|dict|None</code></td>
<td><code>None</code></td>
</tr>
</tbody>
</table>

* The `predict()` method of the target detection model is called for inference prediction. The parameters of the `predict()` method are `input`, `batch_size`, and `threshold`, which are explained as follows:

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
<td><code>input</code></td>
<td>Input data to be predicted. Required. Supports multiple input types:<ul>
<li><b>Python Var</b>: e.g., <code>numpy.ndarray</code> representing image data</li>
<li><b>str</b>: 
  - Local image or PDF file path: <code>/root/data/img.jpg</code>;
  - <b>URL</b> of image or PDF file: e.g., <a href="https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/img_rot180_demo.jpg">example</a>;
  - <b>Local directory</b>: directory containing images for prediction, e.g., <code>/root/data/</code> (Note: directories containing PDF files are not supported; PDFs must be specified by exact file path)</li>
<li><b>list</b>: Elements must be of the above types, e.g., <code>[numpy.ndarray, numpy.ndarray]</code>, <code>["/root/data/img1.jpg", "/root/data/img2.jpg"]</code>, <code>["/root/data1", "/root/data2"]</code></li>
</ul>
</td>
<td><code>Python Var|str|list</code></td>
<td></td>
</tr>
<tr>
<td><code>batch_size</code></td>
<td>Batch size, positive integer.</td>
<td><code>int</code></td>
<td>1</td>
</tr>
<tr>
<td><code>threshold</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_nms</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>bool|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_unclip_ratio</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>float|list|dict|None</code></td>
<td>None</td>
</tr>
<tr>
<td><code>layout_merge_bboxes_mode</code></td>
<td>Same meaning as the instantiation parameters. If set to <code>None</code>, the instantiation value is used; otherwise, this parameter takes precedence.</td>
<td><code>str|dict|None</code></td>
<td>None</td>
</tr>
</tbody>
</table>


* Process the prediction results, with each sample's prediction result being the corresponding Result object, and supporting operations such as printing, saving as an image, and saving as a 'json' file:

<table>
<thead>
<tr>
<th>Method</th>
<th>Method Description</th>
<th>Parameters</th>
<th>Parameter type</th>
<th>Parameter Description</th>
<th>Default value</th>
</tr>
</thead>
<tr>
<td rowspan="3"><code>print()</code></td>
<td rowspan="3">Print the result to the terminal</td>
<td><code>format_json</code></td>
<td><code>bool</code></td>
<td>Do you want to use <code>JSON</code> indentation formatting for the output content</td>
<td><code>True</code></td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to enhance the readability of the <code>JSON</code> data output, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non ASCII characters to Unicode characters. When set to <code>True</code>, all non ASCII </code>characters will be escaped; <code>False</code> preserves the original characters and is only valid when <code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td rowspan="3"><code>save_to_json()</code></td>
<td rowspan="3">Save the result as a JSON format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The saved file path, when it is a directory, the name of the saved file is consistent with the name of the input file type</td>
<td>None</td>
</tr>
<tr>
<td><code>indent</code></td>
<td><code>int</code></td>
<td>Specify the indentation level to enhance the readability of the <code>JSON</code> data output, only valid when <code>format_json</code> is <code>True</code></td>
<td>4</td>
</tr>
<tr>
<td><code>ensure_ascii</code></td>
<td><code>bool</code></td>
<td>Control whether to escape non ASCII characters to Unicode characters. When set to <code>True</code>, all non <code>ASCII</code> characters will be escaped; <code>False</code> preserves the original characters and is only valid when<code>format_json</code> is <code>True</code></td>
<td><code>False</code></td>
</tr>
<tr>
<td><code>save_to_img()</code></td>
<td>Save the results as an image format file</td>
<td><code>save_path</code></td>
<td><code>str</code></td>
<td>The saved file path, when it is a directory, the name of the saved file is consistent with the name of the input file type</td>
<td>None</td>
</tr>
</table>


* Additionally, it also supports obtaining the visualized image with results and the prediction results via attributes, as follows:

<table>
<thead>
<tr>
<th>Attribute</th>
<th>Description</th>
</tr>
</thead>
<tr>
<td rowspan="1"><code>json</code></td>
<td rowspan="1">Get the prediction result in <code>json</code> format</td>
</tr>
<tr>
<td rowspan="1"><code>img</code></td>
<td rowspan="1">Get the visualized image in <code>dict</code> format</td>
</tr>
</table>
