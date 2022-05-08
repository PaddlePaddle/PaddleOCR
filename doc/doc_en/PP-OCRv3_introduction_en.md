English | [简体中文](../doc_ch/PP-OCRv3_introduction.md)




<a name="2"></a>
## 2. Text Detection Optimization

The PP-OCRv3 detection model is an upgrade of the [CML](https://arxiv.org/pdf/2109.03144.pdf) (Collaborative Mutual Learning) text detection distillation strategy in PP-OCRv2. As shown in the figure below, the core idea of CML combines ① the distillation of the traditional Teacher to guide Students and ② the DML mutual learning between the Students network, which allows the Students network to learn from each other. PP-OCRv3 further optimizes the effect of teacher model and student model respectively. For the Teacher model, 
we propose a PAN structure LKPAN with a larger receptive field and use the DML  (Deep Mutual Learning) distillation strategy to optimizing the Teacher model. For the student model, 
we propose a lightweight FPN structure RSEFPN to improve the accuracy of the student model.

<div align="center">
    <img src=".././ppocr_v3/ppocrv3_det_cml.png" width="800">
</div>


The ablation experiment is as follows:

|ID|Strategy|Model Size|Hmean|The Inference Time（cpu + mkldnn)|
|-|-|-|-|-|
|baseline teacher|PP-OCR server|49M|83.2%|171ms|
|teacher1|DB-R50-LK-PAN|124M|85.0%|396ms|
|teacher2|DB-R50-LK-PAN-DML|124M|86.0%|396ms|
|baseline student|PP-OCRv2|3M|83.2%|117ms|
|student0|DB-MV3-RSE-FPN|3.6M|84.5%|124ms|
|student1|DB-MV3-CML（teacher2）|3M|84.3%|117ms|
|student2|DB-MV3-RSE-FPN-CML（teacher2）|3.6M|85.4%|124ms|

The environment: Intel Gold 6148 CPU, with MKLDNN acceleration enabled during inference.



**(1) LK-PAN: PAN structure with large receptive field**

LK-PAN (Large Kernel PAN) is a lightweight [PAN](https://arxiv.org/pdf/1803.01534.pdf) structure with a larger receptive field. The core is to change the convolution kernel in the path augmentation of the PAN structure from `3*3` to `9*9`. By increasing the convolution kernel, the receptive field covered by each position of the feature map is improved, and it is easier to detect text in large fonts and text with extreme aspect ratios. Using the LK-PAN, the hmean of the teacher model can be improved from 83.2% to 85.0%.

<div align="center">
    <img src="../ppocr_v3/LKPAN.png" width="1000">
</div>



**(2) DML: The Mutual Learning Strategy for Teacher Model**

The [DML](https://arxiv.org/abs/1706.00384) method, as shown in the figure below, can effectively improve the accuracy of the text detection model by learning from each other with two models with the same structure. The teacher model adopts the DML strategy, and the hmean is increased from 85% to 86%. By updating the teacher model of CML in PP-OCRv2 to the above-mentioned higher-precision teacher model, the hmean of the student model can be further improved from 83.2% to 84.3%.


<div align="center">
    <img src="../ppocr_v3/teacher_dml.png" width="800">
</div>


**(3) RSE-FPN: FPN structure of residual attention mechanism**

RSE-FPN (Residual Squeeze-and-Excitation FPN) is shown in the figure below. RSEFPN introduces the residual structure and the channel attention structure, and replaces the convolutional layer in the FPN with the RSEConv layer of the channel attention structure to improve the representation ability of the feature map.
Considering that the number of FPN channels in the detection model of PP-OCRv2 is very small, only 96, if SEblock is directly used to replace the convolution in FPN, the features of some channels will be suppressed, and the accuracy will be reduced. The introduction of residual structure in RSEConv will alleviate the above problems and improve the text detection effect. By further updating the FPN structure of the student model of CML in PP-OCRv2 to RSE-FPN, the hmean of the student model can be further improved from 84.3% to 85.4%.


<div align="center">
    <img src=".././ppocr_v3/RSEFPN.png" width="1000">
</div>
