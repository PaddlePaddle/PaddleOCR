English | [简体中文](../doc_ch/PP-OCRv3_introduction.md)



<a name="3"></a>

## 3. Optimization for Text Recognition Model

The recognition module of PP-OCRv3 is optimized based on the text recognition algorithm [SVTR] (https://arxiv.org/abs/2205.00159). RNN is abandoned in SVTR, and the context information of the text line image is more effectively mined by introducing the Transformers structure, thereby improving the text recognition ability. The recognition model of PP-OCRv2 was directly replaced with SVTR_Tiny, and the recognition accuracy increased from 74.8% to 80.1% (+5.3%), but the prediction speed was nearly 11 times slower, and it took nearly 100ms to predict a text line on the CPU. Therefore, as shown in the figure below, PP-OCRv3 adopts the following six optimization strategies to accelerate the recognition model.

<div align="center">
    <img src="../ppocr_v3/v3_rec_pipeline.png" width=800>
</div>

Based on the above strategy, compared with PP-OCRv2, the PP-OCRv3 recognition model further improves the accuracy by 4.6% with comparable speed. The specific ablation experiments are as follows:

| ID | strategy |  Model size | accuracy | prediction speed（CPU + MKLDNN)|
|-----|-----|--------|----| --- |
| 01 | PP-OCRv2 | 8M | 74.8% | 8.54ms |
| 02 | SVTR_Tiny | 21M | 80.1% | 97ms |
| 03 | SVTR_LCNet(h32) | 12M | 71.9% | 6.6ms |
| 04 | SVTR_LCNet(h48) | 12M | 73.98% | 7.6ms |
| 05 | + GTC | 12M | 75.8% | 7.6ms |
| 06 | + TextConAug | 12M | 76.3% | 7.6ms |
| 07 | + TextRotNet | 12M | 76.9% | 7.6ms |
| 08 | + UDML | 12M | 78.4% | 7.6ms |
| 09 | + UIM | 12M | 79.4% | 7.6ms |

Note: When testing the speed, the input image shape of Experiment 01-03 is (3, 32, 320), and the input image shape of 04-08 is (3, 48, 320). In the actual prediction, the image is a variable-length input, and the speed will vary. Test environment: Intel Gold 6148 CPU, with MKLDNN acceleration enabled during prediction.

**（1）SVTR_LCNet：Lightweight Text Recognition Network**

SVTR_LCNet is for text recognition tasks, which is a lightweight text recognition network fused by Transformer-based [SVTR](https://arxiv.org/abs/2205.00159) network and lightweight CNN network [PP-LCNet](https://arxiv.org/abs/ 2109.15099). Using this network, the prediction speed is 20% better than the recognition model of PP-OCRv2, but because the distillation strategy is not adopted, the effect of the recognition model is slightly worse. In addition, the height of the input image is further increased from 32 to 48, and the prediction speed is slightly slower, but the model effect is greatly improved, and the recognition accuracy reaches 73.98% (+2.08%), which is close to the recognition model effect of PP-OCRv2 using the distillation strategy.

SVTR_Tiny network structure is as follows：

<div align="center">
    <img src="../ppocr_v3/svtr_tiny.png" width=800>
</div>

Due to the limited model structure supported by the MKLDNN acceleration library, SVTR is 10 times slower than PP-OCRv2 on CPU+MKLDNN. PP-OCRv3 expects to improve the accuracy of the model without bringing additional inference time. Through analysis, it is found that the main time-consuming module of SVTR_Tiny structure is Mixing Block, so we have carried out a series of optimizations to the structure of SVTR_Tiny (for detailed speed data, please refer to the ablation experiment table below):


1. Replace the first half of the SVTR network with the first three stages of PP-LCNet, retain 4 Global Mixing Blocks, the accuracy is 76%, and the speedup is 69%. The network structure is as follows:

<div align="center">
    <img src="../ppocr_v3/svtr_g4.png" width=800>
</div>

2. Reduce the number of Global Mixing Blocks from 4 to 2, the accuracy is 72.9%, and the speedup is 69%. The network structure is as follows:

<div align="center">
    <img src="../ppocr_v3/svtr_g2.png" width=800>
</div>

3. The experiment found that the prediction speed of the Global Mixing Block is related to the shape of the input features. Therefore, after moving the position of the Global Mixing Block to the back of pooling layer, the accuracy dropped to 71.9%, and the speed surpassed the PP-OCRv2-baseline based on the CNN structure by 22%. The network structure is as follows:

<div align="center">
    <img src="../ppocr_v3/LCNet_SVTR.png" width=800>
</div>

The specific ablation experiments are as follows:

| ID | strategy |  Model size | accuracy | prediction speed（CPU + MKLDNN)|
|-----|-----|--------|----| --- |
| 01 | PP-OCRv2-baseline | 8M | 69.3%  | 8.54ms |
| 02 | SVTR_Tiny | 21M | 80.1% | 97ms |
| 03 | SVTR_LCNet(G4) | 9.2M | 76% | 30ms |
| 04 | SVTR_LCNet(G2) | 13M | 72.98% | 9.37ms |
| 05 | SVTR_LCNet(h32) | 12M | 71.9% | 6.6ms |
| 06 | SVTR_LCNet(h48)  | 12M | 73.98% | 7.6ms |

Note: When testing the speed, the input image shape of 01-05 are all (3, 32, 320); PP-OCRv2-baseline represents the model trained without distillation method

**（2）GTC：Attention guides CTC training strategy**

[GTC](https://arxiv.org/pdf/2002.01276.pdf)（Guided Training of CTC），using the Attention module and loss to guide the CTC loss training and fuse the expression of multiple text features is an effective strategy to improve text recognition. Using this strategy, the Attention module is completely removed during prediction, and no time-consuming is added in the inference stage, and the accuracy of the recognition model is further improved to 75.8% (+1.82%). The training process is as follows:

<div align="center">
    <img src="../ppocr_v3/GTC.png" width=800>
</div>

**（3）TextConAug：Data Augmentation Strategy for Mining Text Context Information**

TextConAug is a data augmentation strategy for mining textual context information. The main idea comes from the paper [ConCLR](https://www.cse.cuhk.edu.hk/~byu/papers/C139-AAAI2022-ConCLR.pdf) , the author proposes ConAug data augmentation to connect 2 different images in a batch to form new images and perform self-supervised comparative learning. PP-OCRv3 applies this method to supervised learning tasks, and designs the TextConAug data augmentation method, which can enrich the context information of training data and improve the diversity of training data. Using this strategy, the accuracy of the recognition model is further improved to 76.3% (+0.5%). The schematic diagram of TextConAug is as follows:

<div align="center">
    <img src="../ppocr_v3/recconaug.png" width=800>
</div>


**（4）TextRotNet：Self-Supervised Pretrained Models**

TextRotNet is a pre-training model, which is trained by using a large amount of unlabeled text line data in a self-supervised manner. Refer to the paper [STR-Fewer-Labels](https://github.com/ku21fan/STR-Fewer-Labels). This model can initialize the initial weights of SVTR_LCNet, which helps the text recognition model to converge to a better position. Using this strategy, the accuracy of the recognition model is further improved to 76.9% (+0.6%). The TextRotNet training process is shown in the following figure:

<div align="center">
    <img src="../ppocr_v3/SSL.png" width="500">
</div>


**（5）UDML：Unified-Deep Mutual Learning**

UDML (Unified-Deep Mutual Learning) is a strategy adopted in PP-OCRv2 that is very effective for text recognition to improve the model effect. In PP-OCRv3, for two different SVTR_LCNet and Attention structures, the feature map of PP-LCNet, the output of the SVTR module and the output of the Attention module between them are simultaneously supervised and trained. Using this strategy, the accuracy of the recognition model is further improved to 78.4% (+1.5%).


**（6）UIM：Unlabeled Images Mining**

UIM (Unlabeled Images Mining) is a very simple unlabeled data mining scheme. The core idea is to use a high-precision text recognition model to predict unlabeled data, obtain pseudo-labels, and select samples with high prediction confidence as training data for training small models. Using this strategy, the accuracy of the recognition model is further improved to 79.4% (+1%).

<div align="center">
    <img src="../ppocr_v3/UIM.png" width="500">
</div>
