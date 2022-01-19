# Model Training

- [1.Yml Configuration ](#1-Yml-Configuration)
- [2. Basic Concepts](#1-basic-concepts)
  * [2.1 Learning Rate](#11-learning-rate)
  * [2.2 Regularization](#12-regularization)
  * [2.3 Evaluation Indicators](#13-evaluation-indicators-)
- [3. Data and Vertical Scenes](#2-data-and-vertical-scenes)
  * [3.1 Training Data](#21-training-data)
  * [3.2 Vertical Scene](#22-vertical-scene)
  * [3.3 Build Your Own Dataset](#23-build-your-own-data-set)
* [4. FAQ](#3-faq)


This article will introduce the basic concepts that is necessary for model training and tuning.

At the same time, it will briefly introduce the structure of the training data and how to prepare the data to fine-tune model in vertical scenes.

<a name="1-Yml-Configuration"></a>

## 1. Yml Configuration

The PaddleOCR uses configuration files to control network training and evaluation parameters. In the configuration file, you can set the model, optimizer, loss function, and pre- and post-processing parameters of the model. PaddleOCR reads these parameters from the configuration file, and then builds a complete training process to train the model. Fine-tuning can also be completed by modifying the parameters in the configuration file, which is simple and convenient.

For the complete configuration file description, please refer to [Configuration File](./config_en.md)

<a name="1-basic-concepts"></a>

## 2. Basic Concepts

During the model training process, some hyper-parameters can be manually specified to obtain the optimal result at the least cost. Different data volumes may require different hyper-parameters. When you want to fine-tune the model based on your own data, there are several parameter adjustment strategies for reference:

<a name="11-learning-rate"></a>
### 2.1 Learning Rate

The learning rate is one of the most important hyper-parameters for training neural networks. It represents the step length of the gradient moving towards the optimal solution of the loss function in each iteration.
A variety of learning rate update strategies are provided by PaddleOCR, which can be specified in configuration files. For example,

```
Optimizer:
  ...
  lr:
    name: Piecewise
    decay_epochs : [700, 800]
    values : [0.001, 0.0001]
    warmup_epoch: 5
```

`Piecewise` stands for piece-wise constant attenuation. Different learning rates are specified in different learning stages, and the learning rate stay the same in each stage.

`warmup_epoch` means that in the first 5 epochs, the learning rate will be increased gradually from 0 to base_lr. For all strategies, please refer to the code [learning_rate.py](../../ppocr/optimizer/learning_rate.py).

<a name="12-regularization"></a>
### 2.2 Regularization

Regularization can effectively avoid algorithm over-fitting. PaddleOCR provides L1 and L2 regularization methods.
L1 and L2 regularization are the most widely used regularization methods.
L1 regularization adds a regularization term to the objective function to reduce the sum of absolute values of the parameters;
while in L2 regularization, the purpose of adding a regularization term is to reduce the sum of squared parameters.
The configuration method is as follows:

```
Optimizer:
  ...
  regularizer:
    name: L2
    factor: 2.0e-05
```
<a name="13-evaluation-indicators-"></a>
### 2.3 Evaluation Indicators

(1) Detection stage: First, evaluate according to the IOU of the detection frame and the labeled frame. If the IOU is greater than a certain threshold, it is judged that the detection is accurate. Here, the detection frame and the label frame are different from the general general target detection frame, and they are represented by polygons. Detection accuracy: the percentage of the correct detection frame number in all detection frames is mainly used to judge the detection index. Detection recall rate: the percentage of correct detection frames in all marked frames, which is mainly an indicator of missed detection.

(2) Recognition stage: Character recognition accuracy, that is, the ratio of correctly recognized text lines to the number of marked text lines. Only the entire line of text recognition pairs can be regarded as correct recognition.

(3) End-to-end statistics: End-to-end recall rate: accurately detect and correctly identify the proportion of text lines in all labeled text lines; End-to-end accuracy rate: accurately detect and correctly identify the number of text lines in the detected text lines The standard for accurate detection is that the IOU of the detection box and the labeled box is greater than a certain threshold, and the text in the correctly identified detection box is the same as the labeled text.

<a name="2-data-and-vertical-scenes"></a>

## 3. Data and Vertical Scenes

<a name="21-training-data"></a>

### 3.1 Training Data

The current open source models, data sets and magnitudes are as follows:

- Detection:
    - English data set, ICDAR2015
    - Chinese data set, LSVT street view data set training data 3w pictures

- Identification:
    - English data set, MJSynth and SynthText synthetic data, the data volume is tens of millions.
    - Chinese data set, LSVT street view data set crops the image according to the truth value, and performs position calibration, a total of 30w images. In addition, based on the LSVT corpus, 500w of synthesized data.
    - Small language data set, using different corpora and fonts, respectively generated 100w synthetic data set, and using ICDAR-MLT as the verification set.

Among them, the public data sets are all open source, users can search and download by themselves, or refer to [Chinese data set](../doc_ch/datasets.md), synthetic data is not open source, users can use open source synthesis tools to synthesize by themselves. Synthesis tools include [text_renderer](https://github.com/Sanster/text_renderer), [SynthText](https://github.com/ankush-me/SynthText), [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) etc.

<a name="22-vertical-scene"></a>

### 3.2 Vertical Scene

PaddleOCR mainly focuses on general OCR. If you have vertical requirements, you can use PaddleOCR + vertical data to train yourself;
If there is a lack of labeled data, or if you do not want to invest in research and development costs, it is recommended to directly call the open API, which covers some of the more common vertical categories.  

<a name="23-build-your-own-data-set"></a>

### 3.3 Build Your Own Dataset

There are several experiences for reference when constructing the data set:

(1) The amount of data in the training set:

    a. The data required for detection is relatively small. For Fine-tune based on the PaddleOCR model, 500 sheets are generally required to achieve good results.
    b. Recognition is divided into English and Chinese. Generally, English scenarios require hundreds of thousands of data to achieve good results, while Chinese requires several million or more.


(2) When the amount of training data is small, you can try the following three ways to get more data:

    a. Manually collect more training data, the most direct and effective way.
    b. Basic image processing or transformation based on PIL and opencv. For example, the three modules of ImageFont, Image, ImageDraw in PIL write text into the background, opencv's rotating affine transformation, Gaussian filtering and so on.
    c. Use data generation algorithms to synthesize data, such as algorithms such as pix2pix.

<a name="3-faq"></a>

## 4. FAQ

**Q**: How to choose a suitable network input shape when training CRNN recognition?

    A: The general height is 32, the longest width is selected, there are two methods:

    (1) Calculate the aspect ratio distribution of training sample images. The selection of the maximum aspect ratio considers 80% of the training samples.

    (2) Count the number of texts in training samples. The selection of the longest number of characters considers the training sample that satisfies 80%. Then the aspect ratio of Chinese characters is approximately considered to be 1, and that of English is 3:1, and the longest width is estimated.

**Q**: During the recognition training, the accuracy of the training set has reached 90, but the accuracy of the verification set has been kept at 70, what should I do?

    A: If the accuracy of the training set is 90 and the test set is more than 70, it should be over-fitting. There are two methods to try:

    (1) Add more augmentation methods or increase the [probability] of augmented prob (https://github.com/PaddlePaddle/PaddleOCR/blob/dygraph/ppocr/data/imaug/rec_img_aug.py#L341), The default is 0.4.

    (2) Increase the [l2 dcay value] of the system (https://github.com/PaddlePaddle/PaddleOCR/blob/a501603d54ff5513fc4fc760319472e59da25424/configs/rec/ch_ppocr_v1.1/rec_chinese_lite_train_v1.1.yml#L47)

**Q**: When the recognition model is trained, loss can drop normally, but acc is always 0

    A: It is normal for the acc to be 0 at the beginning of the recognition model training, and the indicator will come up after a longer training period.

***

Click the following links for detailed training tutorial:  

- [text detection model training](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/detection.md)  
- [text recognition model training](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/recognition.md)  
- [text direction classification model training](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.3/doc/doc_ch/angle_class.md)  
