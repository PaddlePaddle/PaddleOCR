## MODEL TRAINING

This article will introduce the basic concepts that need to be mastered during model training and the tuning methods during training.

At the same time, it will briefly introduce the components of the PaddleOCR model training data and how to prepare the data finetune model in the vertical scene.

### 1. Basic concepts

OCR (Optical Character Recognition) refers to the process of analyzing and recognizing images to obtain text and layout information. It is a typical computer vision task.
It usually consists of two subtasks: text detection and text recognition.

The following parameters need to be paid attention to when tuning the model:

#### 1.1 Learning rate

The learning rate is one of the important hyperparameters for training neural networks. It represents the step length of the gradient moving to the optimal solution of the loss function in each iteration.
A variety of learning rate update strategies are provided in PaddleOCR, which can be modified through configuration files, for example:

```
Optimizer:
  ...
  lr:
    name: Piecewise
    decay_epochs : [700, 800]
    values : [0.001, 0.0001]
    warmup_epoch: 5
```

Piecewise stands for piecewise constant attenuation. Different learning rates are specified in different learning stages,
and the learning rate is the same in each stage.

warmup_epoch means that in the first 5 epochs, the learning rate will gradually increase from 0 to base_lr. For all strategies, please refer to the code [learning_rate.py](../../ppocr/optimizer/learning_rate.py).

#### 1.2 Regularization

Regularization can effectively avoid algorithm overfitting. PaddleOCR provides L1 and L2 regularization methods.
L1 and L2 regularization are the most commonly used regularization methods.
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

#### 1.3 Evaluation indicators:

(1) Detection stage: First, evaluate according to the IOU of the detection frame and the labeled frame. If the IOU is greater than a certain threshold, it is judged that the detection is accurate. Here, the detection frame and the label frame are different from the general general target detection frame, and they are represented by polygons. Detection accuracy: the percentage of the correct detection frame number in all detection frames is mainly used to judge the detection index. Detection recall rate: the percentage of correct detection frames in all marked frames, which is mainly an indicator of missed detection.

(2) Recognition stage: Character recognition accuracy, that is, the ratio of correctly recognized text lines to the number of marked text lines. Only the entire line of text recognition pairs can be regarded as correct recognition.

(3) End-to-end statistics: End-to-end recall rate: accurately detect and correctly identify the proportion of text lines in all labeled text lines; End-to-end accuracy rate: accurately detect and correctly identify the number of text lines in the detected text lines The standard for accurate detection is that the IOU of the detection box and the labeled box is greater than a certain threshold, and the text in the correctly identified detection box is the same as the labeled text.


### 2. FAQ

**Q**: What are the text detection methods based on deep learning? What are the advantages and disadvantages of each?

    A: Commonly used deep learning-based text detection methods can generally be divided into two categories: regression-based and segmentation-based, and of course there are some methods that combine the two.

    (1) Methods based on regression are divided into box regression and pixel value regression. a. The methods that use box regression mainly include CTPN, Textbox series and EAST. This type of algorithm has a better effect on regular shape text detection, but it cannot accurately detect irregular shape text. b. The methods of pixel value regression mainly include CRAFT and SA-Text. This type of algorithm can detect curved text and has an excellent effect on small text, but the real-time performance is not enough.

    (2) Algorithms based on segmentation, such as PSENet, are not limited by the shape of the text, and can achieve better results for texts of various shapes, but the post-processing is often more complicated, leading to serious time-consuming. At present, there are also some algorithms that are specifically improved for this problem, such as DB, which approximates the binarization, makes it guideable, and integrates it into training, so as to obtain a more accurate boundary, which greatly reduces the time-consuming post-processing.



**Q**: For Chinese line text recognition, which is better, CTC or Attention?

    A:
    (1) From the point of view of effect, the recognition effect of CTC in general OCR scene is better than Attention, because there are more characters in the dictionary with recognition, and the commonly used Chinese characters are more than 3,000 characters. If the training samples are insufficient, for these characters Sequence relationship mining is more difficult. The advantages of the Attention model in the Chinese scene cannot be reflected. Moreover, Attention is suitable for short sentence recognition, and it is relatively poor in recognition of long sentences.

    (2) In terms of training and prediction speed, Attention's serial decoding structure limits the prediction speed, while the CTC network structure is more efficient and has an advantage in prediction speed.

**Q**: How to choose a suitable network input shape when training CRNN recognition?

    A: The general height is 32, the longest width is selected, there are two methods:

    (1) Calculate the aspect ratio distribution of training sample images. The selection of the maximum aspect ratio considers 80% of the training samples.

    (2) Count the number of texts in training samples. The selection of the longest number of characters considers the training sample that satisfies 80%. Then the aspect ratio of Chinese characters is approximately considered to be 1, and that of English is 3:1, and the longest width is estimated.



### 3. Data and vertical scenes

#### 3.1 Training data

The current open source models, data sets and magnitudes are as follows:

- Detection:
    - English data set, ICDAR2015
    - Chinese data set, LSVT street view data set training data 3w pictures

- Identification:
    - English data set, MJSynth and SynthText synthetic data, the data volume is tens of millions.
    - Chinese data set, LSVT street view data set crops the image according to the truth value, and performs position calibration, a total of 30w images. In addition, based on the LSVT corpus, 500w of synthesized data.
    - Small language data set, using different corpora and fonts, respectively generated 100w synthetic data set, and using ICDAR-MLT as the verification set.

Among them, the public data sets are all open source, users can search and download by themselves, or refer to [Chinese data set](./datasets.md), synthetic data is not open source, users can use open source synthesis tools to synthesize by themselves. Synthesis tools include [text_renderer](https://github.com/Sanster/text_renderer), [SynthText](https://github.com/ankush-me/SynthText), [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator) etc.


#### 3.2 Vertical scene

PaddleOCR mainly focuses on general OCR. If you have vertical requirements, you can use PaddleOCR + vertical data to train yourself;
If there is a lack of labeled data, or if you do not want to invest in research and development costs, it is recommended to directly call the open API, which covers some of the more common vertical categories.  


#### 3.3 Build your own data set

There are several experiences for reference when constructing the data set:

(1) The amount of data in the training set:

    a. The data required for detection is relatively small. For Fine-tune based on the PaddleOCR model, 500 sheets are generally required to achieve good results.
    b. Recognition is divided into English and Chinese. Generally, English scenarios require hundreds of thousands of data to achieve good results, while Chinese requires several million or more.


(2) When the amount of training data is small, you can try the following three ways to get more data:

    a. Manually collect more training data, the most direct and effective way.
    b. Basic image processing or transformation based on PIL and opencv. For example, the three modules of ImageFont, Image, ImageDraw in PIL write text into the background, opencv's rotating affine transformation, Gaussian filtering and so on.
    c. Use data generation algorithms to synthesize data, such as algorithms such as pix2pix.
