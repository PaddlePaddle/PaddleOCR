## Tricks
Here we have sorted out some Chinese OCR training and prediction tricks, which are being updated continuously. You are welcome to contribute more OCR tricks ~

- [Replace Backbone Network](#ReplaceBackboneNetwork)
- [Long Chinese Text Recognition](#LongChineseTextRecognition)
- [Space Recognition](#SpaceRecognition)

<a name="ReplaceBackboneNetwork"></a>
#### 1、Replace Backbone Network
- **Problem Description**

  At present, ResNet_vd series and MobileNetV3 series are the backbone networks used in PaddleOCR, whether replacing the other backbone networks will help to improve the accuracy? What should be paid attention to when replacing?

- **Tips**
  - Whether text detection or text recognition, the choice of backbone network is a trade-off between prediction effect and prediction efficiency. Generally, a larger backbone network is selected, e.g. ResNet101_vd, then the performance of the detection or recognition is more accurate, but the time cost will increase accordingly. And a smaller backbone network is selected, e.g. MobileNetV3_small_x0_35, the prediction speed is faster, but the accuracy of detection or recognition will be reduced. Fortunately, the detection or recognition effect of different backbone networks is positively correlated with the performance of ImageNet 1000 classification task. [**PaddleClas**](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/en/models/models_intro_en.md) have sorted out the 23 series of classification network structures, such as ResNet_vd、Res2Net、HRNet、MobileNetV3、GhostNet. It provides the top1 accuracy of classification, the time cost of GPU(V100 and T4) and CPU(SD 855), and the 117 pretrained models [**download addresses**](https://paddleclas-en.readthedocs.io/en/latest/models/models_intro_en.html).

  - Similar as the 4 stages of ResNet, the replacement of text detection backbone network is to determine those four stages to facilitate the integration of FPN like the object detection heads. In addition, for the text detection problem, the pre trained model in ImageNet1000 can accelerate the convergence and improve the accuracy.

  - In order to replace the backbone network of text recognition, we need to pay attention to the descending position of network width and height stride. Since the ratio between width and height is large in chinese text recognition, the frequency of height decrease is less and the frequency of width decrease is more. You can refer the [modifies of MobileNetV3](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/modeling/backbones/rec_mobilenet_v3.py) in PaddleOCR.

<a name="LongChineseTextRecognition"></a>
#### 2、Long Chinese Text Recognition
- **Problem Description**
  The maximum resolution of Chinese recognition model during training is [3,32,320], if the text image to be recognized is too long, as shown in the figure below, how to adapt?

  <div align="center">
    <img src="../tricks/long_text_examples.jpg" width="600">
  </div>

- **Tips**

  During the training, the training samples are not directly resized to [3,32,320]. At first, the height of samples are resized to 32 and keep the ratio between the width and the height. When the width is less than 320, the excess parts are padding 0. Besides, when the ratio between the width and the height of the samples is larger than 10, these samples will be ignored. When the prediction for one image, do as above, but do not limit the max ratio between the width and the height. When the prediction for an images batch, do as training, but the resized target width is the longest width of the images in the batch. [Code as following](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/tools/infer/predict_rec.py)：

  ```
    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        assert imgC == img.shape[2]
        if self.character_type == "ch":
            imgW = int((32 * max_wh_ratio))
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im  
  ```

<a name="SpaceRecognition"></a>
#### 3、Space Recognition
- **Problem Description**

  As shown in the figure below, for Chinese and English mixed scenes, in order to facilitate reading and using the recognition results, it is often necessary to recognize the spaces between words. How can this situation be adapted?

  <div align="center">
    <img src="../imgs_results/chinese_db_crnn_server/en_paper.jpg" width="600">
  </div>

- **Tips**

  There are two possible methods for space recognition. (1) Optimize the text detection. For spliting the text at the space in detection results, it needs to divide the text line with space into many segments when label the data for detection. (2) Optimize the text recognition. The space character is introduced into the recognition dictionary. Label the blank line in the training data for text recognition. In addition, we can also concat multiple word lines to synthesize the training data with spaces. PaddleOCR currently uses the second method.
