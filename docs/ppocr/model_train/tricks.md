---
comments: true
---

这里我们整理了一些中文OCR训练和预测技巧，持续更新中，欢迎大家贡献更多OCR技巧~

#### 1、更换骨干网络

- **问题描述**

    目前PaddleOCR使用的主干网络为ResNet_vd系列和MobileNetV3系列，更换其他主干网络是否有助于提高准确率？更换时需要注意什么？

- **技巧**
    - 无论是文本检测还是文本识别，主干网络的选择都是预测效果和预测效率的权衡。一般选择较大的主干网络，如ResNet101_vd，则检测或识别的性能更准确，但时间成本也会相应增加。而选择较小的主干网络，如MobileNetV3_small_x0_35，预测速度更快，但检测或识别的准确率会降低。幸运的是，不同骨干网络的检测或识别效果与ImageNet 1000分类任务的性能呈正相关。[**PaddleClas**](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/docs/en/models/models_intro_en.md)整理了ResNet_vd、Res2Net、HRNet、MobileNetV3、GhostNet等23个系列的分类网络结构，提供了分类top1准确率、GPU(V100和T4)和CPU(SD 855)的时间成本，以及117个预训练模型[**下载地址**](https://paddleclas-en.readthedocs.io/en/latest/models/models_intro_en.html)。

- 和ResNet的4个阶段类似，文本检测骨干网络的更换就是确定这4个阶段，以便于像物体检测heads一样集成FPN。另外，对于文本检测问题，ImageNet1000中的预训练模型可以加速收敛并提高准确率。

- 更换文本识别骨干网络时，需要注意网络宽度和高度步长的下降位置。由于中文文本识别中宽度和高度的比值较大，因此高度下降的频率较少，宽度下降的频率较多。可以参考PaddleOCR中[MobileNetV3的修改](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/modeling/backbones/rec_mobilenet_v3.py)。

#### 2、长中文文本识别

- **问题描述**
  中文识别模型在训练时的最大分辨率为[3,32,320]，如果待识别的文本图像过长，如下图所示，该如何适配？

  ![img](./images/long_text_examples.jpg)

- **小技巧**

在训练时，不要直接将训练样本resize到[3,32,320]，先将样本的高度resize为32，并保持宽高比，当宽度小于320时，超出部分用0填充。另外，当样本的宽高比大于10时，这些样本将被忽略。对一张图片进行预测时，同上，但不限制最大宽高比。对一批图像进行预测时，按照训练的方式进行，但调整后的目标宽度是该批图像的最长宽度。 [代码如下](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/tools/infer/predict_rec.py)：

  ```python linenums="1"
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

#### 3、空格识别

- **问题描述**

如下图所示，对于中英文混合场景，为了方便阅读和使用识别结果，经常需要识别单词之间的空格，这种情况该如何适配？

![img](./images/en_paper.jpg)

- **小技巧**

空格识别有两种可能的方法。（1）优化文本检测。为了将检测结果中的文本分割在空格处，在对数据进行标记时，需要将带有空格的文本行分成许多段
