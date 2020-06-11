## FAQ

1. **Prediction error: got an unexpected keyword argument 'gradient_clip'**  
The installed version of paddle is incorrect. Currently, this project only supports paddle1.7, which will be adapted to 1.8 in the near future.

2. **Error when converting attention recognition model: KeyError: 'predict'**  
Solved. Please update to the latest version of the code.

3. **About inference speed**  
When there are many words in the picture, the prediction time will increase. You can use `--rec_batch_num` to set a smaller prediction batch num. The default value is 30, which can be changed to 10 or other values.

4. **Service deployment and mobile deployment**  
It is expected that the service deployment based on Serving and the mobile deployment based on Paddle Lite will be released successively in mid-to-late June. Stay tuned for more updates.

5. **Release time of self-developed algorithm**  
Baidu Self-developed algorithms such as SAST, SRN and end2end PSL will be released in June or July. Please be patient.

6. **How to run on Windows or Mac?**  
PaddleOCR has completed the adaptation to Windows and MAC systems. Two points should be noted during operation:
    1. In [Quick installation](./installation_en.md), if you do not want to install docker, you can skip the first step and start with the second step.
    2. When downloading the inference model, if wget is not installed, you can directly click the model link or copy the link address to the browser to download, then extract and place it in the corresponding directory.

7. **The difference between ultra-lightweight model and General OCR model**  
At present, PaddleOCR has opensourced two Chinese models, namely 8.6M ultra-lightweight Chinese model and general Chinese OCR model. The comparison information between the two is as follows:
    - Similarities: Both use the same **algorithm** and **training data**；  
    - Differences: The difference lies in **backbone network** and **channel parameters**, the ultra-lightweight model uses MobileNetV3 as the backbone network, the general model uses Resnet50_vd as the detection model backbone, and Resnet34_vd as the recognition model backbone. You can compare the two model training configuration files to see the differences in parameters.

|Model|Backbone|Detection configuration file|Recognition configuration file|
|-|-|-|-|
|8.6M ultra-lightweight Chinese OCR model|MobileNetV3+MobileNetV3|det_mv3_db.yml|rec_chinese_lite_train.yml|
|General Chinese OCR model|Resnet50_vd+Resnet34_vd|det_r50_vd_db.yml|rec_chinese_common_train.yml|

8. **Is there a plan to opensource a model that only recognizes numbers or only English + numbers?**  
It is not planned to opensource numbers only, numbers + English only, or other vertical text models. Paddleocr has opensourced a variety of detection and recognition algorithms for customized training. The two Chinese models are also based on the training output of the open-source algorithm library. You can prepare the data according to the tutorial, choose the appropriate configuration file, train yourselves, and we believe that you can get good result. If you have any questions during the training, you are welcome to open issues or ask in the communication group. We will answer them in time.

9. **What is the training data used by the open-source model? Can it be opensourced?**  
At present, the open source model, dataset and magnitude are as follows:
    - Detection:  
    English dataset: ICDAR2015  
    Chinese dataset: LSVT street view dataset with 3w pictures
    - Recognition:  
    English dataset: MJSynth and SynthText synthetic dataset, the amount of data is tens of millions.  
    Chinese dataset: LSVT street view dataset with cropped text area, a total of 30w images. In addition, the synthesized data based on LSVT corpus is 500w.

    Among them, the public datasets are opensourced, users can search and download by themselves, or refer to [Chinese data set](./datasets_en.md), synthetic data is not opensourced, users can use open-source synthesis tools to synthesize data themselves. Current available synthesis tools include [text_renderer](https://github.com/Sanster/text_renderer), [SynthText](https://github.com/ankush-me/SynthText), [TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator), etc.

10. **Error in using the model with TPS module for prediction**

Error message: Input(X) dims[3] and Input(Grid) dims[2] should be equal, but received X dimension[3](108) != Grid dimension[2](100)

Solution：TPS does not support variable shape. Please set --rec_image_shape='3,32,100' and --rec_char_type='en'
