## FAQ

1. **预测报错：got an unexpected keyword argument 'gradient_clip'**  
安装的paddle版本不对，目前本项目仅支持paddle1.7，近期会适配到1.8。

2. **转换attention识别模型时报错：KeyError: 'predict'**  
问题已解，请更新到最新代码。

3. **关于推理速度**  
图片中的文字较多时，预测时间会增，可以使用--rec_batch_num设置更小预测batch num，默认值为30，可以改为10或其他数值。

4. **服务部署与移动端部署**  
预计6月中下旬会先后发布基于Serving的服务部署方案和基于Paddle Lite的移动端部署方案，欢迎持续关注。

5. **自研算法发布时间**  
自研算法SAST、SRN、End2End-PSL都将在6-7月陆续发布，敬请期待。

6. **如何在Windows或Mac系统上运行**  
PaddleOCR已完成Windows和Mac系统适配，运行时注意两点：1、在[快速安装](./installation.md)时，如果不想安装docker，可跳过第一步，直接从第二步安装paddle开始。2、inference模型下载时，如果没有安装wget，可直接点击模型链接或将链接地址复制到浏览器进行下载，并解压放置到相应目录。

7. **超轻量模型和通用OCR模型的区别**  
目前PaddleOCR开源了2个中文模型，分别是8.6M超轻量中文模型和通用中文OCR模型。两者对比信息如下：
    - 相同点：两者使用相同的**算法**和**训练数据**；  
    - 不同点：不同之处在于**骨干网络**和**通道参数**，超轻量模型使用MobileNetV3作为骨干网络，通用模型使用Resnet50_vd作为检测模型backbone，Resnet34_vd作为识别模型backbone，具体参数差异可对比两种模型训练的配置文件.

|模型|骨干网络|检测训练配置|识别训练配置|
|-|-|-|-|
|8.6M超轻量中文OCR模型|MobileNetV3+MobileNetV3|det_mv3_db.yml|rec_chinese_lite_train.yml|
|通用中文OCR模型|Resnet50_vd+Resnet34_vd|det_r50_vd_db.yml|rec_chinese_common_train.yml|

8. **是否有计划开源仅识别数字或仅识别英文+数字的模型**  
暂不计划开源仅数字、仅数字+英文、或其他小垂类专用模型。PaddleOCR开源了多种检测、识别算法供用户自定义训练，两种中文模型也是基于开源的算法库训练产出，有小垂类需求的小伙伴，可以按照教程准备好数据，选择合适的配置文件，自行训练，相信能有不错的效果。训练有任何问题欢迎提issue或在交流群提问，我们会及时解答。

9. **开源模型使用的训练数据是什么，能否开源**  
目前开源的模型，数据集和量级如下：
    - 检测：  
    英文数据集，ICDAR2015  
    中文数据集，LSVT街景数据集训练数据3w张图片
    - 识别：  
    英文数据集，MJSynth和SynthText合成数据，数据量上千万。  
    中文数据集，LSVT街景数据集根据真值将图crop出来，并进行位置校准，总共30w张图像。此外基于LSVT的语料，合成数据500w。

    其中，公开数据集都是开源的，用户可自行搜索下载，也可参考[中文数据集](./datasets.md)，合成数据暂不开源，用户可使用开源合成工具自行合成，可参考的合成工具包括[text_renderer](https://github.com/Sanster/text_renderer)、[SynthText](https://github.com/ankush-me/SynthText)、[TextRecognitionDataGenerator](https://github.com/Belval/TextRecognitionDataGenerator)等。
