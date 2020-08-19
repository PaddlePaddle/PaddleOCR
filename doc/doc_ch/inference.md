
# 基于Python预测引擎推理

inference 模型（fluid.io.save_inference_model保存的模型）
一般是模型训练完成后保存的固化模型，多用于预测部署。训练过程中保存的模型是checkpoints模型，保存的是模型的参数，多用于恢复训练等。
与checkpoints模型相比，inference 模型会额外保存模型的结构信息，在预测部署、加速推理上性能优越，灵活方便，适合与实际系统集成。更详细的介绍请参考文档[分类预测框架](https://paddleclas.readthedocs.io/zh_CN/latest/extension/paddle_inference.html).

接下来首先介绍如何将训练的模型转换成inference模型，然后将依次介绍文本检测、文本识别以及两者串联基于预测引擎推理。


- [一、训练模型转inference模型](#训练模型转inference模型)
    - [检测模型转inference模型](#检测模型转inference模型)
    - [识别模型转inference模型](#识别模型转inference模型)
    
    
- [二、文本检测模型推理](#文本检测模型推理)
    - [1. 超轻量中文检测模型推理](#超轻量中文检测模型推理)
    - [2. DB文本检测模型推理](#DB文本检测模型推理)
    - [3. EAST文本检测模型推理](#EAST文本检测模型推理)
    - [4. SAST文本检测模型推理](#SAST文本检测模型推理)
    
    
- [三、文本识别模型推理](#文本识别模型推理)
    - [1. 超轻量中文识别模型推理](#超轻量中文识别模型推理)
    - [2. 基于CTC损失的识别模型推理](#基于CTC损失的识别模型推理)
    - [3. 基于Attention损失的识别模型推理](#基于Attention损失的识别模型推理)
    - [4. 自定义文本识别字典的推理](#自定义文本识别字典的推理)
    
    
- [四、文本检测、识别串联推理](#文本检测、识别串联推理)
    - [1. 超轻量中文OCR模型推理](#超轻量中文OCR模型推理)
    - [2. 其他模型推理](#其他模型推理)
    
    
<a name="训练模型转inference模型"></a>
## 一、训练模型转inference模型
<a name="检测模型转inference模型"></a>
### 检测模型转inference模型

下载超轻量级中文检测模型：
```
wget -P ./ch_lite/ https://paddleocr.bj.bcebos.com/ch_models/ch_det_mv3_db.tar && tar xf ./ch_lite/ch_det_mv3_db.tar -C ./ch_lite/
```
上述模型是以MobileNetV3为backbone训练的DB算法，将训练好的模型转换成inference模型只需要运行如下命令：
```
# -c后面设置训练算法的yml配置文件
# -o配置可选参数
# Global.checkpoints参数设置待转换的训练模型地址，不用添加文件后缀.pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3 tools/export_model.py -c configs/det/det_mv3_db.yml -o Global.checkpoints=./ch_lite/det_mv3_db/best_accuracy Global.save_inference_dir=./inference/det_db/
```
转inference模型时，使用的配置文件和训练时使用的配置文件相同。另外，还需要设置配置文件中的`Global.checkpoints`、`Global.save_inference_dir`参数。
其中`Global.checkpoints`指向训练中保存的模型参数文件，`Global.save_inference_dir`是生成的inference模型要保存的目录。
转换成功后，在`save_inference_dir`目录下有两个文件：
```
inference/det_db/
  └─  model     检测inference模型的program文件
  └─  params    检测inference模型的参数文件
```

<a name="识别模型转inference模型"></a>
### 识别模型转inference模型

下载超轻量中文识别模型：
```
wget -P ./ch_lite/ https://paddleocr.bj.bcebos.com/ch_models/ch_rec_mv3_crnn.tar && tar xf ./ch_lite/ch_rec_mv3_crnn.tar -C ./ch_lite/
```

识别模型转inference模型与检测的方式相同，如下：
```
# -c后面设置训练算法的yml配置文件
# -o配置可选参数
# Global.checkpoints参数设置待转换的训练模型地址，不用添加文件后缀.pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3 tools/export_model.py -c configs/rec/rec_chinese_lite_train.yml -o Global.checkpoints=./ch_lite/rec_mv3_crnn/best_accuracy \
        Global.save_inference_dir=./inference/rec_crnn/
```

**注意：**如果您是在自己的数据集上训练的模型，并且调整了中文字符的字典文件，请注意修改配置文件中的`character_dict_path`是否是所需要的字典文件。

转换成功后，在目录下有两个文件：
```
/inference/rec_crnn/
  └─  model     识别inference模型的program文件
  └─  params    识别inference模型的参数文件
```

<a name="文本检测模型推理"></a>
## 二、文本检测模型推理

文本检测模型推理，默认使用DB模型的配置参数。当不使用DB模型时，在推理时，需要通过传入相应的参数进行算法适配，细节参考下文。

<a name="超轻量中文检测模型推理"></a>
### 1. 超轻量中文检测模型推理

超轻量中文检测模型推理，可以执行如下命令：

```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/"
```

可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![](../imgs_results/det_res_2.jpg)

通过设置参数`det_max_side_len`的大小，改变检测算法中图片规范化的最大值。当图片的长宽都小于`det_max_side_len`，则使用原图预测，否则将图片等比例缩放到最大值，进行预测。该参数默认设置为`det_max_side_len=960`。 如果输入图片的分辨率比较大，而且想使用更大的分辨率预测，可以执行如下命令：

```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/" --det_max_side_len=1200
```

如果想使用CPU进行预测，执行命令如下
```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/" --use_gpu=False
```

<a name="DB文本检测模型推理"></a>
### 2. DB文本检测模型推理

首先将DB文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在ICDAR2015英文数据集训练的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/det_r50_vd_db.tar))，可以使用如下命令进行转换：

```
# -c后面设置训练算法的yml配置文件
# Global.checkpoints参数设置待转换的训练模型地址，不用添加文件后缀.pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3 tools/export_model.py -c configs/det/det_r50_vd_db.yml -o Global.checkpoints="./models/det_r50_vd_db/best_accuracy" Global.save_inference_dir="./inference/det_db"
```

DB文本检测模型推理，可以执行如下命令：

```
python3 tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_db/"
```

可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![](../imgs_results/det_res_img_10_db.jpg)

**注意**：由于ICDAR2015数据集只有1000张训练图像，且主要针对英文场景，所以上述模型对中文文本图像检测效果会比较差。

<a name="EAST文本检测模型推理"></a>
### 3. EAST文本检测模型推理

首先将EAST文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在ICDAR2015英文数据集训练的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/det_r50_vd_east.tar))，可以使用如下命令进行转换：

```
# -c后面设置训练算法的yml配置文件
# Global.checkpoints参数设置待转换的训练模型地址，不用添加文件后缀.pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3 tools/export_model.py -c configs/det/det_r50_vd_east.yml -o Global.checkpoints="./models/det_r50_vd_east/best_accuracy" Global.save_inference_dir="./inference/det_east"
```

**EAST文本检测模型推理，需要设置参数`det_algorithm`，指定检测算法类型为EAST**，可以执行如下命令：

```
python3 tools/infer/predict_det.py --det_algorithm="EAST" --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_east/"
```
可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![](../imgs_results/det_res_img_10_east.jpg)

**注意**：本代码库中，EAST后处理Locality-Aware NMS有python和c++两种版本，c++版速度明显快于python版。由于c++版本nms编译版本问题，只有python3.5环境下会调用c++版nms，其他情况将调用python版nms。


<a name="SAST文本检测模型推理"></a>
### 4. SAST文本检测模型推理
#### (1). 四边形文本检测模型（ICDAR2015）  
首先将SAST文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在ICDAR2015英文数据集训练的模型为例([模型下载地址](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_icdar2015.tar))，可以使用如下命令进行转换：
```
python3 tools/export_model.py -c configs/det/det_r50_vd_sast_icdar15.yml -o Global.checkpoints="./models/sast_r50_vd_icdar2015/best_accuracy" Global.save_inference_dir="./inference/det_sast_ic15"
```
**SAST文本检测模型推理，需要设置参数`det_algorithm`，指定检测算法类型为SAST**，可以执行如下命令：
```
python3 tools/infer/predict_det.py --det_algorithm="SAST" --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_sast_ic15/"
```
可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![](../imgs_results/det_res_img_10_sast.jpg)

#### (2). 弯曲文本检测模型（Total-Text）  
首先将SAST文本检测训练过程中保存的模型，转换成inference model。以基于Resnet50_vd骨干网络，在Total-Text英文数据集训练的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/SAST/sast_r50_vd_total_text.tar))，可以使用如下命令进行转换：

```
python3 tools/export_model.py -c configs/det/det_r50_vd_sast_totaltext.yml -o Global.checkpoints="./models/sast_r50_vd_total_text/best_accuracy" Global.save_inference_dir="./inference/det_sast_tt"
```

**SAST文本检测模型推理，需要设置参数`det_algorithm`，指定检测算法类型为SAST**，可以执行如下命令：
```
python3 tools/infer/predict_det.py --det_algorithm="SAST" --image_dir="./doc/imgs_en/img623.jpg" --det_model_dir="./inference/det_sast_tt/" --det_sast_polygon=True
```
可视化文本检测结果默认保存到`./inference_results`文件夹里面，结果文件的名称前缀为'det_res'。结果示例如下：

![](../imgs_results/det_res_img623_sast.jpg)

**注意**：本代码库中，SAST后处理Locality-Aware NMS有python和c++两种版本，c++版速度明显快于python版。由于c++版本nms编译版本问题，只有python3.5环境下会调用c++版nms，其他情况将调用python版nms。


<a name="文本识别模型推理"></a>
## 三、文本识别模型推理

下面将介绍超轻量中文识别模型推理、基于CTC损失的识别模型推理和基于Attention损失的识别模型推理。对于中文文本识别，建议优先选择基于CTC损失的识别模型，实践中也发现基于Attention损失的效果不如基于CTC损失的识别模型。此外，如果训练时修改了文本的字典，请参考下面的自定义文本识别字典的推理。


<a name="超轻量中文识别模型推理"></a>
### 1. 超轻量中文识别模型推理

超轻量中文识别模型推理，可以执行如下命令：

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words/ch/word_4.jpg" --rec_model_dir="./inference/rec_crnn/"
```

![](../imgs_words/ch/word_4.jpg)

执行命令后，上面图像的预测结果（识别的文本和得分）会打印到屏幕上，示例如下：

Predicts of ./doc/imgs_words/ch/word_4.jpg:['实力活力', 0.89552695]


<a name="基于CTC损失的识别模型推理"></a>
### 2. 基于CTC损失的识别模型推理

我们以STAR-Net为例，介绍基于CTC损失的识别模型推理。 CRNN和Rosetta使用方式类似，不用设置识别算法参数rec_algorithm。

首先将STAR-Net文本识别训练过程中保存的模型，转换成inference model。以基于Resnet34_vd骨干网络，使用MJSynth和SynthText两个英文文本识别合成数据集训练
的模型为例（[模型下载地址](https://paddleocr.bj.bcebos.com/rec_r34_vd_tps_bilstm_ctc.tar))，可以使用如下命令进行转换：

```
# -c后面设置训练算法的yml配置文件
# Global.checkpoints参数设置待转换的训练模型地址，不用添加文件后缀.pdmodel，.pdopt或.pdparams。
# Global.save_inference_dir参数设置转换的模型将保存的地址。

python3 tools/export_model.py -c configs/rec/rec_r34_vd_tps_bilstm_ctc.yml -o Global.checkpoints="./models/rec_r34_vd_tps_bilstm_ctc/best_accuracy" Global.save_inference_dir="./inference/starnet"
```

STAR-Net文本识别模型推理，可以执行如下命令：

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./inference/starnet/" --rec_image_shape="3, 32, 100" --rec_char_type="en"
```

<a name="基于Attention损失的识别模型推理"></a>
### 3. 基于Attention损失的识别模型推理

基于Attention损失的识别模型与ctc不同，需要额外设置识别算法参数 --rec_algorithm="RARE"

RARE 文本识别模型推理，可以执行如下命令：
```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./inference/rare/" --rec_image_shape="3, 32, 100" --rec_char_type="en" --rec_algorithm="RARE"
```

![](../imgs_words_en/word_336.png)

执行命令后，上面图像的识别结果如下：

Predicts of ./doc/imgs_words_en/word_336.png:['super', 0.9999555]

**注意**：由于上述模型是参考[DTRB](https://arxiv.org/abs/1904.01906)文本识别训练和评估流程，与超轻量级中文识别模型训练有两方面不同：

- 训练时采用的图像分辨率不同，训练上述模型采用的图像分辨率是[3，32，100]，而中文模型训练时，为了保证长文本的识别效果，训练时采用的图像分辨率是[3, 32, 320]。预测推理程序默认的的形状参数是训练中文采用的图像分辨率，即[3, 32, 320]。因此，这里推理上述英文模型时，需要通过参数rec_image_shape设置识别图像的形状。

- 字符列表，DTRB论文中实验只是针对26个小写英文本母和10个数字进行实验，总共36个字符。所有大小字符都转成了小写字符，不在上面列表的字符都忽略，认为是空格。因此这里没有输入字符字典，而是通过如下命令生成字典.因此在推理时需要设置参数rec_char_type，指定为英文"en"。

```
self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
dict_character = list(self.character_str)
```

<a name="自定义文本识别字典的推理"></a>
### 4. 自定义文本识别字典的推理
如果训练时修改了文本的字典，在使用inference模型预测时，需要通过`--rec_char_dict_path`指定使用的字典路径

```
python3 tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir="./your inference model" --rec_image_shape="3, 32, 100" --rec_char_type="en" --rec_char_dict_path="your text dict path"
```

<a name="文本检测、识别串联推理"></a>
## 四、文本检测、识别串联推理
<a name="超轻量中文OCR模型推理"></a>
### 1. 超轻量中文OCR模型推理

在执行预测时，需要通过参数image_dir指定单张图像或者图像集合的路径、参数det_model_dir指定检测inference模型的路径和参数rec_model_dir指定识别inference模型的路径。可视化识别结果默认保存到 ./inference_results 文件夹里面。

```
python3 tools/infer/predict_system.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./inference/det_db/"  --rec_model_dir="./inference/rec_crnn/"
```

执行命令后，识别结果图像如下：

![](../imgs_results/2.jpg)

<a name="其他模型推理"></a>
### 2. 其他模型推理

如果想尝试使用其他检测算法或者识别算法，请参考上述文本检测模型推理和文本识别模型推理，更新相应配置和模型。

**注意：由于检测框矫正逻辑的局限性，SAST弯曲文本检测模型（即，使用参数`--det_sast_polygon=True`时）暂时无法用来模型串联。**

下面给出基于EAST文本检测和STAR-Net文本识别执行命令：

```
python3 tools/infer/predict_system.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference/det_east/" --det_algorithm="EAST" --rec_model_dir="./inference/starnet/" --rec_image_shape="3, 32, 100" --rec_char_type="en"
```

执行命令后，识别结果图像如下：

![](../imgs_results/img_10.jpg)
