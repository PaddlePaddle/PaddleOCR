# 离线使用PPOCRLabel
首次启动PPOCRLabel时需要联网下载模型权重。如果无法联网就会报错。那么这里可以自己手动去官网下载权重，然后拷贝到离线主机中，按照对应目录结构放好即可使用。
通过如下网址，可以看到官方提供的移动端以及服务器端使用的权重(这里默认是针对中英文模型，你也可以下载其他语言的模型)：
[https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/README_ch.md](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.1/README_ch.md)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210710185935259.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70#pic_center)
这里以下载服务器端的模型权重为例，下载服务器端对应的检测模型权重，方向分类器权重以及识别模型权重（注意是下载推理模型，不是预训练模型）。下载解压后会得到如下三个文件夹，每个文件夹中都有```inference.pdiparams```，```inference.pdiparams.info```以及```inference.pdmodel```三个文件：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210710190712210.png#pic_center)

如果是Linux系统，在自己用户下有一个```.paddleocr```的目录，如果没有自己创建一个，例如我的用户是```wz```那么有```/home/wz/.paddleocr```。
如果是Windows系统，在自己用户下同样会有一个```.paddleocr```的目录，如果没有自己创建一个，例如我的用户是```wz```那么有```C:\User\wz\.paddleocr```。
然后将刚刚下载好的权重依次摆放对应文件夹中。

```
├── .paddleocr: 
│    ├── cls: 存放分类器的权重（分类器不分语言）
│    │     ├── inference.pdiparams
│    │     ├── inference.pdiparams.info
│    │     └── inference.pdmodel
│    │
│    └── 2.1: 注意该目录的名称是根据当前release版本来的，当前使用的release版本是2.1 
│          ├── det: 存放检测器的权重
│          │     └── ch: 代表中英文模型
│          │         ├── inference.pdiparams
│          │         ├── inference.pdiparams.info
│          │         └── inference.pdmodel
│          │
│          └── rec: 存放识别器的权重
│               └── ch: 代表中英文模型
│                     ├── inference.pdiparams
│                     ├── inference.pdiparams.info
│                     └── inference.pdmodel
```

权重摆放完成后，在PPOCRLabel文件下打开终端，通过指令再次启动即可成功使用:
```
python PPOCRLabel.py --lang ch
```
