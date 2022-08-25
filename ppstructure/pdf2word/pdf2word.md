# PDF2Word

PDF2Word是PaddleOCR社区开发者[whjdark](https://github.com/whjdark) 基于PP-Structure智能文档分析模型实现的PDF转换Word应用程序，提供可直接安装的exe，方便windows用户运行

## 1.使用

### 应用程序

1. 下载与安装：针对Windows用户，根据[软件下载]()一节下载软件后，运行 `启动程序.exe` 。若您下载的是lite版本，安装过程中会在线下载环境依赖、模型等必要资源，安装时间较长，请确保网络畅通。serve版本打包了相关依赖，安装时间较短，可按需下载。

2. 转换：由于PP-Structure根据中英文数据分别进行适配，在转换相应文件时可**根据文档语言进行相应选择**。

### 脚本运行

首次运行需要将切换路径到 `/ppstructure/pdf2word` ，然后运行代码

```
python pdf2word.py
```

## 2.软件下载

如需获取已打包程序，可以扫描下方二维码，关注公众号填写问卷后，加入PaddleOCR官方交流群免费获取20G OCR学习大礼包，内含OCR场景应用集合（包含数码管、液晶屏、车牌、高精度SVTR模型等7个垂类模型）、《动手学OCR》电子书、课程回放视频、前沿论文等重磅资料

<div align="center">
<img src="https://user-images.githubusercontent.com/50011306/186369636-35f2008b-df5a-4784-b1f5-cebebcb2b7a5.jpg"  width = "150" height = "150" />
</div>

