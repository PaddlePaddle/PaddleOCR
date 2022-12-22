# PDF2WORD

PDF2Word是PaddleOCR社区开发者 [whjdark](https://github.com/whjdark) 基于PP-StructureV2版面分析与恢复模型实现的PDF转换Word应用程序，提供可直接安装的exe应用程序，**方便Windows用户免环境配置运行**

## 1.使用

### 应用程序

1. 下载与安装：针对Windows用户，根据[软件下载]()一节下载软件后，运行 `pdf2word.exe` 。若您下载的是lite版本，安装过程中会在线下载环境依赖、模型等必要资源，安装时间较长，请确保网络畅通。serve版本打包了相关依赖，安装时间较短，可按需下载。

2. 转换：由于PP-Structure根据中英文数据分别进行适配，在转换相应文件时可**根据文档语言进行相应选择**。

### 脚本运行

3. 打开结果：点击`显示结果`，即可打开转换完成后的文件夹

> 注意：
>
> - 初次安装程序根据不同设备需要等待1-2分钟不等
> - 使用Office与WPS打开的Word结果会出现不同，推荐以Office为准
> - 本程序使用 [QPT](https://github.com/QPT-Family/QPT) 进行应用程序打包，感谢 [GT-ZhangAcer](https://github.com/GT-ZhangAcer) 对打包过程的支持
> - 应用程序仅支持正版win10，11系统，不支持盗版Windows系统，若在安装过程中出现报错或缺少依赖，推荐直接使用 `paddleocr` whl包应用PDF2Word功能，详情可查看[链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/quickstart.md)

### 脚本启动界面

首次运行需要将切换路径到PaddleOCR文件目录 ，然后运行代码

```
cd ./ppstructure/pdf2word
python pdf2word.py
```

### PaddleOCR whl包

针对Linux、Mac用户或已经拥有Python环境的用户，**推荐安装 `paddleocr` whl包直接应用PDF2Word功能**，详情可查看[链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/quickstart.md)

<a name="download"></a>

## 2.软件下载

如需获取已打包程序，可以扫描下方二维码，关注公众号填写问卷后，加入PaddleOCR官方交流群免费获取20G OCR学习大礼包，内含OCR场景应用集合（包含数码管、液晶屏、车牌、高精度SVTR模型等7个垂类模型）、《动手学OCR》电子书、课程回放视频、前沿论文等重磅资料

<div align="center">
<img src="https://user-images.githubusercontent.com/50011306/186369636-35f2008b-df5a-4784-b1f5-cebebcb2b7a5.jpg"  width = "150" height = "150" />
</div>

## 3.版本说明

v0.2版：新加入PDF解析功能，仅提供full版本，打包了所有依赖包与模型文件，尽可能避免安装失败问题。若仍然安装失败，推荐使用 `paddleocr` whl包
