# PDF2Word

PDF2Word是PaddleOCR社区开发者 [whjdark](https://github.com/whjdark) 基于PP-StructureV2版面分析与恢复模型实现的PDF转换Word应用程序，提供可直接安装的exe应用程序，**方便Windows用户离线、免环境配置运行**

## 1.使用

### 应用程序

1. 下载与安装：针对Windows用户，根据[软件下载](#download)一节下载软件并解压后，运行 `启动程序.exe` 。

2. **打开文件与转换：**

   - `中文转换、英文转换` ：针对 `图片型PDF` 文件的转换方法，即**当PDF文件中的文字无法复制粘贴时**，推荐使用本方法通过OCR转换文件，由于PP-Structure根据中英文数据分别进行适配，在转换相应文件时可**根据文档语言进行相应选择**。
   - `PDF解析` ： 针对可以复制文字的PDF文件，推荐直接点击 `PDF解析`，获得更加精准的效果。

3. 打开结果：点击`显示结果`，即可打开转换完成后的文件夹

> 注意：
>
> - 初次安装程序根据不同设备需要等待1-2分钟不等
> - 使用Office与WPS打开的Word结果会出现不同，推荐以Office为准
> - 本程序使用 [QPT](https://github.com/QPT-Family/QPT) 进行应用程序打包，感谢 [GT-ZhangAcer](https://github.com/GT-ZhangAcer) 对打包过程的支持

### 脚本启动界面

首次运行需要将切换路径到 `/ppstructure/pdf2word` ，然后运行代码

```
python pdf2word.py
```

### PaddleOCR whl包

针对Linux、Mac用户或已经拥有Python环境的用户，**推荐安装 `paddleocr` whl包直接应用版面恢复功能**，详情可查看[链接](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppstructure/docs/quickstart.md)

<a name="download"></a>

## 2.软件下载

如需获取已打包程序，可以扫描下方二维码，关注公众号填写问卷后，加入PaddleOCR官方交流群免费获取20G OCR学习大礼包，内含OCR场景应用集合（包含数码管、液晶屏、车牌、高精度SVTR模型等10个垂类模型）、《动手学OCR》电子书、课程回放视频、前沿论文等重磅资料

<div align="center">
<img src="https://user-images.githubusercontent.com/50011306/186369636-35f2008b-df5a-4784-b1f5-cebebcb2b7a5.jpg"  width = "150" height = "150" />
</div>
## 3.版本说明

v0.2版：新加入PDF解析功能，仅提供full版本，打包了所有依赖包与模型文件，尽可能避免安装失败问题

v0.1版：最初版本，分为3个版本：

- mini版：体积较小，在安装过程中会自动下载依赖包、模型等必要资源，安装时间较长，需要确保网络畅通。
- env版：仅打包了项目依赖，避免出现运行过程中找不到cv等资源的情况。
- full版：打包了依赖包与模型文件，故压缩包较大，相对等待时间较短，可按需下载。
