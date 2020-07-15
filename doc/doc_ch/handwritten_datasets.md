# 手写OCR数据集
这里整理了常用手写数据集，持续更新中，欢迎各位小伙伴贡献数据集～
- [中科院自动化研究所-手写中文数据集](#中科院自动化研究所-手写中文数据集)
- [NIST手写单字数据集-英文](#NIST手写单字数据集-英文)

<a name="中科院自动化研究所-手写中文数据集"></a>
## 中科院自动化研究所-手写中文数据集
- **数据来源**：http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
- **数据简介**：
    * 包含在线和离线两类手写数据，`HWDB1.0~1.2`总共有3895135个手写单字样本，分属7356类（7185个汉字和171个英文字母、数字、符号）；`HWDB2.0~2.2`总共有5091页图像，分割为52230个文本行和1349414个文字。所有文字和文本样本均存为灰度图像。部分单字样本图片如下所示。

        ![](../datasets/CASIA_0.jpg)
        (a) 五张单字图片样例

- **下载地址**：http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html
- **使用建议**：数据为单字，白色背景，可以大量合成文字行进行训练。白色背景可以处理成透明状态，方便添加各种背景。对于需要语义的情况，建议从真实语料出发，抽取单字组成文字行


<a name="NIST手写单字数据集-英文"></a>
## NIST手写单字数据集-英文(NIST Handprinted Forms and Characters Database)

- **数据来源**: [https://www.nist.gov/srd/nist-special-database-19](https://www.nist.gov/srd/nist-special-database-19)

- **数据简介**: NIST19数据集适用于手写文档和字符识别的模型训练，从3600位作者的手写样本表格中提取得到，总共包含81万张字符图片。其中9张图片示例如下。

    ![](../datasets/nist_demo.png)


- **下载地址**: [https://www.nist.gov/srd/nist-special-database-19](https://www.nist.gov/srd/nist-special-database-19)
