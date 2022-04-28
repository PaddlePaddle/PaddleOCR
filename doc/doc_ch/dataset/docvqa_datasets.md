## DocVQA数据集
这里整理了常见的DocVQA数据集，持续更新中，欢迎各位小伙伴贡献数据集～
- [FUNSD数据集](#funsd)
- [XFUND数据集](#xfund)

<a name="funsd"></a>
#### 1、FUNSD数据集
- **数据来源**：https://guillaumejaume.github.io/FUNSD/
- **数据简介**：FUNSD数据集是一个用于表单理解的数据集，它包含199张真实的、完全标注的扫描版图片，类型包括市场报告、广告以及学术报告等，并分为149张训练集以及50张测试集。FUNSD数据集适用于多种类型的DocVQA任务，如字段级实体分类、字段级实体连接等。部分图像以及标注框可视化如下所示:
<div align="center">
    <img src="../../datasets/funsd_demo/gt_train_00040534.jpg" width="500">
    <img src="../../datasets/funsd_demo/gt_train_00070353.jpg" width="500">
</div>
    图中，橙色区域代表`header`，淡蓝色区域代表`question`, 绿色区域表`answer`，粉红色代区域表`other`。

- **下载地址**：https://guillaumejaume.github.io/FUNSD/download/

<a name="xfund"></a>
#### 2、XFUND数据集
- **数据来源**：https://github.com/doc-analysis/XFUND
- **数据简介**：XFUND是一个多语种表单理解数据集，它包含7种不同语种的表单数据，并且全部用人工进行了键-值对形式的标注。其中每个语种的数据都包含了199张表单数据，并分为149张训练集以及50张测试集。部分图像以及标注框可视化如下所示:
<div align="center">
    <img src="../../datasets/xfund_demo/gt_zh_train_0.jpg" width="500">
    <img src="../../datasets/xfund_demo/gt_zh_train_1.jpg" width="500">
</div>

- **下载地址**：https://github.com/doc-analysis/XFUND/releases/tag/v1.0
