# PaddleX表格结构识别任务模块数据标注教程

## 1. 数据标注
进行表格数据标注时，使用[PPOCRLabelv2](https://github.com/PFCCLab/PPOCRLabel/blob/main/README_ch.md)工具。详细步骤可以参考：[【视频演示】](https://www.bilibili.com/video/BV1wR4y1v7JE/?share_source=copy_web&vd_source=cf1f9d24648d49636e3d109c9f9a377d&t=1998)

表格标注针对表格的结构化提取，将图片中的表格转换为Excel格式，因此标注时需要配合外部软件打开Excel同时完成。在PPOCRLabel软件中完成表格中的文字信息标注（文字与位置）、在Excel文件中完成表格结构信息标注，推荐的步骤为：

1. 表格识别：打开表格图片后，点击软件右上角`表格识别`按钮，软件调用PP-Structure中的表格识别模型，自动为表格打标签，同时弹出Excel
2. 更改标注结果：**以表格中的单元格为单位增加标注框**（即一个单元格内的文字都标记为一个框）。标注框上鼠标右键后点击`单元格重识别`可利用模型自动识别单元格内的文字。
3. **调整单元格顺序**：点击软件`视图-显示框编号`打开标注框序号，在软件界面右侧拖动`识别结果`一栏下的所有结果，使得标注框编号按照从左到右，从上到下的顺序排列，按行依次标注。
4. 标注表格结构：**在外部Excel软件中，将存在文字的单元格标记为任意标识符（如**`1`**）**，保证Excel中的单元格合并情况与原图相同即可（即不需要Excel中的单元格文字与图片中的文字完全相同）
5. 导出JSON格式：关闭所有表格图像对应的Excel，点击`文件`-`导出表格标注`，生成gt.txt标注文件。
## 2. 数据格式
PaddleX 针对表格识别任务定义的数据集，组织结构和标注格式如下：

```ruby
dataset_dir    # 数据集根目录，目录名称可以改变
├── images     # 图像的保存目录，目录名称可以改变，但要注意和train.txt val.txt的内容对应
├── train.txt  # 训练集标注文件，文件名称不可改变，内容举例：{"filename": "images/border.jpg", "html": {"structure": {"tokens": ["<tr>", "<td", " colspan=\"3\"", ">", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>"]}, "cells": [{"tokens": ["、", "自", "我"], "bbox": [[[5, 2], [231, 2], [231, 35], [5, 35]]]}, {"tokens": ["9"], "bbox": [[[168, 68], [231, 68], [231, 98], [168, 98]]]}]}, "gt": "<html><body><table><tr><td colspan=\"3\">、自我</td></tr><tr><td>Aghas</td><td>失吴</td><td>月，</td></tr><tr><td>lonwyCau</td><td></td><td>9</td></tr></table></body></html>"}
└── val.txt    # 验证集标注文件，文件名称不可改变，内容举例：{"filename": "images/no_border.jpg", "html": {"structure": {"tokens": ["<tr>", "<td", " colspan=\"2\"", ">", "</td>", "<td", " rowspan=\"2\"", ">", "</td>", "<td", " rowspan=\"2\"", ">", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>", "<tr>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "<td>", "</td>", "</tr>"]}, "cells": [{"tokens": ["a", "d", "e", "s"], "bbox": [[[0, 4], [284, 4], [284, 34], [0, 34]]]}, {"tokens": ["$", "7", "5", "1", "8", ".", "8", "3"], "bbox": [[[442, 67], [616, 67], [616, 100], [442, 100]]]}]}, "gt": "<html><body><table><tr><td colspan=\"2\">ades</td><td rowspan=\"2\">研究中心主任滕建</td><td rowspan=\"2\">品、家居用品位居商</td></tr><tr><td>naut</td><td>则是创办思</td></tr><tr><td>各方意见建议，确保</td><td>9.66</td><td>道开业，负责</td><td>$7518.83</td></tr></table></body></html>"}
```
标注文件采用 `PubTabNet` 数据集格式进行标注，每行内容都是一个`python`字典。

请大家参考上述规范准备数据，此外可以参考：[示例数据集](https://paddle-model-ecology.bj.bcebos.com/paddlex/data/table_rec_dataset_examples.tar) 