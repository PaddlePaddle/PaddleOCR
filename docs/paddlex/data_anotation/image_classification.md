# PaddleX图像分类任务模块数据标注教程

本文档将介绍如何使用[Labelme](https://github.com/wkentaro/labelme)标注工具完成图像分类相关单模型的数据标注。 
点击上述链接，参考⾸⻚⽂档即可安装数据标注⼯具并查看详细使⽤流程。

## 1. Labelme 标注
### 1.1 Labelme 标注工具介绍
`Labelme` 是一个 `python` 语言编写，带有图形界面的图像标注软件。可用于图像分类、目标检测、图像分割等任务，在实例分割的标注任务中，标签存储为 `JSON` 文件。

### 1.2 Labelme 安装
为避免环境冲突，建议在 `conda` 环境下安装。.

```bash
conda create -n labelme python=3.10
conda activate labelme
pip install pyqt5
pip install labelme
```
### 1.3 Labelme 标注过程
#### 1.3.1 准备待标注数据
* 创建数据集根目录，如 `pets`。
* 在 `pets` 中创建 `images` 目录（必须为`images`目录），并将待标注图片存储在 `images` 目录下，如下图所示：

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/image_classification/01.png)

* 在 `pets` 文件夹中创建待标注数据集的类别标签文件 `flags.txt`，并在 `flags.txt` 中按行写入待标注数据集的类别。以猫狗分类数据集的 `flags.txt` 为例，如下图所示：

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/image_classification/02.png)
#### 1.3.2 启动 Labelme
终端进入到待标注数据集根目录，并启动 `labelme` 标注工具。

```bash
cd path/to/pets
labelme images --nodata --autosave --output annotations --flags flags.txt
```
* `flags` 为图像创建分类标签，传入标签路径。
* `nodata` 停止将图像数据存储到 JSON 文件。
* `autosave` 自动存储。
* `ouput` 标签文件存储路径。
#### 1.3.3 开始图片标注
* 启动 `labelme` 后如图所示：

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/image_classification/03.png)
* 在 `Flags` 界面选择类别。

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/image_classification/04.png)

* 标注好后点击存储。（若在启动 `labelme` 时未指定 `output` 字段，会在第一次存储时提示选择存储路径，若指定 `autosave` 字段使用自动保存，则无需点击存储按钮）。

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/image_classification/05.png)
* 然后点击 `Next Image` 进行下一张图片的标注。

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/image_classification/06.png)

* 完成全部图片的标注后，使用[convert_to_imagenet.py](https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/doc_images/applications/image_classification_dataset_prepare/convert_to_imagenet.py)脚本将标注好的数据集转换为 `ImageNet-1k` 数据集格式，生成 `train.txt`，`val.txt` 和`label.txt`。

```bash
python convert_to_imagenet.py --dataset_path /path/to/dataset
```
`dataset_path`为标注的 `labelme` 格式分类数据集。

* 经过整理得到的最终目录结构如下：

![alt text](https://raw.githubusercontent.com/cuicheng01/PaddleX_doc_images/main/images/data_prepare/image_classification/07.png)
##  2. 数据格式
* PaddleX 针对图像分类任务定义的数据集，名称是 **ClsDataset**，组织结构和标注格式如下：

```bash
dataset_dir    # 数据集根目录，目录名称可以改变
├── images     # 图像的保存目录，目录名称可以改变，但要注意与train.txt、val.txt的内容对应
├── label.txt  # 标注id和类别名称的对应关系，文件名称不可改变。每行给出类别id和类别名称，内容举例：45 wallflower
├── train.txt  # 训练集标注文件，文件名称不可改变。每行给出图像路径和图像类别id，使用空格分隔，内容举例：images/image_06765.jpg 0
└── val.txt    # 验证集标注文件，文件名称不可改变。每行给出图像路径和图像类别id，使用空格分隔，内容举例：images/image_06767.jpg 10
```
* 如果您已有数据集且数据集格式为如下格式，但是没有标注文件，可以使用[脚本](https://paddleclas.bj.bcebos.com/tools/create_cls_trainval_lists.py)将已有的数据集生成标注文件。

```bash
dataset_dir          # 数据集根目录，目录名称可以改变  
├── images           # 图像的保存目录，目录名称可以改变
   ├── train         # 训练集目录，目录名称可以改变
      ├── class0     # 类名字，最好是有意义的名字，否则生成的类别映射文件label.txt无意义
         ├── xxx.jpg # 图片，此处支持层级嵌套
         ├── xxx.jpg # 图片，此处支持层级嵌套
         ...  
      ├── class1     # 类名字，最好是有意义的名字，否则生成的类别映射文件label.txt无意义
      ...
   ├── val           # 验证集目录，目录名称可以改变
```
* 如果您使用的是 PaddleX 2.x版本的图像分类数据集，在经过训练集/验证集/测试集切分后，手动将 `train_list.txt`、`val_list.txt`、`test_list.txt`修改为`train.txt`、`val.txt`、`test.txt`，并且按照规则修改 `label.txt` 即可。

原版`label.txt`：

```bash
classname1
classname2
classname3
...
```
修改后的`label.txt`：

```bash
0 classname1
1 classname2
2 classname3
...
```
