# 基于Doctr++的文档矫正

本项目基于[Doctr++源代码](https://github.com/fh2019ustc/DocTr-Plus)和开源项目[DocTrPP](https://github.com/GreatV/DocTrPP?tab=readme-ov-file)开发和实现

- [1. 项目背景及意义](#1-项目背景及意义)
- [2. 项目内容](#2-项目内容)
- [3. Doctr++文档矫正算法介绍](#3-Doctr++文档矫正算法介绍)
  - [3.1 动机](#31-动机)
  - [3.2 创新点](#32-创新点)
  - [3.3 现有方法](#33-现有方法)
  - [3.4 网络结构](#34-网络结构)
  - [3.5 代码分析](#35-代码分析)
- [4. 安装环境](#4-安装环境)
- [5. 数据准备](#5-数据准备)
- [6. 模型训练](#6-模型训练)
- [7. 模型评估](#7-模型评估)


## 1. 项目背景及意义
如今，智能手机的普及使得直接使用智能手机来数字化文档文件的趋势越来越流行。相比传统的平板扫描仪，智能手机提供了更为灵活、便携和直接的文档图像数字化替代方案。然而，由于一些无法控制的因素，例如文档的物理形变、光照条件和相机角度等，捕捉到的文档图像不可避免地会出现失真。这种失真影响了文档的数字存储，并可能对后续的应用产生负面影响，如自动文本识别、分析、检索和问答等。在过去几十年里，文档图像校正的研究一直积极进行。但现有的先进算法仅限于处理受限的文档图像，即输入图像必须包含一个完整的文档。一旦捕获的图像仅涉及局部文本区域，其校正质量就会降低并且不令人满意。因此急需一种能够处理任意文档图像的算法。

![example](https://raw.githubusercontent.com/chenjjcccc/image/main/doctr%2B%2B_img1.png)

## 2. 项目内容
[DocTr++](https://arxiv.org/pdf/2304.08796.pdf)目前最优的文档矫正方法，由中科院研发，第一篇对输入不限制的基于学习的文档矫正方法。本项目就是基于PaddlePaddle框架实现Doctr++文档矫正算法的复现。合并到PaddleOCR项目当中。


## 3. Doctr++文档矫正算法介绍
### 3.1 动机
现有的先进算法仅限于处理受限的文档图像，即输入图像必须包含一个完整的文档。一旦捕获的图像仅涉及局部文本区域，其校正质量就会降低并且不令人满意。作者之前提出的DocTr，一种用于文档图像校正的变压器辅助网络，也受到这种限制，因此作者提出了一种新型文档矫正框架DocTr++，该框架可以处理任意文档图像输入。下面展示的就是三种不同的输入文档图像。（a）具有完整的文档边界，（b）具有部分文档边界，以及（c）没有任何文档边界

![example](https://raw.githubusercontent.com/chenjjcccc/image/main/doctr%2B%2B_img2.png)

### 3.2 创新点

1. 升级结构为层级的encoder-decoder，进行特征提取和分割
2. 调整了像素级别的映射关系：矫正前的扭曲图像（不做是否为整篇文档的限制）和矫正后的图像：当矫正后的图片像素源自矫正前的文档外区域，会被直接设置为一个零值，也就是矫正后的黑底。
3. 给出了一个现实世界的测试集和评价指标来验证矫正的质量。
  a. UDIR 训练集， Doc3D的拓展，其训练集包含10万张扭曲和矫正的图像；UDIR进一步裁减边缘
  b. UDIR 测试集，DocUNet Benchmark dataset的拓展，通过补充数据和裁减，保证三类情况各占1/3

### 3.3 现有方法

1. 3D重建：
  - a. 建立被拍摄文档的3D表征，并将其映射到一个没有扭曲的平面。
  - b. 需要拍摄多个平面/额外的激光扫描器的支持，不可拓展。
2. 提取变形表面参数的模型：
  - a. 提取阴影、边界、文本线等参数用于矫正。
  - b. 效果有限、计算量也不小。
3. 深度学习方法：网络学习像素位移场，重采样待矫正的图像，快速矫正
  - a. 输入需要包含所有完整的边缘，否则矫正效果差（why？包含完整边缘的才有完整的角点和参考点信息）
  - b. DocProj 将图像分割成多个patch，解决上述问题，但是计算复杂度大，且不能包含有背景的图像（上图3）。
  - c. DDCP预估几个控制点和参考点+TPS差值算法来矫正。
  - d. PWUNet考虑本地patch的不同扭曲程度和提升全局的矫正效果。

### 3.4 网络结构

![example](https://raw.githubusercontent.com/chenjjcccc/image/main/doctr%2B%2B_img3.png)
1. CNN提取2D特征，8倍下采样。
2. 扭曲编码器：学习结构信息，关注全局的弯曲文本线、纹理等。增加位置编码信息position embedding，因为transformer时扰动不变的。普通的transformer 配置
3. 序列化网络进一步进行序列关注编码。
  a. 编码解码器详细部分如下所示：
  ![example](https://raw.githubusercontent.com/chenjjcccc/image/main/doctr%2B%2B_img4.png)
4. 可学习的queries来提升对不同部分的关注度（相当于学习的关注向量，关注那一部分的特征重要），获得每个patch的warp flow。输出为相对原图八倍下采样。
5. flow head：精细化学习warp flow，进一步加权。两层卷积网络学习D6获得fm，另外有一个分支基于D6学习H/8 × W/8 × 8 × 8 × 9 的权重矩阵加权fm，H*W*2（垂直和水平的warp矩阵。）
6. 基于warp flow就可以对待矫正的图像进行重新矫正，矫正结果利用L1损失进行约束。

### 3.5 代码分析
- 完整网络流程
```python
def forward(self, image):
    fmap = self.fnet(image)
    fmap = F.relu(fmap)

    fmap1= self.__getattr__(self.encoder_block[0])(fmap)
    fmap1d = self.__getattr__(self.down_layer[0])(fmap1)
    fmap2= self.__getattr__(self.encoder_block[1])(fmap1d)
    fmap2d = self.__getattr__(self.down_layer[1])(fmap2)
    fmap3= self.__getattr__(self.encoder_block[2])(fmap2d)

    query_embed0 = self.query_embed.weight.unsqueeze(1).tile([1, fmap3.shape[0], 1])

    fmap3d_ = self.__getattr__(self.decoder_block[0])(fmap3, query_embed0)
    fmap3du_ = (
        self.__getattr__(self.up_layer[0])(fmap3d_)
        .flatten(2)
        .transpose(perm=[2, 0, 1])
    )
    fmap2d_ = self.__getattr__(self.decoder_block[1])(fmap2, fmap3du_)
    fmap2du_ = (
        self.__getattr__(self.up_layer[1])(fmap2d_)
        .flatten(2)
        .transpose(perm=[2, 0, 1])
    )
    fmap_out = self.__getattr__(self.decoder_block[2])(fmap1, fmap2du_)

    coodslar, coords0, coords1 = self.initialize_flow(image)
    coords1 = coords1.detach()
    mask, coords1 = self.update_block(fmap_out, coords1)
    flow_up = self.upsample_flow(coords1 - coords0, mask)
    bm_up = coodslar + flow_up

    return bm_up

```
- 编码模块
```python
class TransEncoder(nn.Layer):
    def __init__(self, num_attn_layers: int, hidden_dim: int = 128):
        super(TransEncoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, image: paddle.Tensor):
        pos = self.position_embedding(
            paddle.ones([image.shape[0], image.shape[2], image.shape[3]], dtype="bool").cuda()
        )
        b, c, h, w = image.shape

        image = image.flatten(2).transpose(perm=[2, 0, 1])
        pos = pos.flatten(2).transpose(perm=[2, 0, 1])

        for layer in self.layers:
            image= layer(image, [image], pos=pos, memory_pos=[pos, pos])
        image = image.transpose(perm=[1, 2, 0]).reshape([b, c, h, w])

        return image
```
- 解码模块
```python
class TransDecoder(nn.Layer):
    def __init__(self, num_attn_layers: int, hidden_dim: int = 128):
        super(TransDecoder, self).__init__()
        attn_layer = attnLayer(hidden_dim)
        self.layers = _get_clones(attn_layer, num_attn_layers)
        self.position_embedding = build_position_encoding(hidden_dim)

    def forward(self, image: paddle.Tensor, query_embed: paddle.Tensor):
        pos = self.position_embedding(
            paddle.ones([image.shape[0], image.shape[2], image.shape[3]], dtype="bool").cuda()
        )
        b, c, h, w = image.shape

        image = image.flatten(2).transpose(perm=[2, 0, 1])
        pos = pos.flatten(2).transpose(perm=[2, 0, 1])

        for layer in self.layers:
            query_embed = layer(query_embed, [image], pos=pos, memory_pos=[pos, pos])

        query_embed = query_embed.transpose(perm=[1, 2, 0]).reshape([b, c, h, w])

        return query_embed
```
## 4. 安装环境

```python
# 首先git官方的PaddleOCR项目，安装需要的依赖
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR
pip install -r requirements.txt
cd applications/Doctr++文档矫正
pip install -r requirements.txt
```

## 5. 数据准备
本项目使用公开的文档矫正数据集doc3d进行训练，数据集地址：https://github.com/cvlab-stonybrook/doc3D-dataset/tree/master, 它包含10万张失真的文件图像及其无失真的地面真相。本项目将数据集划分为训练集(80%)和验证集(20%)进行训练。具体下载方式参考数据集链接。


```python
#下载并解压数据
chmod +x load_dataset.sh
./load_dataset.sh output_dir
# 划分数据集
python split_dataset.py --data_root ./doc3d --train_ratio 0.8
```
数据格式如下：
```
doc3d/
|-- img
|   ├── 1
|   |   ├── img_001.jpg
|   |   ├── img_002.jpg
|   |   ...
|   ├── 2
|   |   ...
|   └── 21
|-- wc
|   ├── 1
|   |   ├── img_001.exr
|   |   ├── img_002.exr
|   |   ...
|   ├── 2
|   |   ...
|   └── 21
|-- bm
|   ├── 1
|   |   ├── img_001.mat
|   |   ├── img_002.mat
|   |   ...
|   ├── 2
|   |   ...
|   └── 21
```

除此之外，本项目使用[DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html)数据集作为测试集，具体下载方式参考(https://github.com/doc-analysis/DocUNet)。下载解压后数据格式如下：

```
 DocUNet/
|-- crop
|   ├── 1_1 copy.png
|   ├── 1_2 copy.png
|   ├── 2_1 copy.png
|   ├── 2_2 copy.png
|   |   ...
|
|-- scan
|   ├── 1.png
|   ├── 2.png
|   |   ...
```

## 6. 模型训练
准备工作完成后，即可开始进行模型训练。本项目提供训练脚本train.sh。其中需要自行修改数据集路径--data-root
```python
# 模型训练
export OPENCV_IO_ENABLE_OPENEXR=1
export FLAGS_logtostderr=0
export CUDA_VISIBLE_DEVICES=4

python train.py --data-root ./doc3d \
    --img-size 288 \
    --name "DocTr++" \
    --batch-size 12 \
    --lr 1e-4 \
    --exist-ok \
    --epochs 150 \
```
每个epoch后，会进行验证集评估，保存best_model以及last_model。
## 7. 模型评估
在训练之前，我们可以直接使用下面命令来评估预训练模型的效果,预训练模型下载地址：https://github.com/

### 单张图预测
```python
python predict.py --i image_path --m pretrained_model_path
```
### DOCUnet测试集评估
```python
python eval_DocUNet.py --i ./crop/ --m pretrained_model_path  --o output_val_path --g ./scan/
```
效果如下：

![example](https://raw.githubusercontent.com/chenjjcccc/image/main/doctr%2B%2B_img5.png)
