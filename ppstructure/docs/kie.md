

# 关键信息提取(Key Information Extraction)

本节介绍PaddleOCR中关键信息提取SDMGR方法的快速使用和训练方法。

SDMGR是一个关键信息提取算法，将每个检测到的文本区域分类为预定义的类别，如订单ID、发票号码，金额等。


* [1. 快速使用](#1-----)
* [2. 执行训练](#2-----)
* [3. 执行评估](#3-----)

<a name="1-----"></a>
## 1. 快速使用

训练和测试的数据采用wildreceipt数据集，通过如下指令下载数据集：

```
wget https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/wildreceipt.tar && tar xf wildreceipt.tar
```

执行预测：

```
cd PaddleOCR/
wget https://paddleocr.bj.bcebos.com/dygraph_v2.1/kie/kie_vgg16.tar && tar xf kie_vgg16.tar
python3.7 tools/infer_kie.py -c configs/kie/kie_unet_sdmgr.yml -o Global.checkpoints=kie_vgg16/best_accuracy  Global.infer_img=../wildreceipt/1.txt
```

执行预测后的结果保存在`./output/sdmgr_kie/predicts_kie.txt`文件中，可视化结果保存在`/output/sdmgr_kie/kie_results/`目录下。

可视化结果如下图所示：
[img](./imgs/0.png)

<a name="2-----"></a>
## 2. 执行训练

创建数据集软链到PaddleOCR/train_data目录下：
```
cd PaddleOCR/ && mkdir train_data && cd train_data

ln -s ../../wildreceipt ./
```

训练采用的配置文件是configs/kie/kie_unet_sdmgr.yml，配置文件中默认训练数据路径是`train_data/wildreceipt`，准备好数据后，可以通过如下指令执行训练：
```
python3.7 tools/train.py -c configs/kie/kie_unet_sdmgr.yml -o Global.save_model_dir=./output/kie/
```
<a name="3-----"></a>
## 3. 执行评估

```
python3.7 tools/eval.py -c configs/kie/kie_unet_sdmgr.yml -o Global.checkpoints=./output/kie/best_accuracy
```


**参考文献：**

<!-- [ALGORITHM] -->

```bibtex
@misc{sun2021spatial,
      title={Spatial Dual-Modality Graph Reasoning for Key Information Extraction},
      author={Hongbin Sun and Zhanghui Kuang and Xiaoyu Yue and Chenhao Lin and Wayne Zhang},
      year={2021},
      eprint={2103.14470},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
