# 配置文件内容与生成

[toc]

## 1. 可选参数列表

以下列表可以通过`--help`查看

|         FLAG             |     支持脚本    |        用途        |      默认值       |         备注         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  指定配置文件  |  None  |  **配置模块说明请参考 参数介绍** |
|          -o              |      ALL       |  设置配置文件里的参数内容  |  None  |  使用-o配置相较于-c选择的配置文件具有更高的优先级。例如：`-o Global.use_gpu=false`  |  


## 2. 配置文件参数介绍

以 `rec_chinese_lite_train_v2.0.yml ` 为例
### 2.1 Global

|         字段             |            用途                |      默认值       |            备注            |
| :----------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      use_gpu             |    设置代码是否在gpu运行           |       true        |                \                 |
|      epoch_num           |    最大训练epoch数             |       500        |                \                 |
|      log_smooth_window   |    log队列长度，每次打印输出队列里的中间值            |       20          |                \                 |
|      print_batch_step    |    设置打印log间隔         |       10          |                \                 |
|      save_model_dir      |    设置模型保存路径        |  output/{算法名称}  |                \                 |
|      save_epoch_step     |    设置模型保存间隔        |       3           |                \                 |
|      eval_batch_step     |    设置模型评估间隔        | 2000 或 [1000, 2000]        | 2000 表示每2000次迭代评估一次，[1000， 2000]表示从1000次迭代开始，每2000次评估一次   |
|      cal_metric_during_train     |    设置是否在训练过程中评估指标，此时评估的是模型在当前batch下的指标        |       true         |                \                 |
|      load_static_weights     |   设置预训练模型是否是静态图模式保存(目前仅检测算法需要)        |       true         |                \                 |
|      pretrained_model    |    设置加载预训练模型路径      |  ./pretrain_models/CRNN/best_accuracy  |  \          |
|      checkpoints         |    加载模型参数路径            |       None        |    用于中断后加载参数继续训练 |
|      use_visualdl  |    设置是否启用visualdl进行可视化log展示 |          False        |    [教程地址](https://www.paddlepaddle.org.cn/paddle/visualdl) |
|      infer_img            |    设置预测图像路径或文件夹路径     |       ./infer_img | \|
|      character_dict_path |    设置字典路径            |  ./ppocr/utils/ppocr_keys_v1.txt  |    \                 |
|      max_text_length     |    设置文本最大长度        |       25          |                \                 |
|      character_type      |    设置字符类型            |       ch          |    en/ch, en时将使用默认dict，ch时使用自定义dict|
|      use_space_char     |    设置是否识别空格             |        True      |          仅在 character_type=ch 时支持空格                 |
|      label_list          |    设置方向分类器支持的角度       |    ['0','180']    |     仅在方向分类器中生效 |
|      save_res_path          |    设置检测模型的结果保存地址       |    ./output/det_db/predicts_db.txt    |     仅在检测模型中生效 |

### Optimizer ([ppocr/optimizer](../../ppocr/optimizer))

|         字段             |            用途            |      默认值        |            备注             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      name        |         优化器类名          |  Adam  |  目前支持`Momentum`,`Adam`,`RMSProp`, 见[ppocr/optimizer/optimizer.py](../../ppocr/optimizer/optimizer.py)  |
|      beta1           |    设置一阶矩估计的指数衰减率  |       0.9         |               \             |
|      beta2           |    设置二阶矩估计的指数衰减率  |     0.999         |               \             |
|      clip_norm           |    所允许的二范数最大值  |              |               \             |
|      **lr**                |         设置学习率decay方式       |   -    |       \  |
|        name    |      学习率decay类名   |         Cosine       | 目前支持`Linear`,`Cosine`,`Step`,`Piecewise`, 见[ppocr/optimizer/learning_rate.py](../../ppocr/optimizer/learning_rate.py) |
|        learning_rate      |    基础学习率        |       0.001      |  \        |
|      **regularizer**      |  设置网络正则化方式        |       -      | \        |
|        name      |    正则化类名      |       L2     | 目前支持`L1`,`L2`, 见[ppocr/optimizer/regularizer.py](../../ppocr/optimizer/regularizer.py)        |
|        factor      |    学习率衰减系数       |       0.00004     |  \        |


### Architecture ([ppocr/modeling](../../ppocr/modeling))
在ppocr中，网络被划分为Transform,Backbone,Neck和Head四个阶段

|         字段             |            用途            |      默认值        |            备注             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      model_type        |         网络类型          |  rec  |  目前支持`rec`,`det`,`cls`  |
|      algorithm           |    模型名称  |       CRNN         |               支持列表见[algorithm_overview](./algorithm_overview.md)             |
|      **Transform**           |    设置变换方式  |       -       |               目前仅rec类型的算法支持, 具体见[ppocr/modeling/transform](../../ppocr/modeling/transform)              |
|        name    |      变换方式类名   |         TPS       | 目前支持`TPS` |
|        num_fiducial      |    TPS控制点数        |       20      |  上下边各十个       |
|        loc_lr      |    定位网络学习率        |       0.1      |  \      |
|        model_name      |    定位网络大小        |       small      |  目前支持`small`,`large`       |
|      **Backbone**      |  设置网络backbone类名        |       -      | 具体见[ppocr/modeling/backbones](../../ppocr/modeling/backbones)        |
|        name      |    backbone类名       |       ResNet     | 目前支持`MobileNetV3`,`ResNet`        |
|        layers      |    resnet层数       |       34     |  支持18,34,50,101,152,200       |
|        model_name      |    MobileNetV3 网络大小       |       small     |  支持`small`,`large`       |
|      **Neck**      |  设置网络neck        |       -      | 具体见[ppocr/modeling/necks](../../ppocr/modeling/necks)        |
|        name      |    neck类名       |       SequenceEncoder     | 目前支持`SequenceEncoder`,`DBFPN`        |
|        encoder_type      |    SequenceEncoder编码器类型       |       rnn     |  支持`reshape`,`fc`,`rnn`       |
|        hidden_size      |   rnn内部单元数       |       48     |  \      |
|        out_channels      |   DBFPN输出通道数       |       256     |  \      |
|      **Head**      |  设置网络Head        |       -      | 具体见[ppocr/modeling/heads](../../ppocr/modeling/heads)        |
|        name      |    head类名       |       CTCHead     | 目前支持`CTCHead`,`DBHead`,`ClsHead`        |
|        fc_decay      |    CTCHead正则化系数       |       0.0004     |  \      |
|        k      |   DBHead二值化系数       |       50     |  \      |
|        class_dim      |   ClsHead输出分类数       |       2     |  \      |


### Loss ([ppocr/losses](../../ppocr/losses))

|         字段             |            用途            |      默认值        |            备注             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      name        |         网络loss类名          |  CTCLoss  |  目前支持`CTCLoss`,`DBLoss`,`ClsLoss`  |
|      balance_loss        |        DBLossloss中是否对正负样本数量进行均衡(使用OHEM)         |  True  |  \  |
|      ohem_ratio        |        DBLossloss中的OHEM的负正样本比例         |  3  |  \  |
|      main_loss_type        |        DBLossloss中shrink_map所采用的的loss        |  DiceLoss  |  支持`DiceLoss`,`BCELoss`  |
|      alpha        |        DBLossloss中shrink_map_loss的系数       |  5  |  \  |
|      beta        |        DBLossloss中threshold_map_loss的系数       |  10  |  \  |

### PostProcess ([ppocr/postprocess](../../ppocr/postprocess))

|         字段             |            用途            |      默认值        |            备注             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      name        |         后处理类名          |  CTCLabelDecode  |  目前支持`CTCLoss`,`AttnLabelDecode`,`DBPostProcess`,`ClsPostProcess`  |
|      thresh        |        DBPostProcess中分割图进行二值化的阈值         |  0.3  |  \  |
|      box_thresh        |        DBPostProcess中对输出框进行过滤的阈值，低于此阈值的框不会输出         |  0.7  |  \  |
|      max_candidates        |        DBPostProcess中输出的最大文本框数量        |  1000  |   |
|      unclip_ratio        |        DBPostProcess中对文本框进行放大的比例       |  2.0  |  \  |

### Metric ([ppocr/metrics](../../ppocr/metrics))

|         字段             |            用途            |      默认值        |            备注             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      name        |         指标评估方法名称          |  CTCLabelDecode  |  目前支持`DetMetric`,`RecMetric`,`ClsMetric`  |
|      main_indicator        |        主要指标,用于选取最优模型         |  acc |  对于检测方法为hmean，识别和分类方法为acc  |

### Dataset  ([ppocr/data](../../ppocr/data))
|         字段             |            用途            |      默认值        |            备注             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      **dataset**        |         每次迭代返回一个样本          |  -  |  -  |
|      name        |        dataset类名         |  SimpleDataSet |  目前支持`SimpleDataSet`和`LMDBDataSet`  |
|      data_dir        |        数据集图片存放路径         |  ./train_data |  \  |
|      label_file_list        |        数据标签路径         |  ["./train_data/train_list.txt"] | dataset为LMDBDataSet时不需要此参数   |
|      ratio_list        |        数据集的比例         |  [1.0] | 若label_file_list中有两个train_list，且ratio_list为[0.4,0.6]，则从train_list1中采样40%，从train_list2中采样60%组合整个dataset   |
|      transforms        |        对图片和标签进行变换的方法列表         |  [DecodeImage,CTCLabelEncode,RecResizeImg,KeepKeys] |   见[ppocr/data/imaug](../../ppocr/data/imaug)  |
|      **loader**        |        dataloader相关         |  - |   |
|      shuffle        |        每个epoch是否将数据集顺序打乱         |  True | \  |
|      batch_size_per_card        |        训练时单卡batch size         |  256 | \  |
|      drop_last        |        是否丢弃因数据集样本数不能被 batch_size 整除而产生的最后一个不完整的mini-batch        |  True | \  |
|      num_workers        |        用于加载数据的子进程个数，若为0即为不开启子进程，在主进程中进行数据加载        |  8 | \  |

## 3. 多语言配置文件生成

【参考识别模型训练补充内容】
