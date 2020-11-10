# 可选参数列表

以下列表可以通过`--help`查看

|         FLAG             |     支持脚本    |        用途        |      默认值       |         备注         |
| :----------------------: | :------------: | :---------------: | :--------------: | :-----------------: |
|          -c              |      ALL       |  指定配置文件  |  None  |  **配置模块说明请参考 参数介绍** |
|          -o              |      ALL       |  设置配置文件里的参数内容  |  None  |  使用-o配置相较于-c选择的配置文件具有更高的优先级。例如：`-o Global.use_gpu=false`  |  


## 配置文件 Global 参数介绍

以 `rec_chinese_lite_train_v1.1.yml ` 为例


|         字段             |            用途                |      默认值       |            备注            |
| :----------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      algorithm           |    设置算法                    |  与配置文件同步   |     选择模型，支持模型请参考[简介](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/README.md) |
|      use_gpu             |    设置代码运行场所            |       true        |                \                 |
|      epoch_num           |    最大训练epoch数             |       3000        |                \                 |
|      log_smooth_window   |    滑动窗口大小            |       20          |                \                 |
|      print_batch_step    |    设置打印log间隔         |       10          |                \                 |
|      save_model_dir      |    设置模型保存路径        |  output/{算法名称}  |                \                 |
|      save_epoch_step     |    设置模型保存间隔        |       3           |                \                 |
|      eval_batch_step     |    设置模型评估间隔        | 2000 或 [1000, 2000]        | 2000 表示每2000次迭代评估一次，[1000， 2000]表示从1000次迭代开始，每2000次评估一次   |
|train_batch_size_per_card |  设置训练时单卡batch size    |         256         |                \                 |
| test_batch_size_per_card |  设置评估时单卡batch size    |         256         |                \                 |
|      image_shape         |    设置输入图片尺寸        |   [3, 32, 100]    |                \                 |
|      max_text_length     |    设置文本最大长度        |       25          |                \                 |
|      character_type      |    设置字符类型            |       ch          |    en/ch, en时将使用默认dict，ch时使用自定义dict|
|      character_dict_path |    设置字典路径            |  ./ppocr/utils/ic15_dict.txt  |    \                 |
|      loss_type           |    设置 loss 类型              |       ctc         |    支持两种loss： ctc / attention |
|       distort            |    设置是否使用数据增强          |       false       |  设置为true时，将在训练时随机进行扰动，支持的扰动操作可阅读[img_tools.py](https://github.com/PaddlePaddle/PaddleOCR/blob/develop/ppocr/data/rec/img_tools.py)                 |
|       use_space_char     |    设置是否识别空格             |        false      |          仅在 character_type=ch 时支持空格                 |
|      label_list          |    设置方向分类器支持的角度       |    ['0','180']    |     仅在方向分类器中生效 |
|      average_window      |    ModelAverage优化器中的窗口长度计算比例 |  0.15       |       目前仅应用与SRN |
|      max_average_window  |    平均值计算窗口长度的最大值   |   15625              | 推荐设置为一轮训练中mini-batchs的数目|
|      min_average_window  |    平均值计算窗口长度的最小值  |    10000              |      \          |
|      reader_yml          |    设置reader配置文件          |  ./configs/rec/rec_icdar15_reader.yml  |  \          |
|      pretrain_weights    |    加载预训练模型路径      |  ./pretrain_models/CRNN/best_accuracy  |  \          |
|      checkpoints         |    加载模型参数路径            |       None        |    用于中断后加载参数继续训练 |
|      save_inference_dir  |    inference model 保存路径 |          None        |    用于保存inference model |

## 配置文件 Reader 系列参数介绍

以 `rec_chinese_reader.yml` 为例

|         字段             |            用途                |      默认值       |            备注            |
| :----------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|      reader_function     |    选择数据读取方式        |  ppocr.data.rec.dataset_traversal,SimpleReader  | 支持SimpleReader / LMDBReader 两种数据读取方式 |
|      num_workers             |    设置数据读取线程数            |       8        |                \                 |
|      img_set_dir          |    数据集路径             |       ./train_data        |                \                 |
|      label_file_path      |    数据标签路径           |       ./train_data/rec_gt_train.txt| \    |
|      infer_img            |    预测图像文件夹路径     |       ./infer_img | \|

## 配置文件 Optimizer 系列参数介绍

以 `rec_icdar15_train.yml` 为例

|         字段             |            用途            |      默认值        |            备注             |
| :---------------------: |  :---------------------:   | :--------------:  |   :--------------------:   |
|         function        |         选择优化器          |  pocr.optimizer,AdamDecay  |  目前只支持Adam方式  |
|         base_lr         |      设置初始学习率          |       0.0005      |               \             |
|         beta1           |    设置一阶矩估计的指数衰减率  |       0.9         |               \             |
|         beta2           |    设置二阶矩估计的指数衰减率  |     0.999         |               \             |
|         decay           |         是否使用decay       |    \              |               \             |
|      function(decay)    |         设置decay方式       |   -    |       目前支持cosine_decay, cosine_decay_warmup与piecewise_decay  |
|      step_each_epoch    |      每个epoch包含多少次迭代, cosine_decay/cosine_decay_warmup时有效   |         20       | 计算方式：total_image_num / (batch_size_per_card * card_size) |
|        total_epoch      |    总共迭代多少个epoch, cosine_decay/cosine_decay_warmup时有效        |       1000      | 与Global.epoch_num 一致        |
|        warmup_minibatch      |  线性warmup的迭代次数, cosine_decay_warmup时有效        |       1000      | \        |
|        boundaries      |    学习率下降时的迭代次数间隔, piecewise_decay时有效       |       -      | 参数为列表形式        |
|        decay_rate      |    学习率衰减系数, piecewise_decay时有效       |       -      |  \        |
