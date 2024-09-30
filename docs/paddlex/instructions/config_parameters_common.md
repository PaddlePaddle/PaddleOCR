# PaddleX通用模型配置文件参数说明

# Global
|参数名|数据类型|描述|默认值|必需/可选|
|-|-|-|-|-|
|model|str|指定模型名称|-|必需|
|mode|str|指定模式（check_dataset/train/evaluate/export/predict）|-|必需|
|dataset_dir|str|数据集路径|-|必需|
|device|str|指定使用的设备|-|必需|
|output|str|输出路径|"output"|可选|
# CheckDataset
|参数名|数据类型|描述|默认值|必需/可选|
|-|-|-|-|-|
|convert.enable|bool|是否进行数据集格式转换|False|可选|
|convert.src_dataset_type|str|需要转换的源数据集格式|null|可选|
|split.enable|bool|是否重新划分数据集|False|可选|
|split.train_percent|int|设置训练集的百分比，类型为0-100之间的任意整数，需要保证和val_percent值加和为100；|null|可选|
|split.val_percent|int|设置验证集的百分比，类型为0-100之间的任意整数，需要保证和train_percent值加和为100；|null|可选|
|split.gallery_percent|int|设置验证集中被查询样本的百分比，类型为0-100之间的任意整数，需要保证和train_percent、query_percent，值加和为100；该参数只有图像特征模块才会使用|null|可选|
|split.query_percent|int|设置验证集中查询样本的百分比，类型为0-100之间的任意整数，需要保证和train_percent、gallery_percent，值加和为100；该参数只有图像特征模块才会使用|null|可选|

# Train
|参数名|数据类型|描述|默认值|必需/可选|
|-|-|-|-|-|
|num_classes|int|数据集中的类别数|-|必需|
|epochs_iters|int|模型对训练数据的重复学习次数|-|必需|
|batch_size|int|训练批大小|-|必需|
|learning_rate|float|初始学习率|-|必需|
|pretrain_weight_path|str|预训练权重路径|null|可选|
|warmup_steps|int|预热步数|-|必需|
|resume_path|str|模型中断后的恢复路径|null|可选|
|log_interval|int|训练日志打印间隔|-|必需|
|eval_interval|int|模型评估间隔|-|必需|
|save_interval|int|模型保存间隔|-|必需|

# Evaluate
|参数名|数据类型|描述|默认值|必需/可选|
|-|-|-|-|-|
|weight_path|str|评估模型路径|-|必需|
|log_interval|int|评估日志打印间隔|-|必需|
# Export
|参数名|数据类型|描述|默认值|必需/可选|
|-|-|-|-|-|
|weight_path|str|导出模型的动态图权重路径|各模型官方动态图权重URL|必需|
# Predict
|参数名|数据类型|描述|默认值|必需/可选|
|-|-|-|-|-|
|batch_size|int|预测批大小|-|必需|
|model_dir|str|预测模型路径|PaddleX模型官方权重|可选|
|input|str|预测输入路径|-|必需|

