# PaddleOCR模型推理参数解释

在使用PaddleOCR进行模型推理时，可以自定义修改参数，来修改模型、数据、预处理、后处理等内容（参数文件：[utility.py](../../tools/infer/utility.py)），详细的参数解释如下所示。

* 全局信息

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  image_dir | str | 无，必须显式指定 | 图像或者文件夹路径 |
|  vis_font_path | str | "./doc/fonts/simfang.ttf" | 用于可视化的字体路径 |
|  drop_score | float | 0.5 | 识别得分小于该值的结果会被丢弃，不会作为返回结果 |
|  use_pdserving | bool | False | 是否使用Paddle Serving进行预测 |
|  warmup | bool | False | 是否开启warmup，在统计预测耗时的时候，可以使用这种方法 |
|  draw_img_save_dir | str | "./inference_results" | 系统串联预测OCR结果的保存文件夹 |
|  save_crop_res | bool | False  | 是否保存OCR的识别文本图像 |
|  crop_res_save_dir | str | "./output" | 保存OCR识别出来的文本图像路径 |
|  use_mp | bool | False | 是否开启多进程预测  |
|  total_process_num | int | 6 | 开启的进城数，`use_mp`为`True`时生效  |
|  process_id | int | 0 | 当前进程的id号，无需自己修改  |
|  benchmark | bool | False | 是否开启benchmark，对预测速度、显存占用等进行统计  |
|  save_log_path | str | "./log_output/" | 开启`benchmark`时，日志结果的保存文件夹 |
|  show_log | bool | True | 是否显示预测中的日志信息  |
|  use_onnx | bool | False | 是否开启onnx预测 |


* 预测引擎相关

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  use_gpu | bool | True | 是否使用GPU进行预测 |
|  ir_optim | bool | True | 是否对计算图进行分析与优化，开启后可以加速预测过程 |
|  use_tensorrt | bool | False | 是否开启tensorrt |
|  min_subgraph_size | int | 15 | tensorrt中最小子图size，当子图的size大于该值时，才会尝试对该子图使用trt engine计算 |
|  precision | str | fp32 | 预测的精度，支持`fp32`, `fp16`, `int8` 3种输入 |
|  enable_mkldnn | bool | True | 是否开启mkldnn |
|  cpu_threads | int | 10 | 开启mkldnn时，cpu预测的线程数 |

* 文本检测模型相关

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  det_algorithm | str | "DB" | 文本检测算法名称，目前支持`DB`, `EAST`, `SAST`, `PSE`  |
|  det_model_dir | str | xx | 检测inference模型路径 |
|  det_limit_side_len | int | 960 | 检测的图像边长限制 |
|  det_limit_type | str | "max" | 检测的变成限制类型，目前支持`min`, `max`，`min`表示保证图像最短边不小于`det_limit_side_len`，`max`表示保证图像最长边不大于`det_limit_side_len` |

其中，DB算法相关参数如下

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  det_db_thresh | float | 0.3 | DB输出的概率图中，得分大于该阈值的像素点才会被认为是文字像素点 |
|  det_db_box_thresh | float | 0.6 | 检测结果边框内，所有像素点的平均得分大于该阈值时，该结果会被认为是文字区域 |
|  det_db_unclip_ratio | float | 1.5 | `Vatti clipping`算法的扩张系数，使用该方法对文字区域进行扩张 |
|  max_batch_size | int | 10 | 预测的batch size |
|  use_dilation | bool | False | 是否对分割结果进行膨胀以获取更优检测效果 |
|  det_db_score_mode | str | "fast" | DB的检测结果得分计算方法，支持`fast`和`slow`，`fast`是根据polygon的外接矩形边框内的所有像素计算平均得分，`slow`是根据原始polygon内的所有像素计算平均得分，计算速度相对较慢一些，但是更加准确一些。 |

EAST算法相关参数如下

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  det_east_score_thresh | float | 0.8 | EAST后处理中score map的阈值 |
|  det_east_cover_thresh | float | 0.1 | EAST后处理中文本框的平均得分阈值 |
|  det_east_nms_thresh | float | 0.2 | EAST后处理中nms的阈值 |

SAST算法相关参数如下

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  det_sast_score_thresh | float | 0.5 | SAST后处理中的得分阈值 |
|  det_sast_nms_thresh | float | 0.5 | SAST后处理中nms的阈值 |
|  det_sast_polygon | bool | False | 是否多边形检测，弯曲文本场景（如Total-Text）设置为True |

PSE算法相关参数如下

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  det_pse_thresh | float | 0.0 | 对输出图做二值化的阈值 |
|  det_pse_box_thresh | float | 0.85 | 对box进行过滤的阈值，低于此阈值的丢弃 |
|  det_pse_min_area | float | 16 | box的最小面积，低于此阈值的丢弃 |
|  det_pse_box_type | str | "box" | 返回框的类型，box:四点坐标，poly: 弯曲文本的所有点坐标 |
|  det_pse_scale | int | 1 | 输入图像相对于进后处理的图的比例，如`640*640`的图像，网络输出为`160*160`，scale为2的情况下，进后处理的图片shape为`320*320`。这个值调大可以加快后处理速度，但是会带来精度的下降 |

* 文本识别模型相关

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  rec_algorithm | str | "CRNN" | 文本识别算法名称，目前支持`CRNN`, `SRN`, `RARE`, `NETR`, `SAR` |
|  rec_model_dir | str | 无，如果使用识别模型，该项是必填项 | 识别inference模型路径 |
|  rec_image_shape | list | [3, 32, 320] | 识别时的图像尺寸， |
|  rec_batch_num | int | 6 | 识别的batch size |
|  max_text_length | int | 25 | 识别结果最大长度，在`SRN`中有效 |
|  rec_char_dict_path | str | "./ppocr/utils/ppocr_keys_v1.txt" | 识别的字符字典文件 |
|  use_space_char | bool | True | 是否包含空格，如果为`True`，则会在最后字符字典中补充`空格`字符 |


* 端到端文本检测与识别模型相关

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  e2e_algorithm | str | "PGNet" | 端到端算法名称，目前支持`PGNet` |
|  e2e_model_dir | str | 无，如果使用端到端模型，该项是必填项 | 端到端模型inference模型路径 |
|  e2e_limit_side_len | int | 768 | 端到端的输入图像边长限制 |
|  e2e_limit_type | str | "max" | 端到端的边长限制类型，目前支持`min`, `max`，`min`表示保证图像最短边不小于`e2e_limit_side_len`，`max`表示保证图像最长边不大于`e2e_limit_side_len` |
|  e2e_pgnet_score_thresh | float | 0.5 | 端到端得分阈值，小于该阈值的结果会被丢弃 |
|  e2e_char_dict_path | str | "./ppocr/utils/ic15_dict.txt" | 识别的字典文件路径 |
|  e2e_pgnet_valid_set | str | "totaltext" | 验证集名称，目前支持`totaltext`, `partvgg`，不同数据集对应的后处理方式不同，与训练过程保持一致即可 |
|  e2e_pgnet_mode | str | "fast" | PGNet的检测结果得分计算方法，支持`fast`和`slow`，`fast`是根据polygon的外接矩形边框内的所有像素计算平均得分，`slow`是根据原始polygon内的所有像素计算平均得分，计算速度相对较慢一些，但是更加准确一些。 |


* 方向分类器模型相关

| 参数名称 | 类型 | 默认值 | 含义 |
| :--: | :--: | :--: | :--: |
|  use_angle_cls | bool | False | 是否使用方向分类器 |
|  cls_model_dir | str | 无，如果需要使用，则必须显式指定路径 | 方向分类器inference模型路径 |
|  cls_image_shape | list | [3, 48, 192] | 预测尺度 |
|  label_list | list | ['0', '180'] | class id对应的角度值 |
|  cls_batch_num | int | 6 | 方向分类器预测的batch size |
|  cls_thresh | float | 0.9 | 预测阈值，模型预测结果为180度，且得分大于该阈值时，认为最终预测结果为180度，需要翻转 |
