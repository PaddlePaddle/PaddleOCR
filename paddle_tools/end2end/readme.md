
# 简介

`tools/end2end`目录下存放了文本检测+文本识别pipeline串联预测的指标评测代码以及可视化工具。本节介绍文本检测+文本识别的端对端指标评估方式。


## 端对端评测步骤

**步骤一：**

运行`tools/infer/predict_system.py`，得到保存的结果:

```
python3 tools/infer/predict_system.py  --det_model_dir=./ch_PP-OCRv2_det_infer/ --rec_model_dir=./ch_PP-OCRv2_rec_infer/  --image_dir=./datasets/img_dir/ --draw_img_save_dir=./ch_PP-OCRv2_results/ --is_visualize=True
```

文本检测识别可视化图默认保存在`./ch_PP-OCRv2_results/`目录下，预测结果默认保存在`./ch_PP-OCRv2_results/system_results.txt`中，格式如下：
```
all-sum-510/00224225.jpg        [{"transcription": "超赞", "points": [[8.0, 48.0], [157.0, 44.0], [159.0, 115.0], [10.0, 119.0]], "score": "0.99396634"}, {"transcription": "中", "points": [[202.0, 152.0], [230.0, 152.0], [230.0, 163.0], [202.0, 163.0]], "score": "0.09310734"}, {"transcription": "58.0m", "points": [[196.0, 192.0], [444.0, 192.0], [444.0, 240.0], [196.0, 240.0]], "score": "0.44041982"}, {"transcription": "汽配", "points": [[55.0, 263.0], [95.0, 263.0], [95.0, 281.0], [55.0, 281.0]], "score": "0.9986651"}, {"transcription": "成总店", "points": [[120.0, 262.0], [176.0, 262.0], [176.0, 283.0], [120.0, 283.0]], "score": "0.9929402"}, {"transcription": "K", "points": [[237.0, 286.0], [311.0, 286.0], [311.0, 345.0], [237.0, 345.0]], "score": "0.6074794"}, {"transcription": "88：-8", "points": [[203.0, 405.0], [477.0, 414.0], [475.0, 459.0], [201.0, 450.0]], "score": "0.7106863"}]
```


**步骤二：**

将步骤一保存的数据转换为端对端评测需要的数据格式：

修改 `tools/end2end/convert_ppocr_label.py`中的代码，convert_label函数中设置输入标签路径，Mode，保存标签路径等，对预测数据的GTlabel和预测结果的label格式进行转换。

```
python3 tools/end2end/convert_ppocr_label.py --mode=gt --label_path=path/to/label_txt --save_folder=save_gt_label

python3 tools/end2end/convert_ppocr_label.py --mode=pred --label_path=path/to/pred_txt --save_folder=save_PPOCRV2_infer
```

得到如下结果：
```
├── ./save_gt_label/
├── ./save_PPOCRV2_infer/
```

**步骤三：**

执行端对端评测，运行`tools/eval_end2end.py`计算端对端指标，运行方式如下：

```
python3 tools/eval_end2end.py "gt_label_dir"  "predict_label_dir"
```

比如：

```
python3 tools/eval_end2end.py ./save_gt_label/ ./save_PPOCRV2_infer/
```
将得到如下结果，fmeasure为主要关注的指标：
```
hit, dt_count, gt_count 1557 2693 3283
character_acc: 61.77%
avg_edit_dist_field: 3.08
avg_edit_dist_img: 51.82
precision: 57.82%
recall: 47.43%
fmeasure: 52.11%
```
