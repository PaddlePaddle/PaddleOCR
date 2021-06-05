# 表格结构和内容预测

先cd到PaddleOCR/ppstructure目录下

预测
```python
python3 table/predict_table.py --det_model_dir=../inference/db --rec_model_dir=../inference/rec_mv3_large1.0/infer --table_model_dir=../inference/explite3/infer --image_dir=../table/imgs/PMC3006023_004_00.png --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --rec_char_type=EN --det_limit_side_len=736 --det_limit_type=min --table_output ../output/table
```
运行完成后，每张图片的excel表格会保存到table_output字段指定的目录下

eval

```python
python3 table/eval_table.py --det_model_dir=../inference/db --rec_model_dir=../inference/rec_mv3_large1.0/infer --table_model_dir=../inference/explite3/infer --image_dir=../table/imgs --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --rec_char_type=EN --det_limit_side_len=736 --det_limit_type=min --gt_path=path/to/gt.json
```
