- [Table Recognition](#table-recognition)
  - [1. pipeline](#1-pipeline)
  - [2. Performance](#2-performance)
  - [3. How to use](#3-how-to-use)
    - [3.1 quick start](#31-quick-start)
    - [3.2 Train](#32-train)
    - [3.3 Eval](#33-eval)
    - [3.4 Inference](#34-inference)


# Table Recognition

## 1. pipeline
The table recognition mainly contains three models
1. Single line text detection-DB
2. Single line text recognition-CRNN
3. Table structure and cell coordinate prediction-RARE

The table recognition flow chart is as follows

![tableocr_pipeline](../docs/table/tableocr_pipeline_en.jpg)

1. The coordinates of single-line text is detected by DB model, and then sends it to the recognition model to get the recognition result.
2. The table structure and cell coordinates is predicted by RARE model.
3. The recognition result of the cell is combined by the coordinates, recognition result of the single line and the coordinates of the cell.
4. The cell recognition result and the table structure together construct the html string of the table.

## 2. Performance
We evaluated the algorithm on the PubTabNet<sup>[1]</sup> eval dataset, and the performance is as follows:


|Method|[TEDS(Tree-Edit-Distance-based Similarity)](https://github.com/ibm-aur-nlp/PubTabNet/tree/master/src)|
| --- | --- |
| EDD<sup>[2]</sup> | 88.3 |
| TableRec-RARE(ours) | 93.32 |
| SLANet(ours) | 94.98 |

## 3. How to use

### 3.1 quick start

```python
cd PaddleOCR/ppstructure

# download model
mkdir inference && cd inference
# Download the detection model of the ultra-lightweight table English OCR model and unzip it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_det_infer.tar && tar xf en_ppocr_mobile_v2.0_table_det_infer.tar
# Download the recognition model of the ultra-lightweight table English OCR model and unzip it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_rec_infer.tar && tar xf en_ppocr_mobile_v2.0_table_rec_infer.tar
# Download the ultra-lightweight English table inch model and unzip it
wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/table/en_ppocr_mobile_v2.0_table_structure_infer.tar && tar xf en_ppocr_mobile_v2.0_table_structure_infer.tar
cd ..
# run
python3 table/predict_table.py --det_model_dir=inference/en_ppocr_mobile_v2.0_table_det_infer --rec_model_dir=inference/en_ppocr_mobile_v2.0_table_rec_infer --table_model_dir=inference/en_ppocr_mobile_v2.0_table_structure_infer --image_dir=./docs/table/table.jpg --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --det_limit_side_len=736 --det_limit_type=min --output ./output/table
```
Note: The above model is trained on the PubLayNet dataset and only supports English scanning scenarios. If you need to identify other scenarios, you need to train the model yourself and replace the three fields `det_model_dir`, `rec_model_dir`, `table_model_dir`.

After the operation is completed, the excel table of each image will be saved to the directory specified by the output field, and an html file will be produced in the directory to visually view the cell coordinates and the recognized table.

### 3.2 Train

In this chapter, we only introduce the training of the table structure model, For model training of [text detection](../../doc/doc_en/detection_en.md) and [text recognition](../../doc/doc_en/recognition_en.md), please refer to the corresponding documents

* data preparation  
The training data uses public data set [PubTabNet](https://arxiv.org/abs/1911.10683 ), Can be downloaded from the official [website](https://github.com/ibm-aur-nlp/PubTabNet) 。The PubTabNet data set contains about 500,000 images, as well as annotations in html format。

* Start training  
*If you are installing the cpu version of paddle, please modify the `use_gpu` field in the configuration file to false*
```shell
# single GPU training
python3 tools/train.py -c configs/table/table_mv3.yml
# multi-GPU training
# Set the GPU ID used by the '--gpus' parameter.
python3 -m paddle.distributed.launch --gpus '0,1,2,3' tools/train.py -c configs/table/table_mv3.yml
```

In the above instruction, use `-c` to select the training to use the `configs/table/table_mv3.yml` configuration file.
For a detailed explanation of the configuration file, please refer to [config](../../doc/doc_en/config_en.md).

* load trained model and continue training

If you expect to load trained model and continue the training again, you can specify the parameter `Global.checkpoints` as the model path to be loaded.

```shell
python3 tools/train.py -c configs/table/table_mv3.yml -o Global.checkpoints=./your/trained/model
```

**Note**: The priority of `Global.checkpoints` is higher than that of `Global.pretrain_weights`, that is, when two parameters are specified at the same time, the model specified by `Global.checkpoints` will be loaded first. If the model path specified by `Global.checkpoints` is wrong, the one specified by `Global.pretrain_weights` will be loaded.

### 3.3 Eval

The table uses [TEDS(Tree-Edit-Distance-based Similarity)](https://github.com/ibm-aur-nlp/PubTabNet/tree/master/src) as the evaluation metric of the model. Before the model evaluation, the three models in the pipeline need to be exported as inference models (we have provided them), and the gt for evaluation needs to be prepared. Examples of gt are as follows:
```txt
PMC5755158_010_01.png    <html><body><table><thead><tr><td></td><td><b>Weaning</b></td><td><b>Week 15</b></td><td><b>Off-test</b></td></tr></thead><tbody><tr><td>Weaning</td><td>–</td><td>–</td><td>–</td></tr><tr><td>Week 15</td><td>–</td><td>0.17 ± 0.08</td><td>0.16 ± 0.03</td></tr><tr><td>Off-test</td><td>–</td><td>0.80 ± 0.24</td><td>0.19 ± 0.09</td></tr></tbody></table></body></html>
```
Each line in gt consists of the file name and the html string of the table. The file name and the html string of the table are separated by `\t`.

You can also use the following command to generate an evaluation gt file from the annotation file:
```python
python3 ppstructure/table/convert_label2html.py --ori_gt_path /path/to/your_label_file --save_path /path/to/save_file
```

Use the following command to evaluate. After the evaluation is completed, the teds indicator will be output.
```python
cd PaddleOCR/ppstructure
python3 table/eval_table.py --det_model_dir=path/to/det_model_dir --rec_model_dir=path/to/rec_model_dir --table_model_dir=path/to/table_model_dir --image_dir=../doc/table/1.png --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --det_limit_side_len=736 --det_limit_type=min --gt_path=path/to/gt.txt
```

If the PubLatNet eval dataset is used, it will be output
```bash
teds: 94.98
```

### 3.4 Inference

```python
cd PaddleOCR/ppstructure
python3 table/predict_table.py --det_model_dir=path/to/det_model_dir --rec_model_dir=path/to/rec_model_dir --table_model_dir=path/to/table_model_dir --image_dir=../doc/table/1.png --rec_char_dict_path=../ppocr/utils/dict/table_dict.txt --table_char_dict_path=../ppocr/utils/dict/table_structure_dict.txt --det_limit_side_len=736 --det_limit_type=min --output ../output/table
```
After running, the excel sheet of each picture will be saved in the directory specified by the output field

Reference
1. https://github.com/ibm-aur-nlp/PubTabNet
2. https://arxiv.org/pdf/1911.10683
