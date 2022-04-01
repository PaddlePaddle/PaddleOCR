# Document Visual Q&A（DOC-VQA）

Document Visual Q&A, mainly for the image content of the question and answer, DOC-VQA is a type of VQA task, DOC-VQA mainly asks questions about the textual content of text images.

The DOC-VQA algorithm in PP-Structure is developed based on PaddleNLP natural language processing algorithm library.

The main features are as follows:

- Integrated LayoutXLM model and PP-OCR prediction engine.
- Support Semantic Entity Recognition (SER) and Relation Extraction (RE) tasks based on multi-modal methods. Based on SER task, text recognition and classification in images can be completed. Based on THE RE task, we can extract the relation of the text content in the image, such as judge the problem pair.

- Support custom training for SER and RE tasks.

- Support OCR+SER end-to-end system prediction and evaluation.

- Support OCR+SER+RE end-to-end system prediction.

**Note**: This project is based on the open source implementation of  [LayoutXLM](https://arxiv.org/pdf/2104.08836.pdf) on Paddle 2.2, and at the same time, after in-depth polishing by the flying Paddle team and the Industrial and **Commercial Bank of China** in the scene of real estate certificate, jointly open source.


## 1.Performance

We evaluated the algorithm on  [XFUN](https://github.com/doc-analysis/XFUND) 's Chinese data set, and the performance is as follows

| Model | Task | F1 | Model Download Link |
|:---:|:---:|:---:| :---:|
| LayoutXLM | RE | 0.7113 | [Link](https://paddleocr.bj.bcebos.com/pplayout/PP-Layout_v1.0_re_pretrained.tar) |
| LayoutXLM | SER | 0.9056 | [Link](https://paddleocr.bj.bcebos.com/pplayout/PP-Layout_v1.0_ser_pretrained.tar) |
| LayoutLM | SER | 0.78 | [Link](https://paddleocr.bj.bcebos.com/pplayout/LayoutLM_ser_pretrained.tar) |



## 2.Demonstration

**Note**: the test images are from the xfun dataset.

### 2.1 SER

![](./images/result_ser/zh_val_0_ser.jpg) | ![](./images/result_ser/zh_val_42_ser.jpg)
---|---

Different colored boxes in the figure represent different categories. For xfun dataset, there are three categories: query, answer and header:

* Dark purple: header
* Light purple: query
* Army green: answer

The corresponding category and OCR recognition results are also marked at the top left of the OCR detection box.


### 2.2 RE

![](./images/result_re/zh_val_21_re.jpg) | ![](./images/result_re/zh_val_40_re.jpg)
---|---


In the figure, the red box represents the question, the blue box represents the answer, and the question and answer are connected by green lines. The corresponding category and OCR recognition results are also marked at the top left of the OCR detection box.


## 3. Setup

### 3.1 Installation dependency

- **（1) Install PaddlePaddle**

```bash
pip3 install --upgrade pip

# GPU PaddlePaddle Install
python3 -m pip install paddlepaddle-gpu==2.2 -i https://mirror.baidu.com/pypi/simple

# CPU PaddlePaddle Install
python3 -m pip install paddlepaddle==2.2 -i https://mirror.baidu.com/pypi/simple

```
For more requirements, please refer to the [instructions](https://www.paddlepaddle.org.cn/install/quick) in the installation document.


### 3.2 Install PaddleOCR (including pp-ocr and VQA)

- **(1) PIP quick install paddleocr WHL package (forecast only)**

```bash
pip install paddleocr
```

- **(2) Download VQA source code (prediction + training)**

```bash
[recommended] git clone https://github.com/PaddlePaddle/PaddleOCR

# If you cannot pull successfully because of network problems, you can also choose to use the hosting on the code cloud:
git clone https://gitee.com/paddlepaddle/PaddleOCR

# Note: the code cloud hosting code may not be able to synchronize the update of this GitHub project in real time, with a delay of 3 ~ 5 days. Please give priority to the recommended method.
```

- **(3) Install PaddleNLP**

```bash
# You need to use the latest code version of paddlenlp for installation
git clone https://github.com/PaddlePaddle/PaddleNLP -b develop
cd PaddleNLP
pip3 install -e .
```


- **(4) Install requirements for VQA**

```bash
cd ppstructure/vqa
pip install -r requirements.txt
```

## 4.Usage


### 4.1 Data and pre training model preparation

Download address of processed xfun Chinese dataset: [https://paddleocr.bj.bcebos.com/dataset/XFUND.tar](https://paddleocr.bj.bcebos.com/dataset/XFUND.tar)。


Download and unzip the dataset, and then place the dataset in the current directory.

```shell
wget https://paddleocr.bj.bcebos.com/dataset/XFUND.tar
```

If you want to convert data sets in other languages in xfun, you can refer to [xfun data conversion script.](helper/trans_xfun_data.py))

If you want to experience the prediction process directly, you can download the pre training model provided by us, skip the training process and predict directly.


### 4.2 SER Task

* Start training

```shell
python3.7 train_ser.py \
    --model_name_or_path "layoutxlm-base-uncased" \
    --ser_model_type "LayoutXLM" \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --num_train_epochs 200 \
    --eval_steps 10 \
    --output_dir "./output/ser/" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --evaluate_during_training \
    --seed 2048
```

Finally, Precision, Recall, F1 and other indicators will be printed, and the model and training log will be saved in/ In the output/Ser/ folder.

* Recovery training

```shell
python3.7 train_ser.py \
    --model_name_or_path "model_path" \
    --ser_model_type "LayoutXLM" \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --num_train_epochs 200 \
    --eval_steps 10 \
    --output_dir "./output/ser/" \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --evaluate_during_training \
    --num_workers 8 \
    --seed 2048 \
    --resume
```

* Evaluation
```shell
export CUDA_VISIBLE_DEVICES=0
python3 eval_ser.py \
    --model_name_or_path "PP-Layout_v1.0_ser_pretrained/" \
    --ser_model_type "LayoutXLM" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --per_gpu_eval_batch_size 8 \
    --num_workers 8 \
    --output_dir "output/ser/"  \
    --seed 2048
```
Finally, Precision, Recall, F1 and other indicators will be printed

* The OCR recognition results provided in the evaluation set are used for prediction

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 infer_ser.py \
    --model_name_or_path "PP-Layout_v1.0_ser_pretrained/" \
    --ser_model_type "LayoutXLM" \
    --output_dir "output/ser/" \
    --infer_imgs "XFUND/zh_val/image/" \
    --ocr_json_path "XFUND/zh_val/xfun_normalize_val.json"
```

It will end up in output_res The visual image of the prediction result and the text file of the prediction result are saved in the res directory. The file name is infer_ results.txt.

* Using OCR engine + SER concatenation results

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 infer_ser_e2e.py \
    --model_name_or_path "PP-Layout_v1.0_ser_pretrained/" \
    --ser_model_type "LayoutXLM" \
    --max_seq_length 512 \
    --output_dir "output/ser_e2e/" \
    --infer_imgs "images/input/zh_val_0.jpg"
```

* End-to-end evaluation of OCR engine + SER prediction system

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 helper/eval_with_label_end2end.py --gt_json_path XFUND/zh_val/xfun_normalize_val.json  --pred_json_path output_res/infer_results.txt
```


### 4.3 RE Task

* Start training

```shell
export CUDA_VISIBLE_DEVICES=0
python3 train_re.py \
    --model_name_or_path "layoutxlm-base-uncased" \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --label_map_path "labels/labels_ser.txt" \
    --num_train_epochs 200 \
    --eval_steps 10 \
    --output_dir "output/re/"  \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --num_workers 8 \
    --evaluate_during_training \
    --seed 2048

```

* Resume training

```shell
export CUDA_VISIBLE_DEVICES=0
python3 train_re.py \
    --model_name_or_path "model_path" \
    --train_data_dir "XFUND/zh_train/image" \
    --train_label_path "XFUND/zh_train/xfun_normalize_train.json" \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --label_map_path "labels/labels_ser.txt" \
    --num_train_epochs 2 \
    --eval_steps 10 \
    --output_dir "output/re/"  \
    --learning_rate 5e-5 \
    --warmup_steps 50 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 8 \
    --num_workers 8 \
    --evaluate_during_training \
    --seed 2048 \
    --resume

```

Finally, Precision, Recall, F1 and other indicators will be printed, and the model and training log will be saved in the output/RE file folder.

* Evaluation
```shell
export CUDA_VISIBLE_DEVICES=0
python3 eval_re.py \
    --model_name_or_path "PP-Layout_v1.0_re_pretrained/" \
    --max_seq_length 512 \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --label_map_path "labels/labels_ser.txt" \
    --output_dir "output/re/"  \
    --per_gpu_eval_batch_size 8 \
    --num_workers 8 \
    --seed 2048
```
Finally, Precision, Recall, F1 and other indicators will be printed


* The OCR recognition results provided in the evaluation set are used for prediction

```shell
export CUDA_VISIBLE_DEVICES=0
python3 infer_re.py \
    --model_name_or_path "PP-Layout_v1.0_re_pretrained/" \
    --max_seq_length 512 \
    --eval_data_dir "XFUND/zh_val/image" \
    --eval_label_path "XFUND/zh_val/xfun_normalize_val.json" \
    --label_map_path "labels/labels_ser.txt" \
    --output_dir "output/re/"  \
    --per_gpu_eval_batch_size 1 \
    --seed 2048
```

The visual image of the prediction result and the text file of the prediction result are saved in the output_res file folder, the file name is`infer_results.txt`。

* Concatenation results using OCR engine + SER+ RE

```shell
export CUDA_VISIBLE_DEVICES=0
python3.7 infer_ser_re_e2e.py \
    --model_name_or_path "PP-Layout_v1.0_ser_pretrained/" \
    --re_model_name_or_path "PP-Layout_v1.0_re_pretrained/" \
    --ser_model_type "LayoutXLM" \
    --max_seq_length 512 \
    --output_dir "output/ser_re_e2e/" \
    --infer_imgs "images/input/zh_val_21.jpg"
```

## Reference

- LayoutXLM: Multimodal Pre-training for Multilingual Visually-rich Document Understanding, https://arxiv.org/pdf/2104.08836.pdf
- microsoft/unilm/layoutxlm, https://github.com/microsoft/unilm/tree/master/layoutxlm
- XFUND dataset, https://github.com/doc-analysis/XFUND

## License

The content of this project itself is licensed under the [Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
