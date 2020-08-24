# paddleocr package使用说明

## 快速上手

### 安装whl包

pip安装
```bash
pip install paddleocr
```

本地构建并安装
```bash
python setup.py bdist_wheel
pip install dist/paddleocr-0.0.3-py3-none-any.whl
```
### 1. 代码使用

* 检测+识别全流程
```python
from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR() # need to run only once to download and load model into memory
img_path = 'PaddleOCR/doc/imgs/11.jpg'
result = ocr.ocr(img_path)
for line in result:
    print(line)

# 显示结果
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```
结果是一个list，每个item包含了文本框，文字和识别置信度
```bash
[[[24.0, 36.0], [304.0, 34.0], [304.0, 72.0], [24.0, 74.0]], ['纯臻营养护发素', 0.964739]]
[[[24.0, 80.0], [172.0, 80.0], [172.0, 104.0], [24.0, 104.0]], ['产品信息/参数', 0.98069626]]
[[[24.0, 109.0], [333.0, 109.0], [333.0, 136.0], [24.0, 136.0]], ['（45元/每公斤，100公斤起订）', 0.9676722]]
......
```
结果可视化

<div align="center">
    <img src="../imgs_results/whl/11_det_rec.jpg" width="800">
</div>

* 单独执行检测
```python
from paddleocr import PaddleOCR, draw_ocr
ocr = PaddleOCR() # need to run only once to download and load model into memory
img_path = 'PaddleOCR/doc/imgs/11.jpg'
result = ocr.ocr(img_path,rec=False)
for line in result:
    print(line)

# 显示结果
from PIL import Image

image = Image.open(img_path).convert('RGB')
im_show = draw_ocr(image, result, txts=None, scores=None, font_path='/path/to/PaddleOCR/doc/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```
结果是一个list，每个item只包含文本框
```bash
[[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]]
[[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]]
[[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]]
......
```
结果可视化


<div align="center">
    <img src="../imgs_results/whl/11_det.jpg" width="800">
</div>

* 单独执行识别
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR() # need to run only once to download and load model into memory
img_path = 'PaddleOCR/doc/imgs_words/ch/word_1.jpg'
result = ocr.ocr(img_path,det=False)
for line in result:
    print(line)
```
结果是一个list，每个item只包含识别结果和识别置信度
```bash
['韩国小馆', 0.9907421]
```

### 通过命令行使用

查看帮助信息
```bash
paddleocr -h
```

* 检测+识别全流程
```bash
paddleocr --image_dir PaddleOCR/doc/imgs/11.jpg
```
结果是一个list，每个item包含了文本框，文字和识别置信度
```bash
[[[24.0, 36.0], [304.0, 34.0], [304.0, 72.0], [24.0, 74.0]], ['纯臻营养护发素', 0.964739]]
[[[24.0, 80.0], [172.0, 80.0], [172.0, 104.0], [24.0, 104.0]], ['产品信息/参数', 0.98069626]]
[[[24.0, 109.0], [333.0, 109.0], [333.0, 136.0], [24.0, 136.0]], ['（45元/每公斤，100公斤起订）', 0.9676722]]
......
```

* 单独执行检测
```bash
paddleocr --image_dir PaddleOCR/doc/imgs/11.jpg --rec false
```
结果是一个list，每个item只包含文本框
```bash
[[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]]
[[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]]
[[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]]
......
```

* 单独执行识别
```bash
paddleocr --image_dir PaddleOCR/doc/imgs_words/ch/word_1.jpg --det false
```

结果是一个list，每个item只包含识别结果和识别置信度
```bash
['韩国小馆', 0.9907421]
```

## 参数说明

| 字段                    | 说明                                                                                                                                                                                                                 | 默认值                  |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| use_gpu                 | 是否使用GPU                                                                                                                                                                                                          | TRUE                    |
| gpu_mem                 | 初始化占用的GPU内存大小                                                                                                                                                                                              | 8000M                   |
| image_dir               | 通过命令行调用时执行预测的图片或文件夹路径                                                                                                                                                                           |                         |
| det_algorithm           | 使用的检测算法类型                                                                                                                                                                                                   | DB                      |
| det_model_dir          |  检测模型所在文件夹。传参方式有两种，1. None: 自动下载内置模型到 `~/.paddleocr/det`；2.自己转换好的inference模型路径，模型路径下必须包含model和params文件 |   None        |
| det_max_side_len        | 检测算法前向时图片长边的最大尺寸，当长边超出这个值时会将长边resize到这个大小，短边等比例缩放                                                                                                                         | 960                     |
| det_db_thresh           | DB模型输出预测图的二值化阈值                                                                                                                                                                                         | 0.3                     |
| det_db_box_thresh       | DB模型输出框的阈值，低于此值的预测框会被丢弃                                                                                                                                                                           | 0.5                     |
| det_db_unclip_ratio     | DB模型输出框扩大的比例                                                                                                                                                                                               | 2                       |
| det_east_score_thresh   | EAST模型输出预测图的二值化阈值                                                                                                                                                                                       | 0.8                     |
| det_east_cover_thresh   | EAST模型输出框的阈值，低于此值的预测框会被丢弃                                                                                                                                                                         | 0.1                     |
| det_east_nms_thresh     | EAST模型输出框NMS的阈值                                                                                                                                                                                              | 0.2                     |
| rec_algorithm           | 使用的识别算法类型                                                                                                                                                                                                   | CRNN                    |
| rec_model_dir          | 识别模型所在文件夹。传承那方式有两种，1. None: 自动下载内置模型到 `~/.paddleocr/rec`；2.自己转换好的inference模型路径，模型路径下必须包含model和params文件 | None |
| rec_image_shape         | 识别算法的输入图片尺寸                                                                                                                                                                                             | "3,32,320"              |
| rec_char_type           | 识别算法的字符类型，中文(ch)或英文(en)                                                                                                                                                                               | ch                      |
| rec_batch_num           | 进行识别时，同时前向的图片数                                                                                                                                                                                         | 30                      |
| max_text_length         | 识别算法能识别的最大文字长度                                                                                                                                                                                         | 25                      |
| rec_char_dict_path      | 识别模型字典路径，当rec_model_dir使用方式2传参时需要修改为自己的字典路径                                                                                                                                                | ./ppocr/utils/ppocr_keys_v1.txt                        |
| use_space_char          | 是否识别空格                                                                                                                                                                                                         | TRUE                    |
| enable_mkldnn           | 是否启用mkldnn                                                                                                                                                                                                       | FALSE                   |
| det                     | 前向时使用启动检测                                                                                                                                                                                                   | TRUE                    |
| rec                     | 前向时是否启动识别                                                                                                                                                                                                   | TRUE                    |
