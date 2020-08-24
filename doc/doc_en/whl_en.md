# paddleocr package

## Get started quickly
### install package
install by pypi
```bash
pip install paddleocr
```

build own whl package and install
```bash
python setup.py bdist_wheel
pip install dist/paddleocr-0.0.3-py3-none-any.whl
```
### 1. Use by code

* detection and recognition
```python
from paddleocr import PaddleOCR,draw_ocr
ocr = PaddleOCR() # need to run only once to download and load model into memory
img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
result = ocr.ocr(img_path)
for line in result:
    print(line)

# draw result
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

Output will be a list, each item contains bounding box, text and recognition confidence
```bash
[[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]]
[[[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]], ['We would like to thank all the designers and', 0.9357758]]
[[[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]], ['contributors whohave been involved in the', 0.9592447]]
......
```

Visualization of results

<div align="center">
    <img src="../imgs_results/whl/12_det_rec.jpg" width="800">
</div>

* only detection
```python
from paddleocr import PaddleOCR,draw_ocr
ocr = PaddleOCR() # need to run only once to download and load model into memory
img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
result = ocr.ocr(img_path,rec=False)
for line in result:
    print(line)

# draw result
from PIL import Image

image = Image.open(img_path).convert('RGB')
im_show = draw_ocr(image, result, txts=None, scores=None, font_path='/path/to/PaddleOCR/doc/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

Output will be a list, each item only contains bounding box
```bash
[[756.0, 812.0], [805.0, 812.0], [805.0, 830.0], [756.0, 830.0]]
[[820.0, 803.0], [1085.0, 801.0], [1085.0, 836.0], [820.0, 838.0]]
[[393.0, 801.0], [715.0, 805.0], [715.0, 839.0], [393.0, 836.0]]
......
```

Visualization of results

<div align="center">
    <img src="../imgs_results/whl/12_det.jpg" width="800">
</div>

* only recognition
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR() # need to run only once to load model into memory
img_path = 'PaddleOCR/doc/imgs_words_en/word_10.png'
result = ocr.ocr(img_path,det=False)
for line in result:
    print(line)
```

Output will be a list, each item contains text and recognition confidence
```bash
['PAIN', 0.990372]
```

### Use by command line

show help information
```bash
paddleocr -h
```

* detection and recognition
```bash
paddleocr --image_dir PaddleOCR/doc/imgs_en/img_12.jpg
```

Output will be a list, each item contains bounding box, text and recognition confidence
```bash
[[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]]
[[[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]], ['We would like to thank all the designers and', 0.9357758]]
[[[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]], ['contributors whohave been involved in the', 0.9592447]]
......
```

* only detection
```bash
paddleocr --image_dir PaddleOCR/doc/imgs_en/img_12.jpg --rec false
```

Output will be a list, each item only contains bounding box
```bash
[[756.0, 812.0], [805.0, 812.0], [805.0, 830.0], [756.0, 830.0]]
[[820.0, 803.0], [1085.0, 801.0], [1085.0, 836.0], [820.0, 838.0]]
[[393.0, 801.0], [715.0, 805.0], [715.0, 839.0], [393.0, 836.0]]
......
```

* only recognition
```bash
paddleocr --image_dir PaddleOCR/doc/imgs_words_en/word_10.png --det false
```

Output will be a list, each item contains text and recognition confidence
```bash
['PAIN', 0.990372]
```

## Use custom model
When the built-in model cannot meet the needs, you need to use your own trained model.
First, refer to the first section of [inference_en.md](./inference_en.md) to convert your det and rec model to inference model, and then use it as follows

### 1. Use by code

```python
from paddleocr import PaddleOCR,draw_ocr
# The path of detection and recognition model must contain model and params files
ocr = PaddleOCR(det_model_dir='{your_det_model_dir}',rec_model_dir='{your_rec_model_dir}å')
img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
result = ocr.ocr(img_path)
for line in result:
    print(line)

# draw result
from PIL import Image
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

### Use by command line

```bash
paddleocr --image_dir PaddleOCR/doc/imgs/11.jpg --det_model_dir {your_det_model_dir} --rec_model_dir {your_rec_model_dir}
```

## Parameter Description

| Parameter                    | Description                                                                                                                                                                                                                 | Default value                  |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| use_gpu                 | use GPU or not                                                                                                                                                                                                          | TRUE                    |
| gpu_mem                 | GPU memory size used for initialization                                                                                                                                                                                              | 8000M                   |
| image_dir               | The images path or folder path for predicting when used by the command line                                                                                                                                                                           |                         |
| det_algorithm           | Type of detection algorithm selected                                                                                                                                                                                                   | DB                      |
| det_model_dir           | the text detection inference model folder. There are two ways to transfer parameters, 1. None: Automatically download the built-in model to `~/.paddleocr/det`; 2. The path of the inference model converted by yourself, the model and params files must be included in the model path | None           |
| det_max_side_len        | The maximum size of the long side of the image. When the long side exceeds this value, the long side will be resized to this size, and the short side will be scaled proportionally                                                                                                                         | 960                     |
| det_db_thresh           | Binarization threshold value of DB output map                                                                                                                                                                                        | 0.3                     |
| det_db_box_thresh       | The threshold value of the DB output box. Boxes score lower than this value will be discarded                                                                                                                                                                         | 0.5                     |
| det_db_unclip_ratio     | The expanded ratio of DB output box                                                                                                                                                                                             | 2                       |
| det_east_score_thresh   | Binarization threshold value of EAST output map                                                                                                                                                                                       | 0.8                     |
| det_east_cover_thresh   | The threshold value of the EAST output box. Boxes score lower than this value will be discarded                                                                                                                                                                         | 0.1                     |
| det_east_nms_thresh     | The NMS threshold value of EAST model output box                                                                                                                                                                                              | 0.2                     |
| rec_algorithm           | Type of recognition algorithm selected                                                                                                                                                                                                | CRNN                    |
| rec_model_dir           | the text recognition inference model folder. There are two ways to transfer parameters, 1. None: Automatically download the built-in model to `~/.paddleocr/rec`; 2. The path of the inference model converted by yourself, the model and params files must be included in the model path | None |
| rec_image_shape         | image shape of recognition algorithm                                                                                                                                                                                            | "3,32,320"              |
| rec_char_type           | Character type of recognition algorithm, Chinese (ch) or English (en)                                                                                                                                                                               | ch                      |
| rec_batch_num           | When performing recognition, the batchsize of forward images                                                                                                                                                                                         | 30                      |
| max_text_length         | The maximum text length that the recognition algorithm can recognize                                                                                                                                                                                         | 25                      |
| rec_char_dict_path      | the alphabet path which needs to be modified to your own path when `rec_model_Name` use mode 2                                                                                                                                              | ./ppocr/utils/ppocr_keys_v1.txt                        |
| use_space_char          | Whether to recognize spaces                                                                                                                                                                                                         | TRUE                    |
| enable_mkldnn           | Whether to enable mkldnn                                                                                                                                                                                                       | FALSE                   |
| det                     | Enable detction when `ppocr.ocr` func exec                                                                                                                                                                                                   | TRUE                    |
| rec                     | Enable detction when `ppocr.ocr` func exec                                                                                                                                                                                                   | TRUE                    |
