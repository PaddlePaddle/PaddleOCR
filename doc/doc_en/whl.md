# ppcor package 

## Get started quickly
### 1. Use by code

detection and recognition
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(model_storage_directory='./model') # need to run only once to load model into memory
img = 'PaddleOCR/doc/imgs/11.jpg'
result = ocr.ocr(img)
for line in result:
    print(line)
```

Output will be a list, each item contains bounding box, text and recognition confidence
```bash
[[[24.0, 36.0], [304.0, 34.0], [304.0, 72.0], [24.0, 74.0]], ['纯臻营养护发素', 0.964739]]
[[[24.0, 80.0], [172.0, 80.0], [172.0, 104.0], [24.0, 104.0]], ['产品信息/参数', 0.98069626]]
[[[24.0, 109.0], [333.0, 109.0], [333.0, 136.0], [24.0, 136.0]], ['（45元/每公斤，100公斤起订）', 0.9676722]]
[[[22.0, 140.0], [284.0, 140.0], [284.0, 167.0], [22.0, 167.0]], ['每瓶22元，1000瓶起订）', 0.97444016]]
[[[22.0, 174.0], [85.0, 174.0], [85.0, 198.0], [22.0, 198.0]], ['【品牌】', 0.8187138]]
[[[89.0, 176.0], [301.0, 176.0], [301.0, 196.0], [89.0, 196.0]], ['：代加工方式/OEMODM', 0.9421848]]
[[[23.0, 205.0], [85.0, 205.0], [85.0, 229.0], [23.0, 229.0]], ['【品名】', 0.76008326]]
[[[88.0, 204.0], [235.0, 206.0], [235.0, 229.0], [88.0, 227.0]], ['：纯臻营养护发素', 0.9633639]]
[[[23.0, 236.0], [121.0, 236.0], [121.0, 261.0], [23.0, 261.0]], ['【产品编号】', 0.84101385]]
[[[110.0, 239.0], [239.0, 239.0], [239.0, 256.0], [110.0, 256.0]], ['1：YM-X-3011', 0.8621878]]
[[[414.0, 233.0], [430.0, 233.0], [430.0, 304.0], [414.0, 304.0]], ['ODM OEM', 0.9084018]]
[[[23.0, 268.0], [183.0, 268.0], [183.0, 292.0], [23.0, 292.0]], ['【净含量】：220ml', 0.9278281]]
[[[24.0, 301.0], [118.0, 301.0], [118.0, 321.0], [24.0, 321.0]], ['【适用人群】', 0.90901047]]
[[[127.0, 300.0], [254.0, 300.0], [254.0, 323.0], [127.0, 323.0]], ['：适合所有肤质', 0.95465785]]
[[[24.0, 332.0], [117.0, 332.0], [117.0, 353.0], [24.0, 353.0]], ['【主要成分】', 0.88936955]]
[[[139.0, 332.0], [236.0, 332.0], [236.0, 352.0], [139.0, 352.0]], ['鲸蜡硬脂醇', 0.9447544]]
[[[248.0, 332.0], [345.0, 332.0], [345.0, 352.0], [248.0, 352.0]], ['燕麦B-葡聚', 0.89748293]]
[[[54.0, 363.0], [232.0, 363.0], [232.0, 383.0], [54.0, 383.0]], [' 椰油酰胺丙基甜菜碱', 0.902023]]
[[[25.0, 364.0], [64.0, 364.0], [64.0, 383.0], [25.0, 383.0]], ['糖、', 0.985203]]
[[[244.0, 363.0], [281.0, 363.0], [281.0, 382.0], [244.0, 382.0]], ['泛服', 0.44537082]]
[[[367.0, 367.0], [475.0, 367.0], [475.0, 388.0], [367.0, 388.0]], ['（成品包材）', 0.9834532]]
[[[24.0, 395.0], [120.0, 395.0], [120.0, 416.0], [24.0, 416.0]], ['【主要功能】', 0.88684446]]
[[[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]], ['：可紧致头发磷层', 0.9342501]]
[[[265.0, 395.0], [361.0, 395.0], [361.0, 415.0], [265.0, 415.0]], ['琴，从而达到', 0.8253762]]
[[[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]], ['即时持久改善头发光泽的效果，给干燥的头', 0.97785276]]
[[[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]], ['发足够的滋养', 0.9577897]]
```

only detection
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(model_storage_directory='./model') # need to run only once to load model into memory
img = 'PaddleOCR/doc/imgs/11.jpg'
result = ocr.ocr(img,rec=False)
for line in result:
    print(line)
```

Output will be a list, each item only contains bounding box
```bash
[[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]]
[[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]]
[[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]]
[[265.0, 395.0], [361.0, 395.0], [361.0, 415.0], [265.0, 415.0]]
[[24.0, 395.0], [120.0, 395.0], [120.0, 416.0], [24.0, 416.0]]
[[367.0, 367.0], [475.0, 367.0], [475.0, 388.0], [367.0, 388.0]]
[[54.0, 363.0], [232.0, 363.0], [232.0, 383.0], [54.0, 383.0]]
[[25.0, 364.0], [64.0, 364.0], [64.0, 383.0], [25.0, 383.0]]
[[244.0, 363.0], [281.0, 363.0], [281.0, 382.0], [244.0, 382.0]]
[[248.0, 332.0], [345.0, 332.0], [345.0, 352.0], [248.0, 352.0]]
[[139.0, 332.0], [236.0, 332.0], [236.0, 352.0], [139.0, 352.0]]
[[24.0, 332.0], [117.0, 332.0], [117.0, 353.0], [24.0, 353.0]]
[[127.0, 300.0], [254.0, 300.0], [254.0, 323.0], [127.0, 323.0]]
[[24.0, 301.0], [118.0, 301.0], [118.0, 321.0], [24.0, 321.0]]
[[23.0, 268.0], [183.0, 268.0], [183.0, 292.0], [23.0, 292.0]]
[[110.0, 239.0], [239.0, 239.0], [239.0, 256.0], [110.0, 256.0]]
[[23.0, 236.0], [121.0, 236.0], [121.0, 261.0], [23.0, 261.0]]
[[414.0, 233.0], [430.0, 233.0], [430.0, 304.0], [414.0, 304.0]]
[[88.0, 204.0], [235.0, 206.0], [235.0, 229.0], [88.0, 227.0]]
[[23.0, 205.0], [85.0, 205.0], [85.0, 229.0], [23.0, 229.0]]
[[89.0, 176.0], [301.0, 176.0], [301.0, 196.0], [89.0, 196.0]]
[[22.0, 174.0], [85.0, 174.0], [85.0, 198.0], [22.0, 198.0]]
[[22.0, 140.0], [284.0, 140.0], [284.0, 167.0], [22.0, 167.0]]
[[24.0, 109.0], [333.0, 109.0], [333.0, 136.0], [24.0, 136.0]]
[[24.0, 80.0], [172.0, 80.0], [172.0, 104.0], [24.0, 104.0]]
[[24.0, 36.0], [304.0, 34.0], [304.0, 72.0], [24.0, 74.0]]
```

only recognition
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(model_storage_directory='./model') # need to run only once to load model into memory
img = 'PaddleOCR/doc/imgs_words/ch/word_1.jpg'
result = ocr.ocr(img,det=False)
for line in result:
    print(line)
```

Output will be a list, each item contains text and recognition confidence
```bash
['韩国小馆', 0.9907421]
```

### Use by command line

show help information
```bash
paddleocr -h
```

detection and recognition
```bash
paddleocr --image_dir PaddleOCR/doc/imgs/11.jpg
```

Output will be a list, each item contains bounding box, text and recognition confidence
```bash
[[[24.0, 36.0], [304.0, 34.0], [304.0, 72.0], [24.0, 74.0]], ['纯臻营养护发素', 0.964739]]
[[[24.0, 80.0], [172.0, 80.0], [172.0, 104.0], [24.0, 104.0]], ['产品信息/参数', 0.98069626]]
[[[24.0, 109.0], [333.0, 109.0], [333.0, 136.0], [24.0, 136.0]], ['（45元/每公斤，100公斤起订）', 0.9676722]]
[[[22.0, 140.0], [284.0, 140.0], [284.0, 167.0], [22.0, 167.0]], ['每瓶22元，1000瓶起订）', 0.97444016]]
[[[22.0, 174.0], [85.0, 174.0], [85.0, 198.0], [22.0, 198.0]], ['【品牌】', 0.8187138]]
[[[89.0, 176.0], [301.0, 176.0], [301.0, 196.0], [89.0, 196.0]], ['：代加工方式/OEMODM', 0.9421848]]
[[[23.0, 205.0], [85.0, 205.0], [85.0, 229.0], [23.0, 229.0]], ['【品名】', 0.76008326]]
[[[88.0, 204.0], [235.0, 206.0], [235.0, 229.0], [88.0, 227.0]], ['：纯臻营养护发素', 0.9633639]]
[[[23.0, 236.0], [121.0, 236.0], [121.0, 261.0], [23.0, 261.0]], ['【产品编号】', 0.84101385]]
[[[110.0, 239.0], [239.0, 239.0], [239.0, 256.0], [110.0, 256.0]], ['1：YM-X-3011', 0.8621878]]
[[[414.0, 233.0], [430.0, 233.0], [430.0, 304.0], [414.0, 304.0]], ['ODM OEM', 0.9084018]]
[[[23.0, 268.0], [183.0, 268.0], [183.0, 292.0], [23.0, 292.0]], ['【净含量】：220ml', 0.9278281]]
[[[24.0, 301.0], [118.0, 301.0], [118.0, 321.0], [24.0, 321.0]], ['【适用人群】', 0.90901047]]
[[[127.0, 300.0], [254.0, 300.0], [254.0, 323.0], [127.0, 323.0]], ['：适合所有肤质', 0.95465785]]
[[[24.0, 332.0], [117.0, 332.0], [117.0, 353.0], [24.0, 353.0]], ['【主要成分】', 0.88936955]]
[[[139.0, 332.0], [236.0, 332.0], [236.0, 352.0], [139.0, 352.0]], ['鲸蜡硬脂醇', 0.9447544]]
[[[248.0, 332.0], [345.0, 332.0], [345.0, 352.0], [248.0, 352.0]], ['燕麦B-葡聚', 0.89748293]]
[[[54.0, 363.0], [232.0, 363.0], [232.0, 383.0], [54.0, 383.0]], [' 椰油酰胺丙基甜菜碱', 0.902023]]
[[[25.0, 364.0], [64.0, 364.0], [64.0, 383.0], [25.0, 383.0]], ['糖、', 0.985203]]
[[[244.0, 363.0], [281.0, 363.0], [281.0, 382.0], [244.0, 382.0]], ['泛服', 0.44537082]]
[[[367.0, 367.0], [475.0, 367.0], [475.0, 388.0], [367.0, 388.0]], ['（成品包材）', 0.9834532]]
[[[24.0, 395.0], [120.0, 395.0], [120.0, 416.0], [24.0, 416.0]], ['【主要功能】', 0.88684446]]
[[[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]], ['：可紧致头发磷层', 0.9342501]]
[[[265.0, 395.0], [361.0, 395.0], [361.0, 415.0], [265.0, 415.0]], ['琴，从而达到', 0.8253762]]
[[[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]], ['即时持久改善头发光泽的效果，给干燥的头', 0.97785276]]
[[[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]], ['发足够的滋养', 0.9577897]]
```

only detection
```bash
paddleocr --image_dir PaddleOCR/doc/imgs_words/ch/word_1.jpg --rec false
```

Output will be a list, each item only contains bounding box
```bash
[[26.0, 457.0], [137.0, 457.0], [137.0, 477.0], [26.0, 477.0]]
[[25.0, 425.0], [372.0, 425.0], [372.0, 448.0], [25.0, 448.0]]
[[128.0, 397.0], [273.0, 397.0], [273.0, 414.0], [128.0, 414.0]]
[[265.0, 395.0], [361.0, 395.0], [361.0, 415.0], [265.0, 415.0]]
[[24.0, 395.0], [120.0, 395.0], [120.0, 416.0], [24.0, 416.0]]
[[367.0, 367.0], [475.0, 367.0], [475.0, 388.0], [367.0, 388.0]]
[[54.0, 363.0], [232.0, 363.0], [232.0, 383.0], [54.0, 383.0]]
[[25.0, 364.0], [64.0, 364.0], [64.0, 383.0], [25.0, 383.0]]
[[244.0, 363.0], [281.0, 363.0], [281.0, 382.0], [244.0, 382.0]]
[[248.0, 332.0], [345.0, 332.0], [345.0, 352.0], [248.0, 352.0]]
[[139.0, 332.0], [236.0, 332.0], [236.0, 352.0], [139.0, 352.0]]
[[24.0, 332.0], [117.0, 332.0], [117.0, 353.0], [24.0, 353.0]]
[[127.0, 300.0], [254.0, 300.0], [254.0, 323.0], [127.0, 323.0]]
[[24.0, 301.0], [118.0, 301.0], [118.0, 321.0], [24.0, 321.0]]
[[23.0, 268.0], [183.0, 268.0], [183.0, 292.0], [23.0, 292.0]]
[[110.0, 239.0], [239.0, 239.0], [239.0, 256.0], [110.0, 256.0]]
[[23.0, 236.0], [121.0, 236.0], [121.0, 261.0], [23.0, 261.0]]
[[414.0, 233.0], [430.0, 233.0], [430.0, 304.0], [414.0, 304.0]]
[[88.0, 204.0], [235.0, 206.0], [235.0, 229.0], [88.0, 227.0]]
[[23.0, 205.0], [85.0, 205.0], [85.0, 229.0], [23.0, 229.0]]
[[89.0, 176.0], [301.0, 176.0], [301.0, 196.0], [89.0, 196.0]]
[[22.0, 174.0], [85.0, 174.0], [85.0, 198.0], [22.0, 198.0]]
[[22.0, 140.0], [284.0, 140.0], [284.0, 167.0], [22.0, 167.0]]
[[24.0, 109.0], [333.0, 109.0], [333.0, 136.0], [24.0, 136.0]]
[[24.0, 80.0], [172.0, 80.0], [172.0, 104.0], [24.0, 104.0]]
[[24.0, 36.0], [304.0, 34.0], [304.0, 72.0], [24.0, 74.0]]
```

only recognition
```bash
paddleocr --image_dir PaddleOCR/doc/imgs_words/ch/word_1.jpg --det false
```

Output will be a list, each item contains text and recognition confidence
```bash
['韩国小馆', 0.9907421]
```

## Parameter Description

| Parameter                    | Description                                                                                                                                                                                                                 | Default value                  |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
| use_gpu                 | use GPU or not                                                                                                                                                                                                          | TRUE                    |
| gpu_mem                 | GPU memory size used for initialization                                                                                                                                                                                              | 8000M                   |
| image_dir               | The images path or folder path for predicting when used by the command line                                                                                                                                                                           |                         |
| det_algorithm           | Type of detection algorithm selected                                                                                                                                                                                                   | DB                      |
| det_model_name          | There are two ways to use: 1. The name of the detection algorithm which must be in the support list(only ch_det_mv3_db is built in currently), and the supported list will be displayed when the wrong parameter is passed in. 2. The path of the inference model that has been converted by yourself. At this time, the model path must contains model and params files. When choosing this method, you need to give the name of det_algorithm | ch_det_mv3_db           |
| det_max_side_len        | The maximum size of the long side of the image. When the long side exceeds this value, the long side will be resized to this size, and the short side will be scaled proportionally                                                                                                                         | 960                     |
| det_db_thresh           | Binarization threshold value of DB output map                                                                                                                                                                                        | 0.3                     |
| det_db_box_thresh       | The threshold value of the DB output box. Boxes score lower than this value will be discarded                                                                                                                                                                         | 0.5                     |
| det_db_unclip_ratio     | The expanded ratio of DB output box                                                                                                                                                                                             | 2                       |
| det_east_score_thresh   | Binarization threshold value of EAST output map                                                                                                                                                                                       | 0.8                     |
| det_east_cover_thresh   | The threshold value of the EAST output box. Boxes score lower than this value will be discarded                                                                                                                                                                         | 0.1                     |
| det_east_nms_thresh     | The NMS threshold value of EAST model output box                                                                                                                                                                                              | 0.2                     |
| rec_algorithm           | Type of recognition algorithm selected                                                                                                                                                                                                | CRNN                    |
| rec_model_name          | There are two ways to use: 1. The name of the recognition algorithm which must be in the support list(only supports CRNN, Rosetta, STAR, RARE and other algorithms currently, but only ch_rec_mv3_crnn_enhance is built-in), and the supported list will be displayed when the wrong parameter is passed in. 2. The path of the inference model that has been converted by yourself. At this time, the model path must contains model and params files. When choosing this method, you need to give the name of rec_algorithm | ch_rec_mv3_crnn_enhance |
| rec_image_shape         | image shape of recognition algorithm                                                                                                                                                                                            | "3,32,320"              |
| rec_char_type           | Character type of recognition algorithm, Chinese (ch) or English (en)                                                                                                                                                                               | ch                      |
| rec_batch_num           | When performing recognition, the batchsize of forward images                                                                                                                                                                                         | 30                      |
| rec_char_dict_path      | the alphabet path which needs to be modified to your own path when `rec_model_Name` use mode 2                                                                                                                                              |                         |
| use_space_char          | Whether to recognize spaces                                                                                                                                                                                                         | TRUE                    |
| enable_mkldnn           | Whether to enable mkldnn                                                                                                                                                                                                       | FALSE                   |
| model_storage_directory | Download model save path when det_model_name or rec_model_name use mode 1                                                                                                                                                                                                     | ～/.ppocr                |
| det                     | Enable detction when `ppocr.ocr` func exec                                                                                                                                                                                                   | TRUE                    |
| rec                     | Enable detction when `ppocr.ocr` func exec                                                                                                                                                                                                   | TRUE                    |

## build own whl package
```bash
python setup.py bdist_wheel
```