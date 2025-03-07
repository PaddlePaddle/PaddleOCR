---
comments: true
hide:
  - navigation
---

### Install

#### 1. Install PaddlePaddle

> If you don't have a basic Python runtime environment, please refer to [Running environment preparation](./ppocr/environment.en.md).

=== "CPU installation"

    ```bash linenums="1"
    python -m pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    ```

=== "GPU installation"

    Since the GPU needs to be installed and used according to the specific CUDA version, the following only takes the Linux platform, pip installation of NVIDIA GPU, CUDA11.8 as an example. For other platforms, please refer to the instructions in [PaddlePaddle official website installation document](https://www.paddlepaddle.org.cn/en).

    ```bash linenums="1"
    python -m pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    ```

NOTE: You can customize the storage location for OCR models by setting the environment variable `PADDLE_OCR_BASE_DIR`. If this variable is not set, the models will be downloaded to the following default locations:

- On Linux/macOS: `${HOME}/.paddleocr`
- On Windows: `C:\Users\{username}\.paddleocr`

### Use by code

=== "Detection + Classification + Recognition"

    ```python linenums="1"
    from paddleocr import PaddleOCR, draw_ocr

    # Paddleocr supports Chinese, English, French, German, Korean and Japanese
    # You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
    # to switch the language model in order
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
    img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
    result = ocr.ocr(img_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # draw result
    from PIL import Image
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')
    ```

    Output will be a list, each item contains bounding box, text and recognition confidence：

    ```python linenums="1"
    [[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]]
    [[[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]], ['We would like to thank all the designers and', 0.9357758]]
    [[[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]], ['contributors whohave been involved in the', 0.9592447]]
    ......
    ```

=== "Detection + Recognition"

    ```python linenums="1"
    from paddleocr import PaddleOCR,draw_ocr
    ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory
    img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
    result = ocr.ocr(img_path, cls=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # draw result
    from PIL import Image
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')
    ```

    Output will be a list, each item contains bounding box, text and recognition confidence：

    ```python linenums="1"
    [[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]]
    [[[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]], ['We would like to thank all the designers and', 0.9357758]]
    [[[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]], ['contributors whohave been involved in the', 0.9592447]]
    ......
    ```

=== "Classification + Recognition"

    ```python linenums="1"
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to load model into memory
    img_path = 'PaddleOCR/doc/imgs_words_en/word_10.png'
    result = ocr.ocr(img_path, det=False, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    ```

    Output will be a list, each item contains recognition text and confidence:

    ```python linenums="1"
    ['PAIN', 0.990372]
    ```

=== "Only detection"

    ```python linenums="1"
    from paddleocr import PaddleOCR,draw_ocr
    ocr = PaddleOCR() # need to run only once to download and load model into memory
    img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'
    result = ocr.ocr(img_path,rec=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # draw result
    from PIL import Image
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    im_show = draw_ocr(image, result, txts=None, scores=None, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save('result.jpg')
    ```

    Output will be a list, each item only contains bounding box:

    ```python linenums="1"
    [[756.0, 812.0], [805.0, 812.0], [805.0, 830.0], [756.0, 830.0]]
    [[820.0, 803.0], [1085.0, 801.0], [1085.0, 836.0], [820.0, 838.0]]
    [[393.0, 801.0], [715.0, 805.0], [715.0, 839.0], [393.0, 836.0]]
    ......
    ```

=== "Only recognition"

    ```python linenums="1"
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(lang='en') # need to run only once to load model into memory
    img_path = 'PaddleOCR/doc/imgs_words_en/word_10.png'
    result = ocr.ocr(img_path, det=False, cls=False)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    ```

    Output will be a list, each item contains recognition text and confidence：

    ```python linenums="1"
    ['PAIN', 0.990372]
    ```

=== "Only classification"

    ```python linenums="1"
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True) # need to run only once to load model into memory
    img_path = 'PaddleOCR/doc/imgs_words_en/word_10.png'
    result = ocr.ocr(img_path, det=False, rec=False, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
    ```

    Output will be a list, each item contains classification result and confidence

    ```python linenums="1"
    ['0', 0.99999964]
    ```

### Use by command line

Show help information

```bash linenums="1"
paddleocr -h
```

=== "Detection + Classification + Recognition"

    ```bash linenums="1"
    paddleocr --image_dir PaddleOCR/doc/imgs_en/img_12.jpg --use_angle_cls true --lang en
    ```

    Output will be a list, each item contains bounding box, text and recognition confidence

    ```python linenums="1"
    [[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]]
    [[[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]], ['We would like to thank all the designers and', 0.9357758]]
    [[[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]], ['contributors whohave been involved in the', 0.9592447]]
    ......
    ```

    pdf file is also supported, you can infer the first few pages by using the page_num parameter, the default is 0, which means infer all pages

    ```bash linenums="1"
    paddleocr --image_dir ./xxx.pdf --use_angle_cls true --use_gpu false --page_num 2
    ```

=== "Detection + Recognition"

    ```bash linenums="1"
    paddleocr --image_dir PaddleOCR/doc/imgs_en/img_12.jpg --lang en
    ```

    Output will be a list, each item contains bounding box, text and recognition confidence

    ```python linenums="1"
    [[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]]
    [[[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]], ['We would like to thank all the designers and', 0.9357758]]
    [[[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]], ['contributors whohave been involved in the', 0.9592447]]
    ......
    ```

=== "Classification + Recognition"

    ```bash linenums="1"
    paddleocr --image_dir PaddleOCR/doc/imgs_words_en/word_10.png --use_angle_cls true --det false --lang en
    ```

    Output will be a list, each item contains text and recognition confidence

    ```python linenums="1"
    ['PAIN', 0.990372]
    ```

=== "Only detection"

    ```bash linenums="1"
    paddleocr --image_dir PaddleOCR/doc/imgs_en/img_12.jpg --rec false
    ```

    Output will be a list, each item only contains bounding box

    ```python linenums="1"
    [[756.0, 812.0], [805.0, 812.0], [805.0, 830.0], [756.0, 830.0]]
    [[820.0, 803.0], [1085.0, 801.0], [1085.0, 836.0], [820.0, 838.0]]
    [[393.0, 801.0], [715.0, 805.0], [715.0, 839.0], [393.0, 836.0]]
    ......
    ```

=== "Only recognition"

    ```bash linenums="1"
    paddleocr --image_dir PaddleOCR/doc/imgs_words_en/word_10.png --det false --lang en
    ```

    Output will be a list, each item contains text and recognition confidence

    ```python linenums="1"
    ['PAIN', 0.990372]
    ```

=== "Only classification"

    ```bash linenums="1"
    paddleocr --image_dir PaddleOCR/doc/imgs_words_en/word_10.png --use_angle_cls true --det false --rec false
    ```

    Output will be a list, each item contains classification result and confidence

    ```python linenums="1"
    ['0', 0.99999964]
    ```

For more detailed documentation, please go to: [PaddleOCR Quick Start](./ppocr/quick_start.en.md)

### Online Demo

- PP-OCRv4 online experience：<https://aistudio.baidu.com/aistudio/projectdetail/6611435>
- SLANet online experience：<https://aistudio.baidu.com/community/app/91661>
- PP-ChatOCRv3-doc online experience：<https://aistudio.baidu.com/community/app/182491>
- PP-ChatOCRv2-common online experience：<https://aistudio.baidu.com/community/app/91662>
- PP-ChatOCRv2-doc online experience：<https://aistudio.baidu.com/community/app/70303>

### Other resources

- [One-Click Call for 48 Core PaddleOCR Models](https://paddlepaddle.github.io/PaddleOCR/latest/en/paddlex/quick_start.html)
- One line of code quick use: [Text Detection and Recognition (Chinese/English/Multilingual)](https://paddlepaddle.github.io/PaddleOCR/latest/en/ppocr/overview.html)
- One line of code quick use: [Document Analysis](https://paddlepaddle.github.io/PaddleOCR/latest/en/ppstructure/overview.html)
