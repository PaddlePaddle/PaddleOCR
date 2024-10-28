---
comments: true
hide:
  - navigation
---

### 安装

```bash linenums="1"
pip install paddleocr
```

### 使用

=== "文本检测+方向分类+文本识别"

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

    输出示例：

    ```python linenums="1"
    [[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]]
    [[[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]], ['We would like to thank all the designers and', 0.9357758]]
    [[[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]], ['contributors whohave been involved in the', 0.9592447]]
    ......
    ```

=== "文本检测+文本识别"

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

    输出示例：

    ```python linenums="1"
    [[[442.0, 173.0], [1169.0, 173.0], [1169.0, 225.0], [442.0, 225.0]], ['ACKNOWLEDGEMENTS', 0.99283075]]
    [[[393.0, 340.0], [1207.0, 342.0], [1207.0, 389.0], [393.0, 387.0]], ['We would like to thank all the designers and', 0.9357758]]
    [[[399.0, 398.0], [1204.0, 398.0], [1204.0, 433.0], [399.0, 433.0]], ['contributors whohave been involved in the', 0.9592447]]
    ......
    ```

=== "方向分类+文本识别"

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

    输出示例：

    ```python linenums="1"
    ['PAIN', 0.990372]
    ```

=== "只有文本检测"

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

    输出示例：

    ```python linenums="1"
    [[756.0, 812.0], [805.0, 812.0], [805.0, 830.0], [756.0, 830.0]]
    [[820.0, 803.0], [1085.0, 801.0], [1085.0, 836.0], [820.0, 838.0]]
    [[393.0, 801.0], [715.0, 805.0], [715.0, 839.0], [393.0, 836.0]]
    ......
    ```

=== "只有识别"

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

    输出示例：

    ```python linenums="1"
    ['PAIN', 0.990372]
    ```

=== "只有方向分类"

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

    输出示例：

    ```python linenums="1"
    ['0', 0.99999964]
    ```

### 在线demo

- PP-OCRv4 在线体验地址：<https://aistudio.baidu.com/community/app/91660>
- SLANet 在线体验地址：<https://aistudio.baidu.com/community/app/91661>
- PP-ChatOCRv3-doc 在线体验地址：<https://aistudio.baidu.com/community/app/182491>
- PP-ChatOCRv2-common 在线体验地址：<https://aistudio.baidu.com/community/app/91662>
- PP-ChatOCRv2-doc 在线体验地址：<https://aistudio.baidu.com/community/app/70303>

### 相关文档

- [一键调用17个PaddleOCR核心模型](https://paddlepaddle.github.io/PaddleOCR/latest/paddlex/quick_start.html)
- 一行命令快速使用：[文本检测识别（中英文/多语言）](https://paddlepaddle.github.io/PaddleOCR/latest/ppocr/overview.html)
- 一行命令快速使用：[文档分析](https://paddlepaddle.github.io/PaddleOCR/latest/ppstructure/overview.html)
