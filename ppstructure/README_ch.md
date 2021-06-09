# TableStructurer

1. 代码使用
```python
import cv2
from paddlestructure import PaddleStructure,draw_result

table_engine = PaddleStructure(
    output='./output/table',
    show_log=True)

img_path = '../doc/table/1.png'
img = cv2.imread(img_path)
result = table_engine(img)
for line in result:
    print(line)

from PIL import Image

font_path = 'path/tp/PaddleOCR/doc/fonts/simfang.ttf'
image = Image.open(img_path).convert('RGB')
im_show = draw_result(image, result,font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
```

2. 命令行使用
```bash
paddlestructure --image_dir=../doc/table/1.png
```
