from paddleocr import PaddleOCR, draw_ocr

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, lang="en", det_db_box_thresh=0.02, use_dilation=True, det_db_thresh=0.02, det_db_unclip_ratio=1.4, max_batch_size=20, det_pse_box_thresh=0.3)
img_path = '/Users/vx/Documents/GitHub/PaddleOCR/doc/imgs/bilibili/001.jpg'
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)

# 显示结果
from PIL import Image

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/Users/vx/Documents/GitHub/PaddleOCR/doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('bilibili_table_visualized_optimized.jpg')
