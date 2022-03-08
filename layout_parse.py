import os
import cv2
from paddleocr import PPStructure,draw_structure_result,save_structure_res

table_engine = PPStructure(show_log=True,
                           det_db_box_thresh=0.02,
                           use_dilation=True,
                           det_db_thresh=0.02,
                           det_db_unclip_ratio=0.35,
                           max_batch_size=50,
                           det_pse_box_thresh=0.3)

save_folder = '/Users/vx/Documents/GitHub/PaddleOCR/ppstructure/output/table'
img_path = '/Users/vx/Documents/GitHub/BigoneMR/imgs/kano/26.jpg'
img = cv2.imread(img_path)
result = table_engine(img)
save_structure_res(result, save_folder, os.path.basename(img_path).split('.')[0])

for line in result:
    line.pop('img')
    print(line)

from PIL import Image

font_path = '/Users/vx/Documents/GitHub/PaddleOCR/doc/fonts/simfang.ttf' # PaddleOCR下提供字体包
image = Image.open(img_path).convert('RGB')
im_show = draw_structure_result(image, result, font_path=font_path)
im_show = Image.fromarray(im_show)
im_show.save('/Users/vx/Documents/GitHub/PaddleOCR/ppstructure/output/table/result.jpg')
print("IMAGE PATH IS: " + str(img_path))
