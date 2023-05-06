from paddleocr import PaddleOCR, draw_ocr
from pdf2image import convert_from_path
import os, cv2, numpy as np
import logging
import datetime
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
# to switch the language model in order.

pdf_folder = './pdf/'
img_folder = './img'
pdfoutputimg_folder = './pdftoimg'
outputfile_folder = './outputfile'
outputlog_folder = './outputlog'
outputimg_folder = './outputimg'

file_names = os.listdir(img_folder)

if not os.path.exists(outputimg_folder):
    os.makedirs(outputimg_folder)

if not os.path.exists(outputfile_folder):
    os.makedirs(outputfile_folder)


# do pdf to image
for filename in os.listdir(pdf_folder):
    pages = convert_from_path(pdf_folder + filename,
                              grayscale = True)
    # 保存圖像
    for i, page in enumerate(pages):
        output_path = os.path.join(pdfoutputimg_folder, f'{filename}_page_{i+1}.jpg')
        page.save(output_path, 'JPEG')

# 配置 logging
logging.basicConfig(
                    level=logging.INFO, # 設置log级别为 INFO
                    format ='%(asctime)s - %(levelname)s - %(message)s' # 设置日志格式
                    )


# create logFileHandler
log_filename = datetime.datetime.now().strftime("%Y-%m-%d.log")#-%M-%S
log_path = os.path.join(outputlog_folder, log_filename)

# 指定log路径和檔名
log_dir = os.path.dirname(log_path)
if not os.path.exists(log_dir):
   os.makedirs(log_dir)

file_handler = logging.FileHandler(log_path) #log_path)

file_handler.setLevel(logging.ERROR)
# 將 FileHandler 添加到 logger
logger = logging.getLogger()
logger.addHandler(file_handler)
logger.error('-------------log start------------------')


# do ocr
for filename in os.listdir(pdfoutputimg_folder):
    # print(filename)
    # print(img_folder+'/'+filename)

    ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory
    img_path = pdfoutputimg_folder+'/'+filename #'PaddleOCR/doc/imgs_en/img_12.jpg'
    # img_path = './img/4-1.jpeg'

    # cv2.imread(img_path)
    # cv_img = cv2.cvtColor(np.asarray(img_path), cv2.COLOR_RGB2BGR)
    # gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # ret, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #prepare output ocr result 
    result = ocr.ocr(img_path, cls=True)

    logger.error(filename)
    logger.error('--------------filestart-----------------')


    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)
            # record log
            print_obj = str(line)
            #print(print_obj)
            logger.error(print_obj)
    logger.error('---------------fileend----------------')

    # draw result
    from PIL import Image
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    print('---------------------------')
    print(boxes)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(os.path.join(outputimg_folder, f'{filename}_result.jpg'))


    # save to file
    righttop_location = [topright[2] for topright in boxes]
    # right to left order [right][top]
    righttop_order = [sorted(righttop_location,reverse=True)]

    # 搭配 with 寫入檔案
    output_path = os.path.join(outputfile_folder, f'{filename}.txt')
    with open(output_path, 'w') as f:
        # get each box righttop(x, y)
        for righttop in righttop_order[0]:
            for line in result:
                # print(righttop)
                
                # check if match the righttop(x, y)
                if line[0][2] == righttop:
                    # write file
                    f.write(line[1][0]+'\n')
                    break
            else:
                continue