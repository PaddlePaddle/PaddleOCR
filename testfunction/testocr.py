import os
from paddleocr import PaddleOCR, draw_ocr
import logging

pdfoutputimg_folder_main = './pdftoimg_main'
pdfoutputimg_binary_folder_main = './pdftoimg_binary_main'
outputfile_folder = './outputfile'
outputimg_folder = './outputimg'
outputimg_binary_folder = './outputbinaryimg'

for filename in os.listdir(pdfoutputimg_binary_folder_main):
    # print(filename)
    # print(img_folder+'/'+filename)

    ocr = PaddleOCR(use_angle_cls=True, lang='ch')#, rec_algorithm = 'chinese_cht') # need to run only once to download and load model into memory
    img_path = pdfoutputimg_binary_folder_main+'/'+filename #'PaddleOCR/doc/imgs_en/img_12.jpg'
    # img_path = './img/4-1.jpeg'

    # cv2.imread(img_path)
    # cv_img = cv2.cvtColor(np.asarray(img_path), cv2.COLOR_RGB2BGR)
    # gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # ret, bin_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #prepare output ocr result 
    result = ocr.ocr(img_path, cls=True)

    logger = logging.getLogger()
    logger.error(filename)
    logger.error('--------------filestart-----------------')


    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            # print(line)
            # record log
            # print_obj = str(line)
            #print(print_obj)
            logger.error(str(line))
    logger.error('---------------fileend----------------')
    # 去掉副檔名
    filename = os.path.splitext(filename)[0]



    # draw result
    from PIL import Image
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    print('---------------------------')
    # print(boxes)
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    #im_show = draw_ocr(image, boxes, txts, scores, font_path='./PaddleOCR/doc/fonts/simfang.ttf')
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./doc/fonts/chinese_cht.ttf')
    im_show = Image.fromarray(im_show)
    im_show.save(os.path.join(outputimg_binary_folder, f'{filename}_result.jpg'))


    # save to file
    righttop_location = [topright[2] for topright in boxes]
    # right to left order [right][top]
    righttop_order = [sorted(righttop_location,reverse=True)]

    # 搭配 with 寫入檔案
    outputresultfile = filename.split("_page")[0]
    print(outputresultfile)
    output_path = os.path.join(outputfile_folder, f'{outputresultfile}_binary.txt')
    if os.path.exists(output_path):
        mode = 'a'  # 如果檔案存在，使用附加模式
    else:
        mode = 'w'  # 如果檔案不存在，使用寫入模式

    with open(output_path, mode) as f:
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