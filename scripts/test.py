from paddleocr import PaddleOCR
import fitz
from PIL import Image
import cv2
import numpy as np
import fitz
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import math

DET_NAME="det_finetune_ch_PP-OCRv3_det_distill_train"
CLS_NAME="ch_ppocr_mobile_v2.0_cls_infer"
REC_NAME="ch_PP-OCRv4_rec_server_infer"

# INPUT_IMAGE='./Image.jpg'
INPUT_IMAGE='E:\\projects\\ballooning\\dataset\\det\\val\\93128019.pdf-0.png'
OUTPUT_NAME="det+cls+rec-"+DET_NAME+".jpg"
OUTPUT_TXT_NAME="det+cls+rec-"+DET_NAME+".txt"
OUTPUT_IMAGE='E:\\projects\\ballooning\\scripts\\ocr\\outputs\\images\\'+OUTPUT_NAME
REC_RESULT_PATH='E:\\projects\\ballooning\\scripts\\ocr\\outputs\\images\\'+OUTPUT_TXT_NAME

FONT_PATH = 'E:\\projects\\ballooning\\PaddleOCR\\doc\\fonts\\simfang.ttf'
DET_MODEL=r"E:\\projects\\ballooning\\PaddleOCR\\output_inference\\det_finetune_ch_PP-OCRv3_det_distill_train"
CLS_MODEL="E:\\projects\\ballooning\\models\\cls\\"+CLS_NAME
REC_MODEL="E:\\projects\\ballooning\\models\\rec\\"+REC_NAME
LANGUAGE="ch"


def resize_img(img, input_size=600):
    """
    resize img and limit the longest side of the image to input_size
    """
    img = np.array(img)
    im_shape = img.shape
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(input_size) / float(im_size_max)
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
    return img


def str_count(s):
    """
    Count the number of Chinese characters,
    a single English character and a single number
    equal to half the length of Chinese characters.
    args:
        s(string): the input of string
    return(int):
        the number of Chinese characters
    """
    import string
    count_zh = count_pu = 0
    s_len = len(s)
    en_dg_count = 0
    for c in s:
        if c in string.ascii_letters or c.isdigit() or c.isspace():
            en_dg_count += 1
        elif c.isalpha():
            count_zh += 1
        else:
            count_pu += 1
    return s_len - math.ceil(en_dg_count / 2)

def text_visual(texts,
                scores,
                img_h=400,
                img_w=600,
                threshold=0.,
                font_path="./doc/simfang.ttf"):
    """
    create new blank img and draw txt on it
    args:
        texts(list): the text will be draw
        scores(list|None): corresponding score of each txt
        img_h(int): the height of blank img
        img_w(int): the width of blank img
        font_path: the path of font which is used to draw text
    return(array):
    """
    if scores is not None:
        assert len(texts) == len(
            scores), "The number of txts and corresponding scores must match"

    def create_blank_img():
        blank_img = np.ones(shape=[img_h, img_w], dtype=np.int8) * 255
        blank_img[:, img_w - 1:] = 0
        blank_img = Image.fromarray(blank_img).convert("RGB")
        draw_txt = ImageDraw.Draw(blank_img)
        return blank_img, draw_txt

    blank_img, draw_txt = create_blank_img()

    font_size = 20
    txt_color = (0, 0, 0)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    gap = font_size + 5
    txt_img_list = []
    count, index = 1, 0
    for idx, txt in enumerate(texts):
        index += 1
        if scores[idx] < threshold or math.isnan(scores[idx]):
            index -= 1
            continue
        first_line = True
        while str_count(txt) >= img_w // font_size - 4:
            tmp = txt
            txt = tmp[:img_w // font_size - 4]
            if first_line:
                new_txt = str(index) + ': ' + txt
                first_line = False
            else:
                new_txt = '    ' + txt
            draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
            txt = tmp[img_w // font_size - 4:]
            if count >= img_h // gap - 1:
                txt_img_list.append(np.array(blank_img))
                blank_img, draw_txt = create_blank_img()
                count = 0
            count += 1
        if first_line:
            new_txt = str(index) + ': ' + txt + '   ' + '%.3f' % (scores[idx])
        else:
            new_txt = "  " + txt + "  " + '%.3f' % (scores[idx])
        draw_txt.text((0, gap * count), new_txt, txt_color, font=font)
        # whether add new blank img or not
        if count >= img_h // gap - 1 and idx + 1 < len(texts):
            txt_img_list.append(np.array(blank_img))
            blank_img, draw_txt = create_blank_img()
            count = 0
        count += 1
    txt_img_list.append(np.array(blank_img))
    if len(txt_img_list) == 1:
        blank_img = np.array(txt_img_list[0])
    else:
        blank_img = np.concatenate(txt_img_list, axis=1)
    return np.array(blank_img)

def draw_ocr2(image,
             boxes,
             txts=None,
             scores=None,
             drop_score=0.5,
             font_path="./doc/fonts/simfang.ttf"):
    """
    Visualize the results of OCR detection and recognition
    args:
        image(Image|array): RGB image
        boxes(list): boxes with shape(N, 4, 2)
        txts(list): the texts
        scores(list): txxs corresponding scores
        drop_score(float): only scores greater than drop_threshold will be visualized
        font_path: the path of font which is used to draw text
    return(array):
        the visualized img
    """
    if scores is None:
        scores = [1] * len(boxes)
    box_num = len(boxes)
    for i in range(box_num):
        if scores is not None and (scores[i] < drop_score or
                                   math.isnan(scores[i])):
            continue
        box = np.reshape(np.array(boxes[i]), [-1, 1, 2]).astype(np.int64)
        image = cv2.polylines(np.array(image), [box], True, (255, 0, 0), 2)
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (box[2][0][0], box[2][0][1] - 5)

        # fontScale
        fontScale = 0.8

        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.putText() method
        image = cv2.putText(image, str(i), org, font,
                           fontScale, color, thickness, cv2.LINE_AA)
        # image = cv2.putText(np.array(image), "1", (box[0][0],box[0][1]))
    if txts is not None:
        img = np.array(resize_img(image, input_size=600))
        txt_img = text_visual(
            txts,
            scores,
            img_h=img.shape[0],
            img_w=600,
            threshold=drop_score,
            font_path=font_path)
        img = np.concatenate([np.array(img), np.array(txt_img)], axis=1)
        return img
    return image

def visualize_result_image(input_path, result, mode):
    result = result[0]
    image = Image.open(input_path).convert('RGB')

    if mode == "det":
        im_show = draw_ocr2(image, result, txts=None, scores=None, font_path=FONT_PATH)
    elif mode == "det+cls+rec": 
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr2(image, boxes, txts=None, scores=None, font_path=FONT_PATH)
        with open(REC_RESULT_PATH, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(txts):
                f.write(f"{idx}   -    {item}")
                f.write('\n')

    im_show = Image.fromarray(im_show)
    im_show.save(OUTPUT_IMAGE)


def visualize_result_pdf(input_path, result, mode):
    imgs = []
    with fitz.open(input_path) as pdf:
        for pg in range(0, pdf.page_count):
            page = pdf[pg]
            mat = fitz.Matrix(2, 2)
            pm = page.get_pixmap(matrix=mat, alpha=False)
            # if width or height > 2000 pixels, don't enlarge the image
            if pm.width > 2000 or pm.height > 2000:
                pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

            img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            imgs.append(img)
    for idx in range(len(result)):
        res = result[idx]
        image = imgs[idx]

        if mode == "det":
            im_show = draw_ocr2(image, res, txts=None, scores=None, font_path=FONT_PATH)
        elif mode == "det+cls+rec": 
            boxes = [line[0] for line in res]
            txts = [line[1][0] for line in res]
            scores = [line[1][1] for line in res]
            im_show = draw_ocr2(image, boxes, txts, scores, font_path=FONT_PATH)

        im_show = Image.fromarray(im_show)
        im_show.save(OUTPUT_IMAGE.split('.')[0] + '_page_{}.jpg'.format(idx))



def inference(mode):
    # Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
    # 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
    ocr = PaddleOCR(
        use_angle_cls=True, 
        lang=LANGUAGE,
        det_model_dir=DET_MODEL,
        rec_model_dir=REC_MODEL,
        cls_model_dir=CLS_MODEL,
        page_num=0,
        # det_max_side_len=10000,
        det_db_box_thresh=0.6,
        det_db_score_mode='slow',
        det_limit_side_len=8000,
        save_crop_res=False
        )  
    # need to run only once to download and load model into memory
    # default en_PP-OCRv3_det_infer & en_PP-OCRv4_rec_infer for en
    # https://github.com/PaddlePaddle/PaddleOCR/blob/fa93f61cc5a49f8e2c80c665d2e57bb1cd15acfc/doc/doc_ch/whl.md 
    img_path = INPUT_IMAGE

    if mode == "det":  
        result = ocr.ocr(
            img_path,
            rec=False)
    elif mode == "det+cls+rec": 
        result = ocr.ocr(img_path, cls=True)
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            print(line)

    # 显示结果
    if ".png" in INPUT_IMAGE:
        visualize_result_image(img_path, result, mode)
    elif ".pdf" in INPUT_IMAGE:
        visualize_result_pdf(img_path, result, mode)


if __name__ == "__main__":
    # det_cls_rec()
    inference(mode="det+cls+rec")