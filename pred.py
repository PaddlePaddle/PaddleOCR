from paddleocr import PaddleOCR
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import matplotlib.pyplot as plt

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    drop_score=0.35,
    det_db_score_mode='fast',
    max_text_length=6,
    rec_batch_num=8,
    rec_algorithm="CRNN", 
    det_db_box_thresh=0.5,
    det_db_thresh=0.35,
    det_db_unclip_ratio=1.9

    # rec_char_dict_path= "./ppocr/utils/ppocr_keys_v1.txt"
) # need to run only once to download and load model into memory

# input and output paths
input_path = './content/inputs'
output_path = './content/outputs'

# Create the directories if they do not exist
if not os.path.exists(input_path):
    os.makedirs(input_path)

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
labels = []
preds = []

dict_replace = {"o":"0", "O":"0", "D":"0", "a":"0", "e":"0", "I":"1", "l":"1", "i":"1", "z":"2", "Z":"2", "y":"4", "L+":"4", "L":"4", "h":"4", "H":"4", "u":"4", "V":"4",
                 "S":"5", "S":"5", "b":"6", "G":"6", "F":"7", "J":"7", "t":"7", "B":"8", "g":"9", ":":".", ",":".", "-":".", "..":"."}


def post_processing(text):
    # POSTPROCESSING
    # Replacing strings and sembols to similar numbers
    for k, v in dict_replace.items():
        text = text.replace(k, v)
    if len(text)>=3:
        if text[-3] not in ["."]:
            text = text[:-2] + "." + text[-2:] # insert "." to the position of -3

    if text[-1] != '0':
        text = text[:-1] + '0' # replace the last character with '0'

    if len(text) < 2:
        text = text + '.00' # if lenght is smaller than 2 add '.00' to the output

    if len(text) < 3:
        text = text + '.00' # if lenght is smaller than 3 add '.00' to the output
    
    if text[-2] == '.':
        text = text[:-2] + '0' + text[-1] # if text[-2] == '.' change its value to '0' (most possible value)

    if len(text) > 2:
        dot_index = text.find('.')
        if dot_index != len(text)-3 and dot_index != -1:
            text = text[:dot_index] + '0' + text[dot_index+1:]
    print("text: ", text)

    check_first = text.split('.')[0]
    check_last = text.split('.')[1]
    print("first: ", check_first)
    print("last: ", check_last)
    if len(check_first) > 3 and len(check_last) == 2:
        lenght = len(check_first)
        index = lenght - 3
        check_first = check_first[:-index]
        text = check_first + '.' + check_last

    return text


for filename in os.listdir(input_path):
    actual_label = filename.split("_")[0]
    actual_label = actual_label.replace(":", ".").replace(",", ".").replace("-", ".")
    print(f"Actual label is : {actual_label}")
    labels.append(actual_label)
    
    image_path = os.path.join(input_path, filename)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = np.array(img) # convert to numpy array
    result = ocr.ocr(img, cls=False) # cls makes rotations for prediction if set True

    # Draw bounding boxes and OCR result on the image
    image = Image.fromarray(img)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("./StyleText/fonts/en_standard.ttf", 30) # modify the font size here

    for res in result:
        if len(res) > 0:
            text = res[0][1][0] # get the OCR result text
            text = post_processing(text)
    
            print(f"prediction : {text}")
            text += ' (' + str(round(res[0][1][1]*100, 2)) + '%)'
            bbox = res[0][0]

            # Convert list of four points to tuple of two points
            bbox = ((bbox[0][0], bbox[0][1]), (bbox[2][0], bbox[2][1]))
            draw.rectangle(bbox, outline='red')
            draw.text((bbox[0][0]+60, bbox[0][1]), text, font=font, fill='red')


    preds.append(text.split()[0])

    # Save the output image with bounding boxes and OCR result
    out_path = os.path.join(output_path, f'{filename}_processed.jpg')
    image.save(out_path)

# calculate accuracy
correct_preds = sum([1 if str(label) == str(pred) else 0 for label, pred in zip(labels, preds)])
print(labels)
print(preds)
accuracy = correct_preds / len(labels) * 100
print(f"Accuracy: {accuracy}%")

char_accuracy = 0
for l, p in zip(labels, preds):
    corrects = 0
    for i in range(len(l)):
        try:
            if l[i] == p[i]:
                corrects += 1
        except:
            continue
    per_accuracy = corrects / len(l)
    char_accuracy += per_accuracy
char_accuracy = char_accuracy / len(labels)
print("Character Based Accuracy : ", char_accuracy)
