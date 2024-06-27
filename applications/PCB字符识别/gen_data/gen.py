# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This code is refer from:
https://github.com/zcswdt/Color_OCR_image_generator
"""
import os
import random
import PIL
from PIL import Image, ImageDraw, ImageFont
import json
import argparse


def get_char_lines(txt_root_path):
    """
    desc:get corpus line
    """
    txt_files = os.listdir(txt_root_path)
    char_lines = []
    for txt in txt_files:
        f = open(os.path.join(txt_root_path, txt), mode="r", encoding="utf-8")
        lines = f.readlines()
        f.close()
        for line in lines:
            char_lines.append(line.strip())
        return char_lines


def get_horizontal_text_picture(image_file, chars, fonts_list, cf):
    """
    desc:gen horizontal text picture
    """
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_w, img_h = img.size

    # random choice font
    font_path = random.choice(fonts_list)
    # random choice font size
    font_size = random.randint(cf.font_min_size, cf.font_max_size)
    font = ImageFont.truetype(font_path, font_size)

    ch_w = []
    ch_h = []
    for ch in chars:
        if int(PIL.__version__.split(".")[0]) < 10:
            wt, ht = font.getsize(ch)
        else:
            left, top, right, bottom = font.getbbox(ch)
            wt, ht = right - left, bottom - top
        ch_w.append(wt)
        ch_h.append(ht)
    f_w = sum(ch_w)
    f_h = max(ch_h)

    # add space
    char_space_width = max(ch_w)
    f_w += char_space_width * (len(chars) - 1)

    x1 = random.randint(0, img_w - f_w)
    y1 = random.randint(0, img_h - f_h)
    x2 = x1 + f_w
    y2 = y1 + f_h

    crop_y1 = y1
    crop_x1 = x1
    crop_y2 = y2
    crop_x2 = x2

    best_color = (0, 0, 0)
    draw = ImageDraw.Draw(img)
    for i, ch in enumerate(chars):
        draw.text((x1, y1), ch, best_color, font=font)
        x1 += ch_w[i] + char_space_width
    crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return crop_img, chars


def get_vertical_text_picture(image_file, chars, fonts_list, cf):
    """
    desc:gen vertical text picture
    """
    img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_w, img_h = img.size
    # random choice font
    font_path = random.choice(fonts_list)
    # random choice font size
    font_size = random.randint(cf.font_min_size, cf.font_max_size)
    font = ImageFont.truetype(font_path, font_size)

    ch_w = []
    ch_h = []
    for ch in chars:
        if int(PIL.__version__.split(".")[0]) < 10:
            wt, ht = font.getsize(ch)
        else:
            left, top, right, bottom = font.getbbox(ch)
            wt, ht = right - left, bottom - top
        ch_w.append(wt)
        ch_h.append(ht)
    f_w = max(ch_w)
    f_h = sum(ch_h)

    x1 = random.randint(0, img_w - f_w)
    y1 = random.randint(0, img_h - f_h)
    x2 = x1 + f_w
    y2 = y1 + f_h

    crop_y1 = y1
    crop_x1 = x1
    crop_y2 = y2
    crop_x2 = x2

    best_color = (0, 0, 0)
    draw = ImageDraw.Draw(img)
    i = 0
    for ch in chars:
        draw.text((x1, y1), ch, best_color, font=font)
        y1 = y1 + ch_h[i]
        i = i + 1
    crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    crop_img = crop_img.transpose(Image.ROTATE_90)
    return crop_img, chars


def get_fonts(fonts_path):
    """
    desc: get all fonts
    """
    font_files = os.listdir(fonts_path)
    fonts_list = []
    for font_file in font_files:
        font_path = os.path.join(fonts_path, font_file)
        fonts_list.append(font_path)
    return fonts_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_img", type=int, default=30, help="Number of images to generate"
    )
    parser.add_argument("--font_min_size", type=int, default=11)
    parser.add_argument(
        "--font_max_size",
        type=int,
        default=12,
        help="Help adjust the size of the generated text and the size of the picture",
    )
    parser.add_argument(
        "--bg_path",
        type=str,
        default="./background",
        help="The generated text pictures will be pasted onto the pictures of this folder",
    )
    parser.add_argument(
        "--det_bg_path",
        type=str,
        default="./det_background",
        help="The generated text pictures will use the pictures of this folder as the background",
    )
    parser.add_argument(
        "--fonts_path",
        type=str,
        default="../../StyleText/fonts",
        help="The font used to generate the picture",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="./corpus",
        help="The corpus used to generate the text picture",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/", help="Images save dir"
    )

    cf = parser.parse_args()
    # save path
    if not os.path.exists(cf.output_dir):
        os.mkdir(cf.output_dir)

    # get corpus
    txt_root_path = cf.corpus_path
    char_lines = get_char_lines(txt_root_path=txt_root_path)

    # get all fonts
    fonts_path = cf.fonts_path
    fonts_list = get_fonts(fonts_path)

    # rec bg
    img_root_path = cf.bg_path
    imnames = os.listdir(img_root_path)

    # det bg
    det_bg_path = cf.det_bg_path
    bg_pics = os.listdir(det_bg_path)

    # OCR det files
    det_val_file = open(cf.output_dir + "det_gt_val.txt", "w", encoding="utf-8")
    det_train_file = open(cf.output_dir + "det_gt_train.txt", "w", encoding="utf-8")
    # det imgs
    det_save_dir = "imgs/"
    if not os.path.exists(cf.output_dir + det_save_dir):
        os.mkdir(cf.output_dir + det_save_dir)
    det_val_save_dir = "imgs_val/"
    if not os.path.exists(cf.output_dir + det_val_save_dir):
        os.mkdir(cf.output_dir + det_val_save_dir)

    # OCR rec files
    rec_val_file = open(cf.output_dir + "rec_gt_val.txt", "w", encoding="utf-8")
    rec_train_file = open(cf.output_dir + "rec_gt_train.txt", "w", encoding="utf-8")
    # rec imgs
    rec_save_dir = "rec_imgs/"
    if not os.path.exists(cf.output_dir + rec_save_dir):
        os.mkdir(cf.output_dir + rec_save_dir)
    rec_val_save_dir = "rec_imgs_val/"
    if not os.path.exists(cf.output_dir + rec_val_save_dir):
        os.mkdir(cf.output_dir + rec_val_save_dir)

    val_ratio = cf.num_img * 0.2  # val dataset ratio

    print("start generating...")
    for i in range(0, cf.num_img):
        imname = random.choice(imnames)
        img_path = os.path.join(img_root_path, imname)

        rnd = random.random()
        # gen horizontal text picture
        if rnd < 0.5:
            gen_img, chars = get_horizontal_text_picture(
                img_path, char_lines[i], fonts_list, cf
            )
            ori_w, ori_h = gen_img.size
            gen_img = gen_img.crop((0, 3, ori_w, ori_h))
        # gen vertical text picture
        else:
            gen_img, chars = get_vertical_text_picture(
                img_path, char_lines[i], fonts_list, cf
            )
            ori_w, ori_h = gen_img.size
            gen_img = gen_img.crop((3, 0, ori_w, ori_h))

        ori_w, ori_h = gen_img.size

        # rec imgs
        save_img_name = str(i).zfill(4) + ".jpg"
        if i < val_ratio:
            save_dir = os.path.join(rec_val_save_dir, save_img_name)
            line = save_dir + "\t" + char_lines[i] + "\n"
            rec_val_file.write(line)
        else:
            save_dir = os.path.join(rec_save_dir, save_img_name)
            line = save_dir + "\t" + char_lines[i] + "\n"
            rec_train_file.write(line)
        gen_img.save(cf.output_dir + save_dir, quality=95, subsampling=0)

        # det img
        # random choice bg
        bg_pic = random.sample(bg_pics, 1)[0]
        det_img = Image.open(os.path.join(det_bg_path, bg_pic))
        # the PCB position is fixed, modify it according to your own scenario
        if bg_pic == "1.png":
            x1 = 38
            y1 = 3
        else:
            x1 = 34
            y1 = 1

        det_img.paste(gen_img, (x1, y1))
        # text pos
        chars_pos = [
            [x1, y1],
            [x1 + ori_w, y1],
            [x1 + ori_w, y1 + ori_h],
            [x1, y1 + ori_h],
        ]
        label = [{"transcription": char_lines[i], "points": chars_pos}]
        if i < val_ratio:
            save_dir = os.path.join(det_val_save_dir, save_img_name)
            det_val_file.write(
                save_dir + "\t" + json.dumps(label, ensure_ascii=False) + "\n"
            )
        else:
            save_dir = os.path.join(det_save_dir, save_img_name)
            det_train_file.write(
                save_dir + "\t" + json.dumps(label, ensure_ascii=False) + "\n"
            )
        det_img.save(cf.output_dir + save_dir, quality=95, subsampling=0)
