import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils.logging import get_logger


class StdTextDrawer(object):
    def __init__(self, config):
        self.logger = get_logger()
        self.max_width = config["Global"]["image_width"]
        self.char_list = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.height = config["Global"]["image_height"]
        self.font_dict = {}
        self.load_fonts(config["TextDrawer"]["fonts"])
        self.support_languages = list(self.font_dict)

    def load_fonts(self, fonts_config):
        for language in fonts_config:
            font_path = fonts_config[language]
            font_height = self.get_valid_height(font_path)
            font = ImageFont.truetype(font_path, font_height)
            self.font_dict[language] = font

    def get_valid_height(self, font_path):
        font = ImageFont.truetype(font_path, self.height - 4)
        _, font_height = font.getsize(self.char_list)
        if font_height <= self.height - 4:
            return self.height - 4
        else:
            return int((self.height - 4)**2 / font_height)

    def draw_text(self, corpus, language="en", crop=True):
        if language not in self.support_languages:
            self.logger.warning(
                "language {} not supported, use en instead.".format(language))
            language = "en"
        if crop:
            width = min(self.max_width, len(corpus) * self.height) + 4
        else:
            width = len(corpus) * self.height + 4
        bg = Image.new("RGB", (width, self.height), color=(127, 127, 127))
        draw = ImageDraw.Draw(bg)

        char_x = 2
        font = self.font_dict[language]
        for i, char_i in enumerate(corpus):
            char_size = font.getsize(char_i)[0]
            draw.text((char_x, 2), char_i, fill=(0, 0, 0), font=font)
            char_x += char_size
            if char_x >= width:
                corpus = corpus[0:i + 1]
                self.logger.warning("corpus length exceed limit: {}".format(
                    corpus))
                break

        text_input = np.array(bg).astype(np.uint8)
        text_input = text_input[:, 0:char_x, :]
        return corpus, text_input
