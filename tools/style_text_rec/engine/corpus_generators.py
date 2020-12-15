import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from utils.logging import get_logger


class FileCorpus(object):
    def __init__(self, config):
        self.logger = get_logger()
        self.logger.info("using FileCorpus")

        self.char_list = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        corpus_file = config["CorpusGenerator"]["corpus_file"]
        self.language = config["CorpusGenerator"]["language"]
        with open(corpus_file, 'r') as f:
            corpus_raw = f.read()
        self.corpus_list = corpus_raw.split("\n")[:-1]
        assert len(self.corpus_list) > 0
        random.shuffle(self.corpus_list)
        self.index = 0

    def generate(self, corpus_length=0):
        if self.index >= len(self.corpus_list):
            self.index = 0
            random.shuffle(self.corpus_list)
        corpus = self.corpus_list[self.index]
        if corpus_length != 0:
            corpus = corpus[0:corpus_length]
        if corpus_length > len(corpus):
            self.logger.warning("generated corpus is shorter than expected.")
        self.index += 1
        return self.language, corpus


class EnNumCorpus(object):
    def __init__(self, config):
        self.logger = get_logger()
        self.logger.info("using NumberCorpus")
        self.num_list = "0123456789"
        self.en_char_list = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.height = config["Global"]["image_height"]
        self.max_width = config["Global"]["image_width"]

    def generate(self, corpus_length=0):
        corpus = ""
        if corpus_length == 0:
            corpus_length = random.randint(5, 15)
        for i in range(corpus_length):
            if random.random() < 0.2:
                corpus += "{}".format(random.choice(self.en_char_list))
            else:
                corpus += "{}".format(random.choice(self.num_list))
        return "en", corpus
