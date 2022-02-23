import collections
import os
import warnings

from paddle.io import Dataset
from paddle.dataset.common import md5file
from paddle.utils.download import get_path_from_url
from paddlenlp.utils.env import DATA_HOME
from . import DatasetBuilder

__all__ = ['WMT14ende']


class WMT14ende(DatasetBuilder):
    '''
    This dataset is a translation dataset for machine translation task. More 
    specifically, this dataset is a WMT14 English to German translation dataset 
    which uses commoncrawl, europarl and news-commentary as train dataset and 
    uses newstest2014 as test dataset.
    '''
    URL = "https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.tar.gz"
    META_INFO = collections.namedtuple('META_INFO', ('src_file', 'tgt_file',
                                                     'src_md5', 'tgt_md5'))
    SPLITS = {
        'train': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "train.tok.clean.bpe.33708.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "train.tok.clean.bpe.33708.de"),
            "c7c0b77e672fc69f20be182ae37ff62c",
            "1865ece46948fda1209d3b7794770a0a"),
        'dev': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2013.tok.bpe.33708.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2013.tok.bpe.33708.de"),
            "aa4228a4bedb6c45d67525fbfbcee75e",
            "9b1eeaff43a6d5e78a381a9b03170501"),
        'test': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2014.tok.bpe.33708.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                         "newstest2014.tok.bpe.33708.de"),
            "c9403eacf623c6e2d9e5a1155bdff0b5",
            "0058855b55e37c4acfcb8cffecba1050"),
        'dev-eval': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2013.tok.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2013.tok.de"),
            "d74712eb35578aec022265c439831b0e",
            "6ff76ced35b70e63a61ecec77a1c418f"),
        'test-eval': META_INFO(
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2014.tok.en"),
            os.path.join("WMT14.en-de", "wmt14_ende_data",
                         "newstest2014.tok.de"),
            "8cce2028e4ca3d4cc039dfd33adbfb43",
            "a1b1f4c47f487253e1ac88947b68b3b8")
    }
    VOCAB_INFO = [(os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                                "vocab_all.bpe.33708"),
                   "2fc775b7df37368e936a8e1f63846bb0"),
                  (os.path.join("WMT14.en-de", "wmt14_ende_data_bpe",
                                "vocab_all.bpe.33712"),
                   "de485e3c2e17e23acf4b4b70b54682dd")]
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "<e>"

    MD5 = "a2b8410709ff760a3b40b84bd62dfbd8"

    def _get_data(self, mode, **kwargs):
        default_root = os.path.join(DATA_HOME, self.__class__.__name__)
        src_filename, tgt_filename, src_data_hash, tgt_data_hash = self.SPLITS[
            mode]
        src_fullname = os.path.join(default_root, src_filename)
        tgt_fullname = os.path.join(default_root, tgt_filename)

        (bpe_vocab_filename, bpe_vocab_hash), (sub_vocab_filename,
                                               sub_vocab_hash) = self.VOCAB_INFO
        bpe_vocab_fullname = os.path.join(default_root, bpe_vocab_filename)
        sub_vocab_fullname = os.path.join(default_root, sub_vocab_filename)

        if (not os.path.exists(src_fullname) or
            (src_data_hash and not md5file(src_fullname) == src_data_hash)) or (
                not os.path.exists(tgt_fullname) or
                (tgt_data_hash and
                 not md5file(tgt_fullname) == tgt_data_hash)) or (
                     not os.path.exists(bpe_vocab_fullname) or
                     (bpe_vocab_hash and
                      not md5file(bpe_vocab_fullname) == bpe_vocab_hash)) or (
                          not os.path.exists(sub_vocab_fullname) or
                          (sub_vocab_hash and
                           not md5file(sub_vocab_fullname) == sub_vocab_hash)):
            get_path_from_url(self.URL, default_root, self.MD5)

        return src_fullname, tgt_fullname

    def _read(self, filename, *args):
        src_filename, tgt_filename = filename
        with open(src_filename, 'r', encoding='utf-8') as src_f:
            with open(tgt_filename, 'r', encoding='utf-8') as tgt_f:
                for src_line, tgt_line in zip(src_f, tgt_f):
                    src_line = src_line.strip()
                    tgt_line = tgt_line.strip()
                    if not src_line and not tgt_line:
                        continue
                    yield {"en": src_line, "de": tgt_line}

    def get_vocab(self):
        bpe_vocab_fullname = os.path.join(DATA_HOME, self.__class__.__name__,
                                          self.VOCAB_INFO[0][0])
        sub_vocab_fullname = os.path.join(DATA_HOME, self.__class__.__name__,
                                          self.VOCAB_INFO[1][0])
        vocab_info = {
            'bpe': {
                'filepath': bpe_vocab_fullname,
                'unk_token': self.UNK_TOKEN,
                'bos_token': self.BOS_TOKEN,
                'eos_token': self.EOS_TOKEN
            },
            'benchmark': {
                'filepath': sub_vocab_fullname,
                'unk_token': self.UNK_TOKEN,
                'bos_token': self.BOS_TOKEN,
                'eos_token': self.EOS_TOKEN
            }
        }
        return vocab_info
