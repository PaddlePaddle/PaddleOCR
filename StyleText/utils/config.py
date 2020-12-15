# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import yaml
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter


def override(dl, ks, v):
    """
    Recursively replace dict of list

    Args:
        dl(dict or list): dict or list to be replaced
        ks(list): list of keys
        v(str): value to be replaced
    """

    def str2num(v):
        try:
            return eval(v)
        except Exception:
            return v

    assert isinstance(dl, (list, dict)), ("{} should be a list or a dict")
    assert len(ks) > 0, ('lenght of keys should larger than 0')
    if isinstance(dl, list):
        k = str2num(ks[0])
        if len(ks) == 1:
            assert k < len(dl), ('index({}) out of range({})'.format(k, dl))
            dl[k] = str2num(v)
        else:
            override(dl[k], ks[1:], v)
    else:
        if len(ks) == 1:
            #assert ks[0] in dl, ('{} is not exist in {}'.format(ks[0], dl))
            if not ks[0] in dl:
                logger.warning('A new filed ({}) detected!'.format(ks[0], dl))
            dl[ks[0]] = str2num(v)
        else:
            assert ks[0] in dl, (
                '({}) doesn\'t exist in {}, a new dict field is invalid'.
                format(ks[0], dl))
            override(dl[ks[0]], ks[1:], v)


def override_config(config, options=None):
    """
    Recursively override the config

    Args:
        config(dict): dict to be replaced
        options(list): list of pairs(key0.key1.idx.key2=value)
            such as: [
                'topk=2',
                'VALID.transforms.1.ResizeImage.resize_short=300'
            ]

    Returns:
        config(dict): replaced config
    """
    if options is not None:
        for opt in options:
            assert isinstance(opt, str), (
                "option({}) should be a str".format(opt))
            assert "=" in opt, (
                "option({}) should contain a ="
                "to distinguish between key and value".format(opt))
            pair = opt.split('=')
            assert len(pair) == 2, ("there can be only a = in the option")
            key, value = pair
            keys = key.split('.')
            override(config, keys, value)

    return config


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument(
            "-t", "--tag", default="0", help="tag for marking worker")
        self.add_argument(
            '-o',
            '--override',
            action='append',
            default=[],
            help='config options to be overridden')
        self.add_argument(
            "--style_image", default="examples/style_images/1.jpg", help="tag for marking worker")
        self.add_argument(
            "--text_corpus", default="PaddleOCR", help="tag for marking worker")
        self.add_argument(
            "--language", default="en", help="tag for marking worker")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, \
            "Please specify --config=configure_file_path."
        return args


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: config
    """
    ext = os.path.splitext(file_path)[1]
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    with open(file_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config


def gen_config():
    base_config = {
        "Global": {
            "algorithm": "SRNet",
            "use_gpu": True,
            "start_epoch": 1,
            "stage1_epoch_num": 100,
            "stage2_epoch_num": 100,
            "log_smooth_window": 20,
            "print_batch_step": 2,
            "save_model_dir": "./output/SRNet",
            "use_visualdl": False,
            "save_epoch_step": 10,
            "vgg_pretrain": "./pretrained/VGG19_pretrained",
            "vgg_load_static_pretrain": True
        },
        "Architecture": {
            "model_type": "data_aug",
            "algorithm": "SRNet",
            "net_g": {
                "name": "srnet_net_g",
                "encode_dim": 64,
                "norm": "batch",
                "use_dropout": False,
                "init_type": "xavier",
                "init_gain": 0.02,
                "use_dilation": 1
            },
            # input_nc, ndf, netD,
            # n_layers_D=3, norm='instance', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_id='cuda:0'
            "bg_discriminator": {
                "name": "srnet_bg_discriminator",
                "input_nc": 6,
                "ndf": 64,
                "netD": "basic",
                "norm": "none",
                "init_type": "xavier",
            },
            "fusion_discriminator": {
                "name": "srnet_fusion_discriminator",
                "input_nc": 6,
                "ndf": 64,
                "netD": "basic",
                "norm": "none",
                "init_type": "xavier",
            }
        },
        "Loss": {
            "lamb": 10,
            "perceptual_lamb": 1,
            "muvar_lamb": 50,
            "style_lamb": 500
        },
        "Optimizer": {
            "name": "Adam",
            "learning_rate": {
                "name": "lambda",
                "lr": 0.0002,
                "lr_decay_iters": 50
            },
            "beta1": 0.5,
            "beta2": 0.999,
        },
        "Train": {
            "batch_size_per_card": 8,
            "num_workers_per_card": 4,
            "dataset": {
                "delimiter": "\t",
                "data_dir": "/",
                "label_file": "tmp/label.txt",
                "transforms": [{
                    "DecodeImage": {
                        "to_rgb": True,
                        "to_np": False,
                        "channel_first": False
                    }
                }, {
                    "NormalizeImage": {
                        "scale": 1. / 255.,
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                        "order": None
                    }
                }, {
                    "ToCHWImage": None
                }]
            }
        }
    }
    with open("config.yml", "w") as f:
        yaml.dump(base_config, f)


if __name__ == '__main__':
    gen_config()
