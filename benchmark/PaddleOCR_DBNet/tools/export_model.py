import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import argparse

import paddle
from paddle.jit import to_static

from models import build_model
from utils import Config, ArgsParser


def init_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


def load_checkpoint(model, checkpoint_path):
    """
    load checkpoints
    :param checkpoint_path: Checkpoint path to be loaded
    """
    checkpoint = paddle.load(checkpoint_path)
    model.set_state_dict(checkpoint['state_dict'])
    print('load checkpoint from {}'.format(checkpoint_path))


def main(config):
    model = build_model(config['arch'])
    load_checkpoint(model, config['trainer']['resume_checkpoint'])
    model.eval()

    save_path = config["trainer"]["output_dir"]
    save_path = os.path.join(save_path, "inference")
    infer_shape = [3, -1, -1]
    model = to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + infer_shape, dtype="float32")
        ])

    paddle.jit.save(model, save_path)
    print("inference model is saved to {}".format(save_path))


if __name__ == "__main__":
    args = init_args()
    assert os.path.exists(args.config_file)
    config = Config(args.config_file)
    config.merge_dict(args.opt)
    main(config.cfg)
