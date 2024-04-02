# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import argparse
import time
import paddle
from tqdm.auto import tqdm


class EVAL():
    def __init__(self, model_path, gpu_id=0):
        from models import build_model
        from data_loader import get_dataloader
        from post_processing import get_post_processing
        from utils import get_metric
        self.gpu_id = gpu_id
        if self.gpu_id is not None and isinstance(
                self.gpu_id, int) and paddle.device.is_compiled_with_cuda():
            paddle.device.set_device("gpu:{}".format(self.gpu_id))
        else:
            paddle.device.set_device("cpu")
        checkpoint = paddle.load(model_path)
        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False

        self.validate_loader = get_dataloader(config['dataset']['validate'],
                                              config['distributed'])

        self.model = build_model(config['arch'])
        self.model.set_state_dict(checkpoint['state_dict'])

        self.post_process = get_post_processing(config['post_processing'])
        self.metric_cls = get_metric(config['metric'])

    def eval(self):
        self.model.eval()
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(
                enumerate(self.validate_loader),
                total=len(self.validate_loader),
                desc='test model'):
            with paddle.no_grad():
                start = time.time()
                preds = self.model(batch['img'])
                boxes, scores = self.post_process(
                    batch,
                    preds,
                    is_output_polygon=self.metric_cls.is_output_polygon)
                total_frame += batch['img'].shape[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch,
                                                              (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        print('FPS:{}'.format(total_frame / total_time))
        return {
            'recall': metrics['recall'].avg,
            'precision': metrics['precision'].avg,
            'fmeasure': metrics['fmeasure'].avg
        }


def init_args():
    parser = argparse.ArgumentParser(description='DBNet.paddle')
    parser.add_argument(
        '--model_path',
        required=False,
        default='output/DBNet_resnet18_FPN_DBHead/checkpoint/1.pth',
        type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    eval = EVAL(args.model_path)
    result = eval.eval()
    print(result)
