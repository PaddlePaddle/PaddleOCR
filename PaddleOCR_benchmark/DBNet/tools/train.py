import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import paddle
import paddle.distributed as dist
from utils import Config, ArgsParser


def init_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args


def main(config, profiler_options):
    from models import build_model, build_loss
    from data_loader import get_dataloader
    from trainer import Trainer
    from post_processing import get_post_processing
    from utils import get_metric
    if paddle.device.cuda.device_count() > 1:
        dist.init_parallel_env()
        config['distributed'] = True
    else:
        config['distributed'] = False
    train_loader = get_dataloader(config['dataset']['train'],
                                  config['distributed'])
    assert train_loader is not None
    if 'validate' in config['dataset']:
        validate_loader = get_dataloader(config['dataset']['validate'], False)
    else:
        validate_loader = None
    criterion = build_loss(config['loss'])
    config['arch']['backbone']['in_channels'] = 3 if config['dataset']['train'][
        'dataset']['args']['img_mode'] != 'GRAY' else 1
    model = build_model(config['arch'])
    # set @to_static for benchmark, skip this by default.
    post_p = get_post_processing(config['post_processing'])
    metric = get_metric(config['metric'])
    trainer = Trainer(
        config=config,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        post_process=post_p,
        metric_cls=metric,
        validate_loader=validate_loader,
        profiler_options=profiler_options)
    trainer.train()


if __name__ == '__main__':
    args = init_args()
    assert os.path.exists(args.config_file)
    config = Config(args.config_file)
    config.merge_dict(args.opt)
    main(config.cfg, args.profiler_options)
