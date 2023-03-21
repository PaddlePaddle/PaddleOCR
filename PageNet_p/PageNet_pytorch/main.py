import torch, gc

gc.collect()
torch.cuda.empty_cache()


import os
from model import build_model
from engine.val import validate
from utils.converter import Converter
from data import build_dataset, build_dataloader


def main(cfg):
    val_dataset = build_dataset(cfg, 'val')
    val_dataloader = build_dataloader(val_dataset, 'val', cfg)
    converter = Converter(cfg['DATA']['DICT'])

    model = build_model(cfg)
    model = model.cuda()  

    os.makedirs(cfg['OUTPUT_FOLDER'], exist_ok=True)

    validate(model, val_dataloader, converter, cfg)

if __name__ == '__main__':
    import yaml
    from utils.parser import default_parser

    parser = default_parser()
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)  

    main(cfg)