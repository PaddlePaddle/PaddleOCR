import argparse

def default_parser_pytorch():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--config", type=str, default="PageNet/PageNet_pytorch/configs/casia-hwdb.yaml")
    #parser.add_argument("--config", type=str, default="PageNet/PageNet_pytorch/configs/scut-hccdoc.yaml")
    #parser.add_argument("--config", type=str, default="PageNet/PageNet_pytorch/configs/mthv2.yaml")
    return parser