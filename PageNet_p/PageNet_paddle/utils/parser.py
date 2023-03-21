import argparse

def default_parser_paddle():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--config", type=str, default="PageNet/PageNet_paddle/configs/casia-hwdb.yaml")
    #parser.add_argument("--config", type=str, default="PageNet/PageNet_paddle/configs/scut-hccdoc.yaml")
    #parser.add_argument("--config", type=str, default="PageNet/PageNet_paddle/configs/mthv2.yaml")
    return parser