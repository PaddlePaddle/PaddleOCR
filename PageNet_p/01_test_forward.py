import torch
import paddle
import numpy as np
from reprod_log import ReprodDiffHelper
from reprod_log import ReprodLogger

from PageNet_paddle.model import build_model_paddle
from PageNet_pytorch.model import build_model_pytorch


def test_forward(cfg_paddle, cfg_pytorch):
    device = "gpu"  # you can also set it as "cpu"
    torch_device = torch.device("cuda:0" if device == "gpu" else "cpu")
    paddle.set_device(device)

    # load paddle model
    paddle_model = model = build_model_paddle(cfg_paddle)
    paddle_model.eval()
    paddle_state_dict = paddle.load("PageNet/PageNet_paddle/outputs/casia-hwdb/checkpoints/casia-hwdb-paddle.pdparams")
    paddle_model.set_dict(paddle_state_dict)

    # load torch model
    torch_model = build_model_pytorch(cfg_pytorch)
    torch_model.eval()
    torch_state_dict = torch.load("PageNet/PageNet_pytorch/outputs/casia-hwdb/checkpoints/casia-hwdb.pth")
    torch_model.load_state_dict(torch_state_dict)

    torch_model.to(torch_device)

    # load data
    inputs = np.load("PageNet/data/pagenet_fake_data.npy")

    # save the paddle output
    reprod_logger = ReprodLogger()
    paddle_out = paddle_model(paddle.to_tensor(inputs, dtype="float32"))
    i = 0
    for t in paddle_out:
        reprod_logger.add("logits"[i], t.cpu().detach().numpy())
        i=i+1
    reprod_logger.save("PageNet/result/forward_paddle.npy")
        

    # save the torch output
    torch_out = torch_model(
        torch.tensor(
            inputs, dtype=torch.float32).to(torch_device))
    #reprod_logger.add("logits", torch_out.cpu().detach().numpy())
    i=0
    for t in torch_out:
        reprod_logger.add("logits"[i], t.cpu().detach().numpy())
        i=i+1
    reprod_logger.save("PageNet/result/forward_ref.npy")


if __name__ == "__main__":
    import sys
    import os

    sys.path.append("..")
    sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])

    import yaml
    from PageNet_paddle.utils.parser import default_parser_paddle
    from PageNet_pytorch.utils.parser import default_parser_pytorch

    parser_paddle = default_parser_paddle()
    args_paddle = parser_paddle.parse_args()
    cfg_paddle = yaml.load(open(args_paddle.config, 'r'), Loader=yaml.FullLoader)  

    parser_pytorch = default_parser_pytorch()
    args_pytorch = parser_pytorch.parse_args()
    cfg_pytorch = yaml.load(open(args_pytorch.config, 'r'), Loader=yaml.FullLoader) 

    test_forward(cfg_paddle, cfg_pytorch)

    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("PageNet/result/forward_ref.npy")
    paddle_info = diff_helper.load_info("PageNet/result/forward_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="PageNet/result/log/forward_diff.log", diff_threshold=1e-5)
