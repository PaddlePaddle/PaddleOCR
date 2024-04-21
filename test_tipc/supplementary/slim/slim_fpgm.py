import paddleslim
import paddle
import numpy as np

from paddleslim.dygraph import FPGMFilterPruner


def prune_model(model, input_shape, prune_ratio=0.1):
    flops = paddle.flops(model, input_shape)
    pruner = FPGMFilterPruner(model, input_shape)

    params_sensitive = {}
    for param in model.parameters():
        if "transpose" not in param.name and "linear" not in param.name:
            # set prune ratio as 10%. The larger the value, the more convolution weights will be cropped
            params_sensitive[param.name] = prune_ratio

    plan = pruner.prune_vars(params_sensitive, [0])

    flops = paddle.flops(model, input_shape)
    return model
