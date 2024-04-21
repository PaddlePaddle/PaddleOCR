import paddle
import paddle.nn.functional as F
from collections import OrderedDict


def create_metric(
    out,
    label,
    architecture=None,
    topk=5,
    classes_num=1000,
    use_distillation=False,
    mode="train",
):
    """
    Create measures of model accuracy, such as top1 and top5

    Args:
        out(variable): model output variable
        feeds(dict): dict of model input variables(included label)
        topk(int): usually top5
        classes_num(int): num of classes
        use_distillation(bool): whether to use distillation training
        mode(str): mode, train/valid

    Returns:
        fetchs(dict): dict of measures
    """
    # if architecture["name"] == "GoogLeNet":
    #     assert len(out) == 3, "GoogLeNet should have 3 outputs"
    #     out = out[0]
    # else:
    #     # just need student label to get metrics
    #     if use_distillation:
    #         out = out[1]
    softmax_out = F.softmax(out)

    fetchs = OrderedDict()
    # set top1 to fetchs
    top1 = paddle.metric.accuracy(softmax_out, label=label, k=1)
    # set topk to fetchs
    k = min(topk, classes_num)
    topk = paddle.metric.accuracy(softmax_out, label=label, k=k)

    # multi cards' eval
    if mode != "train" and paddle.distributed.get_world_size() > 1:
        top1 = (
            paddle.distributed.all_reduce(top1, op=paddle.distributed.ReduceOp.SUM)
            / paddle.distributed.get_world_size()
        )
        topk = (
            paddle.distributed.all_reduce(topk, op=paddle.distributed.ReduceOp.SUM)
            / paddle.distributed.get_world_size()
        )

    fetchs["top1"] = top1
    topk_name = "top{}".format(k)
    fetchs[topk_name] = topk

    return fetchs
