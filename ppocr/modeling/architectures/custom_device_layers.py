import paddle
import paddle.nn as nn
import paddle.distributed as dist

__all__ = ["NaiveSyncBatchNorm"]


class _AllReduce(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, input):
        input_list = [paddle.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, sync_op=True)
        inputs = paddle.stack(input_list, axis=0)
        return paddle.sum(inputs, axis=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, sync_op=True)
        return grad_output


def differentiable_all_reduce(input):
    """
    Differentiable counterpart of `dist.all_reduce`.
    """
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return input
    return _AllReduce.apply(input)


class NaiveSyncBatchNorm(nn.BatchNorm2D):

    def __init__(self, *args, stats_mode="", **kwargs):
        super().__init__(*args, **kwargs)
        assert stats_mode in ["", "N"]
        self._stats_mode = stats_mode

    def forward(self, input):
        if dist.get_world_size() == 1 or not self.training:
            return super().forward(input)

        B, C = input.shape[0], input.shape[1]

        mean = paddle.mean(input, axis=[0, 2, 3])
        meansqr = paddle.mean(input * input, axis=[0, 2, 3])

        if self._stats_mode == "":
            assert (
                B > 0
            ), 'SyncBatchNorm(stats_mode="") does not support zero batch size.'
            vec = paddle.concat([mean, meansqr], axis=0)
            vec = differentiable_all_reduce(vec) * (1.0 / dist.get_world_size())
            mean, meansqr = paddle.split(vec, [C, C])
            momentum = (
                1 - self._momentum
            )  # NOTE: paddle has reverse momentum defination
        else:
            if B == 0:
                vec = paddle.zeros([2 * C + 1], dtype=mean.dtype)
                vec = vec + input.sum()  # make sure there is gradient w.r.t input
            else:
                vec = paddle.concat(
                    [
                        mean,
                        meansqr,
                        paddle.ones([1], dtype=mean.dtype),
                    ],
                    axis=0,
                )
            vec = differentiable_all_reduce(vec * B)

            total_batch = vec[-1].detach()
            momentum = total_batch.clip(max=1) * (
                1 - self._momentum
            )  # no update if total_batch is 0
            mean, meansqr, _ = paddle.split(
                vec / total_batch.clip(min=1), [C, C, int(vec.shape[0] - 2 * C)]
            )  # avoid div-by-zero

        var = meansqr - mean * mean
        invstd = paddle.rsqrt(var + self._epsilon)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape([1, -1, 1, 1])
        bias = bias.reshape([1, -1, 1, 1])

        tmp_mean = self._mean + momentum * (mean.detach() - self._mean)
        self._mean.set_value(tmp_mean)
        tmp_variance = self._variance + (momentum * (var.detach() - self._variance))
        self._variance.set_value(tmp_variance)
        ret = input * scale + bias
        return ret

    @classmethod
    def convert_sync_batchnorm(cls, layer):
        layer_output = layer
        if isinstance(layer, nn.BatchNorm2D):

            layer_output = NaiveSyncBatchNorm(
                layer._num_features,
                layer._momentum,
                layer._epsilon,
                layer._weight_attr,
                layer._bias_attr,
                layer._data_format,
                layer._name,
            )

            if layer._weight_attr is not False and layer._bias_attr is not False:
                with paddle.no_grad():
                    layer_output.weight = layer.weight
                    layer_output.bias = layer.bias
            layer_output._mean = layer._mean
            layer_output._variance = layer._variance

        for name, sublayer in layer.named_children():
            layer_output.add_sublayer(name, cls.convert_sync_batchnorm(sublayer))
        del layer
        return layer_output
