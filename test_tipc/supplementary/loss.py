import paddle
import paddle.nn.functional as F


class Loss(object):
    """
    Loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        assert class_dim > 1, "class_dim=%d is not larger than 1" % (class_dim)
        self._class_dim = class_dim
        if epsilon is not None and epsilon >= 0.0 and epsilon <= 1.0:
            self._epsilon = epsilon
            self._label_smoothing = True
        else:
            self._epsilon = None
            self._label_smoothing = False

    def _labelsmoothing(self, target):
        if target.shape[-1] != self._class_dim:
            one_hot_target = F.one_hot(target, self._class_dim)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self._epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, self._class_dim])
        return soft_target

    def _crossentropy(self, input, target, use_pure_fp16=False):
        if self._label_smoothing:
            target = self._labelsmoothing(target)
            input = -F.log_softmax(input, axis=-1)
            cost = paddle.sum(target * input, axis=-1)
        else:
            cost = F.cross_entropy(input=input, label=target)
        if use_pure_fp16:
            avg_cost = paddle.sum(cost)
        else:
            avg_cost = paddle.mean(cost)
        return avg_cost

    def __call__(self, input, target):
        return self._crossentropy(input, target)


def build_loss(config, epsilon=None):
    class_dim = config['class_dim']
    loss_func = Loss(class_dim=class_dim, epsilon=epsilon)
    return loss_func


class LossDistill(Loss):
    def __init__(self, model_name_list, class_dim=1000, epsilon=None):
        assert class_dim > 1, "class_dim=%d is not larger than 1" % (class_dim)
        self._class_dim = class_dim
        if epsilon is not None and epsilon >= 0.0 and epsilon <= 1.0:
            self._epsilon = epsilon
            self._label_smoothing = True
        else:
            self._epsilon = None
            self._label_smoothing = False

        self.model_name_list = model_name_list
        assert len(self.model_name_list) > 1, "error"

    def __call__(self, input, target):
        losses = {}
        for k in self.model_name_list:
            inp = input[k]
            losses[k] = self._crossentropy(inp, target)
        return losses


class KLJSLoss(object):
    def __init__(self, mode='kl'):
        assert mode in ['kl', 'js', 'KL', 'JS'
                        ], "mode can only be one of ['kl', 'js', 'KL', 'JS']"
        self.mode = mode

    def __call__(self, p1, p2, reduction="mean"):
        p1 = F.softmax(p1, axis=-1)
        p2 = F.softmax(p2, axis=-1)

        loss = paddle.multiply(p2, paddle.log((p2 + 1e-5) / (p1 + 1e-5) + 1e-5))

        if self.mode.lower() == "js":
            loss += paddle.multiply(
                p1, paddle.log((p1 + 1e-5) / (p2 + 1e-5) + 1e-5))
            loss *= 0.5
        if reduction == "mean":
            loss = paddle.mean(loss)
        elif reduction == "none" or reduction is None:
            return loss
        else:
            loss = paddle.sum(loss)
        return loss


class DMLLoss(object):
    def __init__(self, model_name_pairs, mode='js'):

        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.kljs_loss = KLJSLoss(mode=mode)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(
                model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def __call__(self, predicts, target=None):
        loss_dict = dict()
        for pairs in self.model_name_pairs:
            p1 = predicts[pairs[0]]
            p2 = predicts[pairs[1]]

            loss_dict[pairs[0] + "_" + pairs[1]] = self.kljs_loss(p1, p2)

        return loss_dict


# def build_distill_loss(config, epsilon=None):
#     class_dim = config['class_dim']
#     loss = LossDistill(model_name_list=['student', 'student1'], )
#     return loss_func
