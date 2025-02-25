from paddle import nn


class IGTRLoss(nn.Layer):

    def __init__(self, **kwargs):
        super(IGTRLoss, self).__init__()

    def forward(self, predicts, batch):
        if isinstance(predicts, list):
            predicts = predicts[0]
        return predicts
