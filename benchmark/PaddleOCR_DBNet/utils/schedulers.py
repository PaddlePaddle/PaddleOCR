from paddle.optimizer import lr
import logging

__all__ = ["Polynomial"]


class Polynomial(object):
    """
    Polynomial learning rate decay
    Args:
        learning_rate (float): The initial learning rate. It is a python float number.
        epochs(int): The decay epoch size. It determines the decay cycle, when by_epoch is set to true, it will change to epochs=epochs*step_each_epoch.
        step_each_epoch: all steps in each epoch.
        end_lr(float, optional): The minimum final learning rate. Default: 0.0001.
        power(float, optional): Power of polynomial. Default: 1.0.
        warmup_epoch(int): The epoch numbers for LinearWarmup. Default: 0, , when by_epoch is set to true, it will change to warmup_epoch=warmup_epoch*step_each_epoch.
        warmup_start_lr(float): Initial learning rate of warm up. Default: 0.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        by_epoch: Whether the set parameter is based on epoch or iter, when set to true,, epochs and warmup_epoch will be automatically multiplied by step_each_epoch. Default: True
    """

    def __init__(
        self,
        learning_rate,
        epochs,
        step_each_epoch,
        end_lr=0.0,
        power=1.0,
        warmup_epoch=0,
        warmup_start_lr=0.0,
        last_epoch=-1,
        by_epoch=True,
        **kwargs,
    ):
        super().__init__()
        if warmup_epoch >= epochs:
            msg = f'When using warm up, the value of "epochs" must be greater than value of "Optimizer.lr.warmup_epoch". The value of "Optimizer.lr.warmup_epoch" has been set to {epochs}.'
            logging.warning(msg)
            warmup_epoch = epochs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.end_lr = end_lr
        self.power = power
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.warmup_start_lr = warmup_start_lr

        if by_epoch:
            self.epochs *= step_each_epoch
            self.warmup_epoch = int(self.warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = (
            lr.PolynomialDecay(
                learning_rate=self.learning_rate,
                decay_steps=self.epochs,
                end_lr=self.end_lr,
                power=self.power,
                last_epoch=self.last_epoch,
            )
            if self.epochs > 0
            else self.learning_rate
        )
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch,
            )
        return learning_rate
