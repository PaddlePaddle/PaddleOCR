# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:50
# @Author  : zhoujun

import os
import pathlib
import shutil
from pprint import pformat

import anyconfig
import paddle
import numpy as np
import random
from paddle.jit import to_static
from paddle.static import InputSpec

from utils import setup_logger


class BaseTrainer:
    def __init__(
        self,
        config,
        model,
        criterion,
        train_loader,
        validate_loader,
        metric_cls,
        post_process=None,
    ):
        config["trainer"]["output_dir"] = os.path.join(
            str(pathlib.Path(os.path.abspath(__name__)).parent),
            config["trainer"]["output_dir"],
        )
        config["name"] = config["name"] + "_" + model.name
        self.save_dir = config["trainer"]["output_dir"]
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoint")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.global_step = 0
        self.start_epoch = 0
        self.config = config
        self.criterion = criterion
        # logger and tensorboard
        self.visualdl_enable = self.config["trainer"].get("visual_dl", False)
        self.epochs = self.config["trainer"]["epochs"]
        self.log_iter = self.config["trainer"]["log_iter"]
        if paddle.distributed.get_rank() == 0:
            anyconfig.dump(config, os.path.join(self.save_dir, "config.yaml"))
            self.logger = setup_logger(os.path.join(self.save_dir, "train.log"))
            self.logger_info(pformat(self.config))

        self.model = self.apply_to_static(model)

        # device
        if (
            paddle.device.cuda.device_count() > 0
            and paddle.device.is_compiled_with_cuda()
        ):
            self.with_cuda = True
            random.seed(self.config["trainer"]["seed"])
            np.random.seed(self.config["trainer"]["seed"])
            paddle.seed(self.config["trainer"]["seed"])
        else:
            self.with_cuda = False
        self.logger_info("train with and paddle {}".format(paddle.__version__))
        # metrics
        self.metrics = {
            "recall": 0,
            "precision": 0,
            "hmean": 0,
            "train_loss": float("inf"),
            "best_model_epoch": 0,
        }

        self.train_loader = train_loader
        if validate_loader is not None:
            assert post_process is not None and metric_cls is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls
        self.train_loader_len = len(train_loader)

        if self.validate_loader is not None:
            self.logger_info(
                "train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader".format(
                    len(self.train_loader.dataset),
                    self.train_loader_len,
                    len(self.validate_loader.dataset),
                    len(self.validate_loader),
                )
            )
        else:
            self.logger_info(
                "train dataset has {} samples,{} in dataloader".format(
                    len(self.train_loader.dataset), self.train_loader_len
                )
            )

        self._initialize_scheduler()

        self._initialize_optimizer()

        # resume or finetune
        if self.config["trainer"]["resume_checkpoint"] != "":
            self._load_checkpoint(
                self.config["trainer"]["resume_checkpoint"], resume=True
            )
        elif self.config["trainer"]["finetune_checkpoint"] != "":
            self._load_checkpoint(
                self.config["trainer"]["finetune_checkpoint"], resume=False
            )

        if self.visualdl_enable and paddle.distributed.get_rank() == 0:
            from visualdl import LogWriter

            self.writer = LogWriter(self.save_dir)

        # 混合精度训练
        self.amp = self.config.get("amp", None)
        if self.amp == "None":
            self.amp = None
        if self.amp:
            self.amp["scaler"] = paddle.amp.GradScaler(
                init_loss_scaling=self.amp.get("scale_loss", 1024),
                use_dynamic_loss_scaling=self.amp.get("use_dynamic_loss_scaling", True),
            )
            self.model, self.optimizer = paddle.amp.decorate(
                models=self.model,
                optimizers=self.optimizer,
                level=self.amp.get("amp_level", "O2"),
            )

        # 分布式训练
        if paddle.device.cuda.device_count() > 1:
            self.model = paddle.DataParallel(self.model)
        # make inverse Normalize
        self.UN_Normalize = False
        for t in self.config["dataset"]["train"]["dataset"]["args"]["transforms"]:
            if t["type"] == "Normalize":
                self.normalize_mean = t["args"]["mean"]
                self.normalize_std = t["args"]["std"]
                self.UN_Normalize = True

    def apply_to_static(self, model):
        support_to_static = self.config["trainer"].get("to_static", False)
        if support_to_static:
            specs = None
            print("static")
            specs = [InputSpec([None, 3, -1, -1])]
            model = to_static(model, input_spec=specs)
            self.logger_info(
                "Successfully to apply @to_static with specs: {}".format(specs)
            )
        return model

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch + 1, self.epochs + 1):
            self.epoch_result = self._train_epoch(epoch)
            self._on_epoch_finish()
        if paddle.distributed.get_rank() == 0 and self.visualdl_enable:
            self.writer.close()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _eval(self, epoch):
        """
        eval logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _on_epoch_finish(self):
        raise NotImplementedError

    def _on_train_finish(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, file_name):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state_dict = self.model.state_dict()
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "state_dict": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
            "metrics": self.metrics,
        }
        filename = os.path.join(self.checkpoint_dir, file_name)
        paddle.save(state, filename)

    def _load_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger_info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = paddle.load(checkpoint_path)
        self.model.set_state_dict(checkpoint["state_dict"])
        if resume:
            self.global_step = checkpoint["global_step"]
            self.start_epoch = checkpoint["epoch"]
            self.config["lr_scheduler"]["args"]["last_epoch"] = self.start_epoch
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.set_state_dict(checkpoint["optimizer"])
            if "metrics" in checkpoint:
                self.metrics = checkpoint["metrics"]
            self.logger_info(
                "resume from checkpoint {} (epoch {})".format(
                    checkpoint_path, self.start_epoch
                )
            )
        else:
            self.logger_info("finetune from checkpoint {}".format(checkpoint_path))

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]["type"]
        module_args = self.config[name].get("args", {})
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def _initialize_scheduler(self):
        self.lr_scheduler = self._initialize("lr_scheduler", paddle.optimizer.lr)

    def _initialize_optimizer(self):
        self.optimizer = self._initialize(
            "optimizer",
            paddle.optimizer,
            parameters=self.model.parameters(),
            learning_rate=self.lr_scheduler,
        )

    def inverse_normalize(self, batch_img):
        if self.UN_Normalize:
            batch_img[:, 0, :, :] = (
                batch_img[:, 0, :, :] * self.normalize_std[0] + self.normalize_mean[0]
            )
            batch_img[:, 1, :, :] = (
                batch_img[:, 1, :, :] * self.normalize_std[1] + self.normalize_mean[1]
            )
            batch_img[:, 2, :, :] = (
                batch_img[:, 2, :, :] * self.normalize_std[2] + self.normalize_mean[2]
            )

    def logger_info(self, s):
        if paddle.distributed.get_rank() == 0:
            self.logger.info(s)
