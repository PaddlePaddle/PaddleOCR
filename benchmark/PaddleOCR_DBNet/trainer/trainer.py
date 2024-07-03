# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:58
# @Author  : zhoujun
import time

import paddle
from tqdm import tqdm

from base import BaseTrainer
from utils import runningScore, cal_text_score, Polynomial, profiler


class Trainer(BaseTrainer):
    def __init__(
        self,
        config,
        model,
        criterion,
        train_loader,
        validate_loader,
        metric_cls,
        post_process=None,
        profiler_options=None,
    ):
        super(Trainer, self).__init__(
            config,
            model,
            criterion,
            train_loader,
            validate_loader,
            metric_cls,
            post_process,
        )
        self.profiler_options = profiler_options
        self.enable_eval = config["trainer"].get("enable_eval", True)

    def _train_epoch(self, epoch):
        self.model.train()
        total_samples = 0
        train_reader_cost = 0.0
        train_batch_cost = 0.0
        reader_start = time.time()
        epoch_start = time.time()
        train_loss = 0.0
        running_metric_text = runningScore(2)

        for i, batch in enumerate(self.train_loader):
            profiler.add_profiler_step(self.profiler_options)
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.get_lr()

            cur_batch_size = batch["img"].shape[0]

            train_reader_cost += time.time() - reader_start
            if self.amp:
                with paddle.amp.auto_cast(
                    enable="gpu" in paddle.device.get_device(),
                    custom_white_list=self.amp.get("custom_white_list", []),
                    custom_black_list=self.amp.get("custom_black_list", []),
                    level=self.amp.get("level", "O2"),
                ):
                    preds = self.model(batch["img"])
                loss_dict = self.criterion(preds.astype(paddle.float32), batch)
                scaled_loss = self.amp["scaler"].scale(loss_dict["loss"])
                scaled_loss.backward()
                self.amp["scaler"].minimize(self.optimizer, scaled_loss)
            else:
                preds = self.model(batch["img"])
                loss_dict = self.criterion(preds, batch)
                # backward
                loss_dict["loss"].backward()
                self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.clear_grad()

            train_batch_time = time.time() - reader_start
            train_batch_cost += train_batch_time
            total_samples += cur_batch_size

            # acc iou
            score_shrink_map = cal_text_score(
                preds[:, 0, :, :],
                batch["shrink_map"],
                batch["shrink_mask"],
                running_metric_text,
                thred=self.config["post_processing"]["args"]["thresh"],
            )

            # loss 和 acc 记录到日志
            loss_str = "loss: {:.4f}, ".format(loss_dict["loss"].item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == "loss":
                    continue
                loss_str += "{}: {:.4f}".format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ", "

            train_loss += loss_dict["loss"]
            acc = score_shrink_map["Mean Acc"]
            iou_shrink_map = score_shrink_map["Mean IoU"]

            if self.global_step % self.log_iter == 0:
                self.logger_info(
                    "[{}/{}], [{}/{}], global_step: {}, ips: {:.1f} samples/sec, avg_reader_cost: {:.5f} s, avg_batch_cost: {:.5f} s, avg_samples: {}, acc: {:.4f}, iou_shrink_map: {:.4f}, {}lr:{:.6}, time:{:.2f}".format(
                        epoch,
                        self.epochs,
                        i + 1,
                        self.train_loader_len,
                        self.global_step,
                        total_samples / train_batch_cost,
                        train_reader_cost / self.log_iter,
                        train_batch_cost / self.log_iter,
                        total_samples / self.log_iter,
                        acc,
                        iou_shrink_map,
                        loss_str,
                        lr,
                        train_batch_cost,
                    )
                )
                total_samples = 0
                train_reader_cost = 0.0
                train_batch_cost = 0.0

            if self.visualdl_enable and paddle.distributed.get_rank() == 0:
                # write tensorboard
                for key, value in loss_dict.items():
                    self.writer.add_scalar(
                        "TRAIN/LOSS/{}".format(key), value, self.global_step
                    )
                self.writer.add_scalar("TRAIN/ACC_IOU/acc", acc, self.global_step)
                self.writer.add_scalar(
                    "TRAIN/ACC_IOU/iou_shrink_map", iou_shrink_map, self.global_step
                )
                self.writer.add_scalar("TRAIN/lr", lr, self.global_step)
            reader_start = time.time()
        return {
            "train_loss": train_loss / self.train_loader_len,
            "lr": lr,
            "time": time.time() - epoch_start,
            "epoch": epoch,
        }

    def _eval(self, epoch):
        self.model.eval()
        raw_metrics = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(
            enumerate(self.validate_loader),
            total=len(self.validate_loader),
            desc="test model",
        ):
            with paddle.no_grad():
                start = time.time()
                if self.amp:
                    with paddle.amp.auto_cast(
                        enable="gpu" in paddle.device.get_device(),
                        custom_white_list=self.amp.get("custom_white_list", []),
                        custom_black_list=self.amp.get("custom_black_list", []),
                        level=self.amp.get("level", "O2"),
                    ):
                        preds = self.model(batch["img"])
                    preds = preds.astype(paddle.float32)
                else:
                    preds = self.model(batch["img"])
                boxes, scores = self.post_process(
                    batch, preds, is_output_polygon=self.metric_cls.is_output_polygon
                )
                total_frame += batch["img"].shape[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, (boxes, scores))
                raw_metrics.append(raw_metric)
        metrics = self.metric_cls.gather_measure(raw_metrics)
        self.logger_info("FPS:{}".format(total_frame / total_time))
        return metrics["recall"].avg, metrics["precision"].avg, metrics["fmeasure"].avg

    def _on_epoch_finish(self):
        self.logger_info(
            "[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}".format(
                self.epoch_result["epoch"],
                self.epochs,
                self.epoch_result["train_loss"],
                self.epoch_result["time"],
                self.epoch_result["lr"],
            )
        )
        net_save_path = "{}/model_latest.pth".format(self.checkpoint_dir)
        net_save_path_best = "{}/model_best.pth".format(self.checkpoint_dir)

        if paddle.distributed.get_rank() == 0:
            self._save_checkpoint(self.epoch_result["epoch"], net_save_path)
            save_best = False
            if (
                self.validate_loader is not None
                and self.metric_cls is not None
                and self.enable_eval
            ):  # 使用f1作为最优模型指标
                recall, precision, hmean = self._eval(self.epoch_result["epoch"])

                if self.visualdl_enable:
                    self.writer.add_scalar("EVAL/recall", recall, self.global_step)
                    self.writer.add_scalar(
                        "EVAL/precision", precision, self.global_step
                    )
                    self.writer.add_scalar("EVAL/hmean", hmean, self.global_step)
                self.logger_info(
                    "test: recall: {:.6f}, precision: {:.6f}, hmean: {:.6f}".format(
                        recall, precision, hmean
                    )
                )

                if hmean >= self.metrics["hmean"]:
                    save_best = True
                    self.metrics["train_loss"] = self.epoch_result["train_loss"]
                    self.metrics["hmean"] = hmean
                    self.metrics["precision"] = precision
                    self.metrics["recall"] = recall
                    self.metrics["best_model_epoch"] = self.epoch_result["epoch"]
            else:
                if self.epoch_result["train_loss"] <= self.metrics["train_loss"]:
                    save_best = True
                    self.metrics["train_loss"] = self.epoch_result["train_loss"]
                    self.metrics["best_model_epoch"] = self.epoch_result["epoch"]
            best_str = "current best, "
            for k, v in self.metrics.items():
                best_str += "{}: {:.6f}, ".format(k, v)
            self.logger_info(best_str)
            if save_best:
                import shutil

                shutil.copy(net_save_path, net_save_path_best)
                self.logger_info("Saving current best: {}".format(net_save_path_best))
            else:
                self.logger_info("Saving checkpoint: {}".format(net_save_path))

    def _on_train_finish(self):
        if self.enable_eval:
            for k, v in self.metrics.items():
                self.logger_info("{}:{}".format(k, v))
        self.logger_info("finish train")

    def _initialize_scheduler(self):
        if self.config["lr_scheduler"]["type"] == "Polynomial":
            self.config["lr_scheduler"]["args"]["epochs"] = self.config["trainer"][
                "epochs"
            ]
            self.config["lr_scheduler"]["args"]["step_each_epoch"] = len(
                self.train_loader
            )
            self.lr_scheduler = Polynomial(**self.config["lr_scheduler"]["args"])()
        else:
            self.lr_scheduler = self._initialize("lr_scheduler", paddle.optimizer.lr)
