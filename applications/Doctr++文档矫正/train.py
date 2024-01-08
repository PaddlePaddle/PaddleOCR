# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import inspect
import os
import random
from pathlib import Path
from typing import Optional
import cv2
# cv2.setNumThreads(30)
import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from loguru import logger
from paddle.io import DataLoader
from paddle.nn import functional as F
from paddle_msssim import ms_ssim, ssim

from doc3d_dataset import Doc3dDataset
from GeoTr_PP import GeoTr
from utils import to_image
import paddle.distributed as dist

# 导入分布式专用 Fleet API
from paddle.distributed import fleet
# 导入分布式训练数据所需 API
from paddle.io import DataLoader, DistributedBatchSampler

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
RANK = int(os.getenv("RANK", -1))
logger.add("training2.log")

# strategy = fleet.DistributedStrategy()
# fleet.init(is_collective=True, strategy=strategy)


def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    if deterministic:
        os.environ["FLAGS_cudnn_deterministic"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code,
    # i.e.  colorstr('blue', 'hello world')

    *args, string = (input if len(input) > 1 else
                     ("blue", "bold", input[0]))  # color arguments, string

    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }

    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]


def print_args(args: Optional[dict]=None, show_file=True, show_func=False):
    # Print function arguments (optional args dict)
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    logger.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    # Increment file or directory path,
    # i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = ((path.with_suffix(""), path.suffix)
                        if path.is_file() else (path, ""))

        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def train(args):
    # dist.init_parallel_env()
    save_dir = Path(args.save_dir)

    use_vdl = args.use_vdl
    if use_vdl:
        from visualdl import LogWriter

        log_dir = save_dir / "vdl"
        vdl_writer = LogWriter(str(log_dir))

    # Directories
    weights_dir = save_dir / "weights"
    weights_dir.parent.mkdir(parents=True, exist_ok=True)

    last = weights_dir / "last.ckpt"
    best = weights_dir / "best.ckpt"

    # Hyperparameters

    # Config
    init_seeds(args.seed)

    # Train loader
    train_dataset = Doc3dDataset(
        args.data_root,
        split="train",
        is_augment=True,
        image_size=args.img_size, )
    # train_sampler = DistributedBatchSampler(train_dataset, args.batch_size, shuffle=True, drop_last=True)

    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_sampler=train_sampler,
    #     num_workers=args.workers,
    # )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers, )

    # Validation loader
    val_dataset = Doc3dDataset(
        args.data_root,
        split="val",
        is_augment=False,
        image_size=args.img_size, )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers, )

    # Model
    model = GeoTr()
    # model =fleet.distributed_model(model)

    if use_vdl:
        vdl_writer.add_graph(
            model,
            input_spec=[
                paddle.static.InputSpec([1, 3, args.img_size, args.img_size],
                                        "float32")
            ], )

    # Scheduler
    scheduler = optim.lr.OneCycleLR(
        max_learning_rate=args.lr,
        total_steps=args.epochs * len(train_loader),
        phase_pct=0.1,
        end_learning_rate=args.lr / 2.5e5, )

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=scheduler,
        parameters=model.parameters(), )
    # optimizer=fleet.distributed_optimizer(optimizer)

    # loss function
    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if args.resume:
        ckpt = paddle.load(args.resume)
        model.set_state_dict(ckpt["model"])
        optimizer.set_state_dict(ckpt["optimizer"])
        scheduler.set_state_dict(ckpt["scheduler"])
        best_fitness = ckpt["best_fitness"]
        start_epoch = ckpt["epoch"] + 1

    # Train
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):

            pred = model(img)  # NCHW
            pred_nhwc = pred.transpose([0, 2, 3, 1])

            loss = l1_loss_fn(pred_nhwc, target)
            mse_loss = mse_loss_fn(pred_nhwc, target)

            if use_vdl:
                vdl_writer.add_scalar("Train/L1 Loss",
                                      float(loss),
                                      epoch * len(train_loader) + i)
                vdl_writer.add_scalar("Train/MSE Loss",
                                      float(mse_loss),
                                      epoch * len(train_loader) + i)
                vdl_writer.add_scalar(
                    "Train/Learning Rate",
                    float(scheduler.get_lr()),
                    epoch * len(train_loader) + i, )

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.clear_grad()

            if i % 10 == 0:
                logger.info(
                    f"[TRAIN MODE] Epoch: {epoch}, Iter: {i}, L1 Loss: {float(loss)}, "
                    f"MSE Loss: {float(mse_loss)}, LR: {float(scheduler.get_lr())}"
                )

        # Validation
        model.eval()

        with paddle.no_grad():
            avg_ssim = paddle.zeros([])
            avg_ms_ssim = paddle.zeros([])
            avg_l1_loss = paddle.zeros([])
            avg_mse_loss = paddle.zeros([])

            for i, (img, target) in enumerate(val_loader):

                pred = model(img)
                pred_nhwc = pred.transpose([0, 2, 3, 1])

                # predict image
                out = F.grid_sample(img, pred_nhwc)
                out_gt = F.grid_sample(img, target)

                # calculate ssim
                ssim_val = ssim(out, out_gt, data_range=1.0)
                ms_ssim_val = ms_ssim(out, out_gt, data_range=1.0)

                loss = l1_loss_fn(pred_nhwc, target)
                mse_loss = mse_loss_fn(pred_nhwc, target)

                # calculate fitness
                avg_ssim += ssim_val
                avg_ms_ssim += ms_ssim_val
                avg_l1_loss += loss
                avg_mse_loss += mse_loss

                if i % 10 == 0:
                    logger.info(
                        f"[VAL MODE] Epoch: {epoch}, VAL Iter: {i}, "
                        f"L1 Loss: {float(loss)} MSE Loss: {float(mse_loss)}, "
                        f"MS-SSIM: {float(ms_ssim_val)}, SSIM: {float(ssim_val)}"
                    )

                if use_vdl and i == 0:
                    img_0 = to_image(out[0])
                    img_gt_0 = to_image(out_gt[0])
                    vdl_writer.add_image("Val/Predicted Image No.0", img_0,
                                         epoch)
                    vdl_writer.add_image("Val/Target Image No.0", img_gt_0,
                                         epoch)

                    img_1 = to_image(out[1])
                    img_gt_1 = to_image(out_gt[1])
                    img_gt_1 = img_gt_1.astype("uint8")
                    vdl_writer.add_image("Val/Predicted Image No.1", img_1,
                                         epoch)
                    vdl_writer.add_image("Val/Target Image No.1", img_gt_1,
                                         epoch)

                    img_2 = to_image(out[2])
                    img_gt_2 = to_image(out_gt[2])
                    vdl_writer.add_image("Val/Predicted Image No.2", img_2,
                                         epoch)
                    vdl_writer.add_image("Val/Target Image No.2", img_gt_2,
                                         epoch)

            avg_ssim /= len(val_loader)
            avg_ms_ssim /= len(val_loader)
            avg_l1_loss /= len(val_loader)
            avg_mse_loss /= len(val_loader)
            logger.info("val_loss_show:"
                        f"[VAL MODE] avg_l1_loss: {float(avg_l1_loss)}, "
                        f"avg_mse_loss: {float(avg_mse_loss)}, "
                        f"avg_ssim: {float(avg_ssim)}, "
                        f"avg_ms_ssim: {float(avg_ms_ssim)}")

            if use_vdl:
                vdl_writer.add_scalar("Val/L1 Loss", float(loss), epoch)
                vdl_writer.add_scalar("Val/MSE Loss", float(mse_loss), epoch)
                vdl_writer.add_scalar("Val/SSIM", float(ssim_val), epoch)
                vdl_writer.add_scalar("Val/MS-SSIM", float(ms_ssim_val), epoch)

# Save
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_fitness": best_fitness,
            "epoch": epoch,
        }

        paddle.save(ckpt, str(last))

        if best_fitness < avg_ms_ssim:
            best_fitness = avg_ms_ssim
            paddle.save(ckpt, str(best))

    if use_vdl:
        vdl_writer.close()


def main(args):
    print_args(vars(args))

    args.save_dir = str(
        increment_path(
            Path(args.project) / args.name, exist_ok=args.exist_ok))

    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--data-root",
        nargs="?",
        type=str,
        default="~/datasets/doc3d",
        help="The root path of the dataset", )
    parser.add_argument(
        "--img-size",
        nargs="?",
        type=int,
        default=288,
        help="The size of the input image", )
    parser.add_argument(
        "--epochs",
        nargs="?",
        type=int,
        default=65,
        help="The number of training epochs", )
    parser.add_argument(
        "--batch-size", nargs="?", type=int, default=12, help="Batch Size")
    parser.add_argument(
        "--lr", nargs="?", type=float, default=1e-04, help="Learning Rate")
    parser.add_argument(
        "--resume",
        nargs="?",
        type=str,
        default=None,
        help="Path to previous saved model to restart from", )
    parser.add_argument(
        "--workers", type=int, default=4, help="max dataloader workers")
    parser.add_argument(
        "--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment", )
    parser.add_argument(
        "--seed", type=int, default=0, help="Global training seed")
    parser.add_argument(
        "--use-vdl", action="store_true", help="use VisualDL as logger")
    args = parser.parse_args()
    main(args)
