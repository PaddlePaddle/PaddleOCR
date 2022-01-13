import numpy as np
from paddle.vision.datasets import Cifar100
from paddle.vision.transforms import Normalize
from paddle.fluid.dataloader.collate import default_collate_fn
import signal
import os
from paddle.io import Dataset, DataLoader, DistributedBatchSampler


def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)
    return


def build_dataloader(mode,
                     batch_size=4,
                     seed=None,
                     num_workers=0,
                     device='gpu:0'):

    normalize = Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='HWC')

    if mode.lower() == "train":
        dataset = Cifar100(mode=mode, transform=normalize)
    elif mode.lower() in ["test", 'valid', 'eval']:
        dataset = Cifar100(mode="test", transform=normalize)
    else:
        raise ValueError(f"{mode} should be one of ['train', 'test']")

    # define batch sampler
    batch_sampler = DistributedBatchSampler(
        dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    data_loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        places=device,
        num_workers=num_workers,
        return_list=True,
        use_shared_memory=False)

    # support exit using ctrl+c
    signal.signal(signal.SIGINT, term_mp)
    signal.signal(signal.SIGTERM, term_mp)

    return data_loader


# cifar100 = Cifar100(mode='train', transform=normalize)

# data = cifar100[0]

# image, label = data

# reader = build_dataloader('train')

# for idx, data in enumerate(reader):
#     print(idx, data[0].shape, data[1].shape)
#     if idx >= 10:
#         break
