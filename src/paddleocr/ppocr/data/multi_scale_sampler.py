from paddle.io import Sampler
import paddle.distributed as dist

import numpy as np
import random
import math


class MultiScaleSampler(Sampler):
    def __init__(
        self,
        data_source,
        scales,
        first_bs=128,
        fix_bs=True,
        divided_factor=[8, 16],
        is_training=True,
        ratio_wh=0.8,
        max_w=480.0,
        seed=None,
    ):
        """
        multi scale samper
        Args:
            data_source(dataset)
            scales(list): several scales for image resolution
            first_bs(int): batch size for the first scale in scales
            divided_factor(list[w, h]): ImageNet models down-sample images by a factor, ensure that width and height dimensions are multiples are multiple of devided_factor.
            is_training(boolean): mode
        """
        # min. and max. spatial dimensions
        self.data_source = data_source
        self.data_idx_order_list = np.array(data_source.data_idx_order_list)
        self.ds_width = data_source.ds_width
        self.seed = data_source.seed
        if self.ds_width:
            self.wh_ratio = data_source.wh_ratio
            self.wh_ratio_sort = data_source.wh_ratio_sort
        self.n_data_samples = len(self.data_source)
        self.ratio_wh = ratio_wh
        self.max_w = max_w

        if isinstance(scales[0], list):
            width_dims = [i[0] for i in scales]
            height_dims = [i[1] for i in scales]
        elif isinstance(scales[0], int):
            width_dims = scales
            height_dims = scales
        base_im_w = width_dims[0]
        base_im_h = height_dims[0]
        base_batch_size = first_bs

        # Get the GPU and node related information
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        # adjust the total samples to avoid batch dropping
        num_samples_per_replica = int(self.n_data_samples * 1.0 / num_replicas)

        img_indices = [idx for idx in range(self.n_data_samples)]

        self.shuffle = False
        if is_training:
            # compute the spatial dimensions and corresponding batch size
            # ImageNet models down-sample images by a factor of 32.
            # Ensure that width and height dimensions are multiples are multiple of 32.
            width_dims = [
                int((w // divided_factor[0]) * divided_factor[0]) for w in width_dims
            ]
            height_dims = [
                int((h // divided_factor[1]) * divided_factor[1]) for h in height_dims
            ]

            img_batch_pairs = list()
            base_elements = base_im_w * base_im_h * base_batch_size
            for h, w in zip(height_dims, width_dims):
                if fix_bs:
                    batch_size = base_batch_size
                else:
                    batch_size = int(max(1, (base_elements / (h * w))))
                img_batch_pairs.append((w, h, batch_size))
            self.img_batch_pairs = img_batch_pairs
            self.shuffle = True
        else:
            self.img_batch_pairs = [(base_im_w, base_im_h, base_batch_size)]

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas

        self.batch_list = []
        self.current = 0
        last_index = num_samples_per_replica * num_replicas
        indices_rank_i = self.img_indices[self.rank : last_index : self.num_replicas]
        while self.current < self.n_samples_per_replica:
            for curr_w, curr_h, curr_bsz in self.img_batch_pairs:
                end_index = min(self.current + curr_bsz, self.n_samples_per_replica)
                batch_ids = indices_rank_i[self.current : end_index]
                n_batch_samples = len(batch_ids)
                if n_batch_samples != curr_bsz:
                    batch_ids += indices_rank_i[: (curr_bsz - n_batch_samples)]
                self.current += curr_bsz

                if len(batch_ids) > 0:
                    batch = [curr_w, curr_h, len(batch_ids)]
                    self.batch_list.append(batch)
        random.shuffle(self.batch_list)
        self.length = len(self.batch_list)
        self.batchs_in_one_epoch = self.iter()
        self.batchs_in_one_epoch_id = [i for i in range(len(self.batchs_in_one_epoch))]

    def __iter__(self):
        if self.seed is None:
            random.seed(self.epoch)
            self.epoch += 1
        else:
            random.seed(self.seed)
        random.shuffle(self.batchs_in_one_epoch_id)
        for batch_tuple_id in self.batchs_in_one_epoch_id:
            yield self.batchs_in_one_epoch[batch_tuple_id]

    def iter(self):
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            else:
                random.seed(self.epoch)
            if not self.ds_width:
                random.shuffle(self.img_indices)
            random.shuffle(self.img_batch_pairs)
            indices_rank_i = self.img_indices[
                self.rank : len(self.img_indices) : self.num_replicas
            ]
        else:
            indices_rank_i = self.img_indices[
                self.rank : len(self.img_indices) : self.num_replicas
            ]

        start_index = 0
        batchs_in_one_epoch = []
        for batch_tuple in self.batch_list:
            curr_w, curr_h, curr_bsz = batch_tuple
            end_index = min(start_index + curr_bsz, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != curr_bsz:
                batch_ids += indices_rank_i[: (curr_bsz - n_batch_samples)]
            start_index += curr_bsz

            if len(batch_ids) > 0:
                if self.ds_width:
                    wh_ratio_current = self.wh_ratio[self.wh_ratio_sort[batch_ids]]
                    ratio_current = wh_ratio_current.mean()
                    ratio_current = (
                        ratio_current
                        if ratio_current * curr_h < self.max_w
                        else self.max_w / curr_h
                    )
                else:
                    ratio_current = None
                batch = [(curr_w, curr_h, b_id, ratio_current) for b_id in batch_ids]
                # yield batch
                batchs_in_one_epoch.append(batch)
        return batchs_in_one_epoch

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length
