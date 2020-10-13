# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
import copy
import numpy as np
import os
import lmdb
import random
import signal
import paddle
from paddle.io import Dataset, DataLoader, DistributedBatchSampler, BatchSampler

from .imaug import transform, create_operators
from ppocr.utils.logging import get_logger


def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


signal.signal(signal.SIGINT, term_mp)
signal.signal(signal.SIGTERM, term_mp)


class ModeException(Exception):
    """
    ModeException
    """

    def __init__(self, message='', mode=''):
        message += "\nOnly the following 3 modes are supported: " \
                   "train, valid, test. Given mode is {}".format(mode)
        super(ModeException, self).__init__(message)


class SampleNumException(Exception):
    """
    SampleNumException
    """

    def __init__(self, message='', sample_num=0, batch_size=1):
        message += "\nError: The number of the whole data ({}) " \
                   "is smaller than the batch_size ({}), and drop_last " \
                   "is turnning on, so nothing  will feed in program, " \
                   "Terminated now. Please reset batch_size to a smaller " \
                   "number or feed more data!".format(sample_num, batch_size)
        super(SampleNumException, self).__init__(message)


def get_file_list(file_list, data_dir, delimiter='\t'):
    """
    read label list from file and shuffle the list

    Args:
        params(dict):
    """
    if isinstance(file_list, str):
        file_list = [file_list]
    data_source_list = []
    for file in file_list:
        with open(file) as f:
            full_lines = [line.strip() for line in f]
            for line in full_lines:
                try:
                    img_path, label = line.split(delimiter)
                except:
                    logger = get_logger()
                    logger.warning('label error in {}'.format(line))
                img_path = os.path.join(data_dir, img_path)
                data = {'img_path': img_path, 'label': label}
                data_source_list.append(data)
    return data_source_list


class LMDBDateSet(Dataset):
    def __init__(self, config, global_config):
        super(LMDBDateSet, self).__init__()
        self.data_list = self.load_lmdb_dataset(
            config['file_list'], global_config['max_text_length'])
        random.shuffle(self.data_list)

        self.ops = create_operators(config['transforms'], global_config)

        # for rec
        character = ''
        for op in self.ops:
            if hasattr(op, 'character'):
                character = getattr(op, 'character')

        self.info_dict = {'character': character}

    def load_lmdb_dataset(self, data_dir, max_text_length):
        self.env = lmdb.open(
            data_dir,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (data_dir))
            exit(0)

        filtered_index_list = []
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')
                if len(label) > max_text_length:
                    # print(f'The length of the label is longer than max_length: length
                    # {len(label)}, {label} in dataset {self.root}')
                    continue

                # By default, images containing characters which are not in opt.character are filtered.
                # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                filtered_index_list.append(index)
        return filtered_index_list

    def print_lmdb_sets_info(self, lmdb_sets):
        lmdb_info_strs = []
        for dataset_idx in range(len(lmdb_sets)):
            tmp_str = " %s:%d," % (lmdb_sets[dataset_idx]['dirpath'],
                                   lmdb_sets[dataset_idx]['num_samples'])
            lmdb_info_strs.append(tmp_str)
        lmdb_info_strs = ''.join(lmdb_info_strs)
        logger = get_logger()
        logger.info("DataSummary:" + lmdb_info_strs)
        return

    def __getitem__(self, idx):
        idx = self.data_list[idx]
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % idx
            label = txn.get(label_key)
            if label is not None:
                label = label.decode('utf-8')
                img_key = 'image-%09d'.encode() % idx
                imgbuf = txn.get(img_key)
                data = {'image': imgbuf, 'label': label}
                outs = transform(data, self.ops)
            else:
                outs = None
            if outs is None:
                return self.__getitem__(np.random.randint(self.__len__()))
            return outs

    def __len__(self):
        return len(self.data_list)


class SimpleDataSet(Dataset):
    def __init__(self, config, global_config):
        super(SimpleDataSet, self).__init__()
        delimiter = config.get('delimiter', '\t')
        self.data_list = get_file_list(config['file_list'], config['data_dir'],
                                       delimiter)
        random.shuffle(self.data_list)

        self.ops = create_operators(config['transforms'], global_config)

        # for rec
        character = ''
        for op in self.ops:
            if hasattr(op, 'character'):
                character = getattr(op, 'character')

        self.info_dict = {'character': character}

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data_list[idx])
        with open(data['img_path'], 'rb') as f:
            img = f.read()
            data['image'] = img
        outs = transform(data, self.ops)
        if outs is None:
            return self.__getitem__(np.random.randint(self.__len__()))
        return outs

    def __len__(self):
        return len(self.data_list)


class BatchBalancedDataLoader(object):
    def __init__(self,
                 dataset_list: list,
                 ratio_list: list,
                 distributed,
                 device,
                 loader_args: dict):
        """
        对datasetlist里的dataset按照ratio_list里对应的比例组合，似的每个batch里的数据按按照比例采样的
        :param dataset_list: 数据集列表
        :param ratio_list: 比例列表
        :param loader_args: dataloader的配置
        """
        assert sum(ratio_list) == 1 and len(dataset_list) == len(ratio_list)

        self.dataset_len = 0
        self.data_loader_list = []
        self.dataloader_iter_list = []
        all_batch_size = loader_args.pop('batch_size')
        batch_size_list = list(
            map(int, [max(1.0, all_batch_size * x) for x in ratio_list]))
        remain_num = all_batch_size - sum(batch_size_list)
        batch_size_list[np.argmax(ratio_list)] += remain_num

        for _dataset, _batch_size in zip(dataset_list, batch_size_list):
            if distributed:
                batch_sampler_class = DistributedBatchSampler
            else:
                batch_sampler_class = BatchSampler
            batch_sampler = batch_sampler_class(
                dataset=_dataset,
                batch_size=_batch_size,
                shuffle=loader_args['shuffle'],
                drop_last=loader_args['drop_last'], )
            _data_loader = DataLoader(
                dataset=_dataset,
                batch_sampler=batch_sampler,
                places=device,
                num_workers=loader_args['num_workers'],
                return_list=True, )
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
            self.dataset_len += len(_dataset)

    def __iter__(self):
        return self

    def __len__(self):
        return min([len(x) for x in self.data_loader_list])

    def __next__(self):
        batch = []
        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                _batch_i = next(data_loader_iter)
                batch.append(_batch_i)
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                _batch_i = next(self.dataloader_iter_list[i])
                batch.append(_batch_i)
            except ValueError:
                pass
        if len(batch) > 0:
            batch_list = []
            batch_item_size = len(batch[0])
            for i in range(batch_item_size):
                cur_item_list = [batch_i[i] for batch_i in batch]
                batch_list.append(paddle.concat(cur_item_list, axis=0))
        else:
            batch_list = batch[0]
        return batch_list


def fill_batch(batch):
    """
    2020.09.08： The current paddle version only supports returning data with the same length.
                Therefore, fill in the batches with inconsistent lengths.
                this method is currently only useful for text detection
    """
    keys = list(range(len(batch[0])))
    v_max_len_dict = {}
    for k in keys:
        v_max_len_dict[k] = max([len(item[k]) for item in batch])
    for item in batch:
        length = []
        for k in keys:
            v = item[k]
            length.append(len(v))
            assert isinstance(v, np.ndarray)
            if len(v) == v_max_len_dict[k]:
                continue
            try:
                tmp_shape = [v_max_len_dict[k] - len(v)] + list(v[0].shape)
            except:
                a = 1
            tmp_array = np.zeros(tmp_shape, dtype=v[0].dtype)
            new_array = np.concatenate([v, tmp_array])
            item[k] = new_array
        item.append(length)
    return batch
