# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 13:12
# @Author  : zhoujun
import copy
from paddle.io import Dataset
from data_loader.modules import *


class BaseDataSet(Dataset):
    def __init__(self,
                 data_path: str,
                 img_mode,
                 pre_processes,
                 filter_keys,
                 ignore_tags,
                 transform=None,
                 target_transform=None):
        assert img_mode in ['RGB', 'BRG', 'GRAY']
        self.ignore_tags = ignore_tags
        self.data_list = self.load_data(data_path)
        item_keys = [
            'img_path', 'img_name', 'text_polys', 'texts', 'ignore_tags'
        ]
        for item in item_keys:
            assert item in self.data_list[
                0], 'data_list from load_data must contains {}'.format(
                    item_keys)
        self.img_mode = img_mode
        self.filter_keys = filter_keys
        self.transform = transform
        self.target_transform = target_transform
        self._init_pre_processes(pre_processes)

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self, data_path: str) -> list:
        """
        把数据加载为一个list：
        :params data_path: 存储数据的文件夹或者文件
        return a dict ,包含了，'img_path','img_name','text_polys','texts','ignore_tags'
        """
        raise NotImplementedError

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def __getitem__(self, index):
        try:
            data = copy.deepcopy(self.data_list[index])
            im = cv2.imread(data['img_path'], 1
                            if self.img_mode != 'GRAY' else 0)
            if self.img_mode == 'RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            data['img'] = im
            data['shape'] = [im.shape[0], im.shape[1]]
            data = self.apply_pre_processes(data)

            if self.transform:
                data['img'] = self.transform(data['img'])
            data['text_polys'] = data['text_polys'].tolist()
            if len(self.filter_keys):
                data_dict = {}
                for k, v in data.items():
                    if k not in self.filter_keys:
                        data_dict[k] = v
                return data_dict
            else:
                return data
        except:
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)
