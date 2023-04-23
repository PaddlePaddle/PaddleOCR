import os
import copy
import json
import bezier
import numpy as np
from PIL import Image
from collections import defaultdict
from paddle.io import Dataset

from .imaug import transform, create_operators
from .imaug.spts_process import bezier2bbox


class TextSpottingDataset(Dataset):
    """spts数据集"""
    def __init__(self, config, mode, logger, seed=None, epoch=-1):
        super(TextSpottingDataset, self).__init__()
        self.global_config = config['Global']
        dataset_config = config[mode]['dataset']
        self.ratio_list = dataset_config['ratio_list']
        self.init_dataset(dataset_config=dataset_config)
        self.ops = create_operators(dataset_config['transforms'])
        self.mode = mode
        self.dataset_config = config[mode]

        self.need_reset = True in [x < 1 for x in self.ratio_list]

    def init_dataset(self, dataset_config):
        label_file = os.path.join(dataset_config['data_dir'],dataset_config['label_file_list'][1])

        with open(label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        assert type(labels)==dict, f'annotation file format {type(labels)} not supported'
        anns, self.imgs = {}, {}
        self.imgToAnns = defaultdict(list)
        if 'annotations' in labels:
            for ann in labels['annotations']:
                self.imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann
        if 'images' in labels:
            for img in labels['images']:
                self.imgs[img['id']] = img
        self.ids = list(sorted(self.imgs.keys()))

    def __getitem__(self, index):
        idx = self.ids[index]
        image = Image.open(os.path.join(self.dataset_config['dataset']['data_dir'], 
                                        self.dataset_config['dataset']['label_file_list'][0], 
                                        self.imgs[idx]['file_name'])).convert('RGB')
        anno = [] if idx not in self.imgToAnns.keys() else self.imgToAnns[idx]
        # anno = []
        image_w, image_h = image.size
        anno = [ele for ele in anno if 'iscrowd' not in anno or ele['iscrowd'] == 0]

        target = {}
        target['image_id'] = idx
        target['area'] = np.array([ele['area'] for ele in anno])
        target['labels'] = np.array([ann['category_id'] for ann in anno])
        target['iscrowd'] = np.array([ann['iscrowd'] for ann in anno])
        image_size = np.array([int(image_h), int(image_w)])
        target['orig_size'] = image_size 
        target['size'] = image_size 
        target['recog'] = np.array([ann['rec'] for ann in anno])
        target['bezier_pts'] = np.array([ann['bezier_pts'] for ann in anno])
        bboxes = []
        for bezier_pt in target['bezier_pts']:
            bbox = bezier2bbox(bezier_pt)
            bboxes.append(bbox)
        target['bboxes'] = np.array(bboxes).reshape([-1, 4])

        data = {'image':image, 'target':target}

        return_data = {}
        image_list = []
        # target_list = []
        seq_list = []
        # 以第一份数据的shape为准
        tmp_data = copy.deepcopy(data)
        tmp_data = transform(tmp_data, self.ops) if len(self.ops) != 0 else data
        tmp_image_data_shape = tmp_data['image'].shape
        padding_shape = list(tmp_data['image'].shape)
        image_list.append(tmp_data['image'])
        seq_list.append(tmp_data['sequence'] if self.mode == 'Train' else tmp_data['val_sequence'])
        for batch_idx in range(self.dataset_config['loader']['num_batch'] - 1):
            tmp_data = copy.deepcopy(data)
            tmp_data = transform(tmp_data, self.ops) if len(self.ops) != 0 else data
            while tmp_data['sequence'].shape != seq_list[0].shape:
                tmp_data = copy.deepcopy(data)
                tmp_data = transform(tmp_data, self.ops) if len(self.ops) != 0 else data
            tmp_image_data_shape = tmp_data['image'].shape
            padding_shape[1] = max(padding_shape[1], tmp_image_data_shape[1])
            padding_shape[2] = max(padding_shape[2], tmp_image_data_shape[2])
            image_list.append(tmp_data['image'])
            # target_list.append(tmp_data['target'])
            seq_list.append(tmp_data['sequence'] if self.mode == 'Train' else tmp_data['val_sequence'])

        image_list, mask_list = self.get_mask(image_list, padding_shape)
        return_data['image'] = image_list
        return_data['sequence'] = seq_list
        return_data['mask'] = mask_list
        if self.mode == 'Eval':
            # return [return_data, tmp_data['target']]
            return [[return_data['image'], return_data['sequence'], return_data['mask']], tmp_data['target']]
        return [[return_data['image'], return_data['sequence'], return_data['mask']]]

    def __len__(self):
        return int(len(self.ids) * self.ratio_list[0])

    def get_mask(self, image_list, padding_shape):
        return_image_list, mask_list = [], []
        for image in image_list:
            pad_img = np.zeros(padding_shape).astype("float32")
            mask = np.ones(pad_img.shape[1:])
            pad_img[: image.shape[0], :image.shape[1], :image.shape[2]] = image
            mask[:image.shape[1], :image.shape[2]] = 0
            return_image_list.append(pad_img)
            mask_list.append(mask)
        return return_image_list, mask_list
