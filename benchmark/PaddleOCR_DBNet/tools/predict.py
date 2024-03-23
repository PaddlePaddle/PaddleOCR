# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun

import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import time
import cv2
import paddle

from data_loader import get_transforms
from models import build_model
from post_processing import get_post_processing


def resize_image(img, short_size):
    height, width, _ = img.shape
    if height < width:
        new_height = short_size
        new_width = new_height / height * width
    else:
        new_width = short_size
        new_height = new_width / width * height
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


class PaddleModel:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        '''
        初始化模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(
                self.gpu_id, int) and paddle.device.is_compiled_with_cuda():
            paddle.device.set_device("gpu:{}".format(self.gpu_id))
        else:
            paddle.device.set_device("cpu")
        checkpoint = paddle.load(model_path)

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.model = build_model(config['arch'])
        self.post_process = get_post_processing(config['post_processing'])
        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args'][
            'img_mode']
        self.model.set_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

    def predict(self,
                img_path: str,
                is_output_polygon=False,
                short_size: int=1024):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        batch = {'shape': [(h, w)]}
        with paddle.no_grad():
            start = time.time()
            preds = self.model(tensor)
            box_list, score_list = self.post_process(
                batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(
                        axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


def save_depoly(net, input, save_path):
    input_spec = [
        paddle.static.InputSpec(
            shape=[None, 3, None, None], dtype="float32")
    ]
    net = paddle.jit.to_static(net, input_spec=input_spec)

    # save static model for inference directly
    paddle.jit.save(net, save_path)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.paddle')
    parser.add_argument('--model_path', default=r'model_best.pth', type=str)
    parser.add_argument(
        '--input_folder',
        default='./test/input',
        type=str,
        help='img path for predict')
    parser.add_argument(
        '--output_folder',
        default='./test/output',
        type=str,
        help='img path for output')
    parser.add_argument('--gpu', default=0, type=int, help='gpu for inference')
    parser.add_argument(
        '--thre', default=0.3, type=float, help='the thresh of post_processing')
    parser.add_argument(
        '--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument(
        '--save_result',
        action='store_true',
        help='save box and score to txt file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import pathlib
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from utils.util import show_img, draw_bbox, save_result, get_image_file_list

    args = init_args()
    print(args)
    # 初始化网络
    model = PaddleModel(args.model_path, post_p_thre=args.thre, gpu_id=args.gpu)
    img_folder = pathlib.Path(args.input_folder)
    for img_path in tqdm(get_image_file_list(args.input_folder)):
        preds, boxes_list, score_list, t = model.predict(
            img_path, is_output_polygon=args.polygon)
        img = draw_bbox(cv2.imread(img_path)[:, :, ::-1], boxes_list)
        if args.show:
            show_img(preds)
            show_img(img, title=os.path.basename(img_path))
            plt.show()
        # 保存结果到路径
        os.makedirs(args.output_folder, exist_ok=True)
        img_path = pathlib.Path(img_path)
        output_path = os.path.join(args.output_folder,
                                   img_path.stem + '_result.jpg')
        pred_path = os.path.join(args.output_folder,
                                 img_path.stem + '_pred.jpg')
        cv2.imwrite(output_path, img[:, :, ::-1])
        cv2.imwrite(pred_path, preds * 255)
        save_result(
            output_path.replace('_result.jpg', '.txt'), boxes_list, score_list,
            args.polygon)
