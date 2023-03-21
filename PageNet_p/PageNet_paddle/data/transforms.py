import cv2
import random
import paddle.vision.transforms as transforms

class RandomResize(object):
    def __init__(self, widths, max_height, force_resize=True):
        self.widths = widths 
        self.max_height = max_height
        self.force_resize = force_resize

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        img_h, img_w = image.shape[:2]
        tgt_w = random.choice(self.widths)   

        if (not self.force_resize) \
           and img_h <= self.max_height \
           and img_w <= tgt_w:
            return sample

        fx = tgt_w / img_w
        if img_h * fx <= self.max_height:
            fy = fx
        else:
            fy = self.max_height / img_h
            fx = fy

        image = cv2.resize(image, None, fx=fx, fy=fy)
        label[:, [1, 3]] = label[:, [1, 3]] * fx
        label[:, [2, 4]] = label[:, [2, 4]] * fy

        sample['image'] = image
        sample['label'] = label

        return sample


class SizeAjust(object):
    def __init__(self, stride):
        self.stride = stride  

    def __call__(self, sample):
        image = sample['image']
        label = sample['label']

        img_h, img_w = image.shape[:2]
        tgt_h = (img_h // self.stride + 1) * self.stride if img_h % self.stride != 0 else img_h 
        tgt_w = (img_w // self.stride + 1) * self.stride if img_w % self.stride != 0 else img_w 
        image = cv2.resize(image, (tgt_w, tgt_h))

        fx = tgt_w / img_w 
        fy = tgt_h / img_h 
        label[:, [1, 3]] = label[:, [1, 3]] * fx 
        label[:, [2, 4]] = label[:, [2, 4]] * fy 

        sample['image'] = image 
        sample['label'] = label 

        return sample


class ToTensor(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()  

    def __call__(self, sample):
        sample['image'] = self.to_tensor(sample['image'])
        return sample