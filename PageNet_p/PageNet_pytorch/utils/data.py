import cv2
import numpy as np 

def split_to_lines(label, num_char):
    lines = []
    start = 0
    for num in num_char:
        lines.append(label[start:start+num])
        start += num 
    return lines

def image_tensor_to_opencv(image, image_size=None):
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    image = image * 255 
    image = image.astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if not image_size is None:
        image_h, image_w = image_size 
        image = image[:image_h, :image_w]

    return image 