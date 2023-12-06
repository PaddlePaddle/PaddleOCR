import os  
import cv2  
from PIL import Image  
  
# 定义你的数据集和标注文件的路径  
dataset_path = '/data/ailab/2022/rcz/paddles/PaddleOCR/datasets/v4_4_test_dataset'  
annotation_file = '/data/ailab/2022/rcz/paddles/PaddleOCR/datasets/v4_4_test_dataset/label.txt'  
  
# 定义你想要保存的小图片的路径和新的标注文件路径  
small_images_path = '/data/ailab/2022/rcz/paddles/PaddleOCR/datasets/v4_4_test_dataset_small'  
new_annotation_file = '/data/ailab/2022/rcz/paddles/PaddleOCR/datasets/v4_4_test_dataset_small/label.txt'  
  
# 确保目标文件夹存在  
os.makedirs(small_images_path, exist_ok=True)  
  
# 打开并读取标注文件  
with open(annotation_file, 'r') as f:  
    lines = f.readlines()  
  
# 遍历每一行  
for i, line in enumerate(lines):  
    # 获取图片名和标注信息  
    image_name = line.split("	")[0]
      
    # 获取图像的完整路径  
    image_path = os.path.join(dataset_path, image_name)  
      
    # 加载图像并获取其尺寸  
    # print(i, image_path)
    try:
        image = cv2.imread(image_path)  
        height, width, _ = image.shape  
        
        # 如果图像的宽度和高度都小于2000而且长宽比小于2，将其复制到新的文件夹，并保存其标注信息  
        if height < 2000 and width < 2000:  
            if max(height, width)/min(height,width) < 2:
                print(i, height, width, image_path)
                small_image_path = os.path.join(small_images_path, image_name)  
                cv2.imwrite(small_image_path, image)
                with open(new_annotation_file, 'a') as f:  
                    f.write(f'{line}')
    except:
        continue