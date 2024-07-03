import os
import cv2

dataset_path = "datasets/v4_4_test_dataset"
annotation_file = "datasets/v4_4_test_dataset/label.txt"

small_images_path = "datasets/v4_4_test_dataset_small"
new_annotation_file = "datasets/v4_4_test_dataset_small/label.txt"

os.makedirs(small_images_path, exist_ok=True)

with open(annotation_file, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    image_name = line.split("   ")[0]

    image_path = os.path.join(dataset_path, image_name)

    try:
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # 如果图像的宽度和高度都小于2000而且长宽比小于2，将其复制到新的文件夹，并保存其标注信息
        if height < 2000 and width < 2000:
            if max(height, width) / min(height, width) < 2:
                print(i, height, width, image_path)
                small_image_path = os.path.join(small_images_path, image_name)
                cv2.imwrite(small_image_path, image)
                with open(new_annotation_file, "a") as f:
                    f.write(f"{line}")
    except:
        continue
