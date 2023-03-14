"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2023-03-10
"""
import os
from typing import List

import cv2
import numpy as np
from pydantic import BaseModel

from ocr.paddle_ocr import PaddleOCR
from paddleocr import logger


class Point(BaseModel):
    x: int = 0
    y: int = 0


def list_files(directory, extensions=None, shuffle=False):
    """
    Lists files in a directory
    :return:
    """

    images = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)

            if extensions is not None:
                if file_path.endswith(tuple(extensions)):
                    images.append(file_path)
            else:
                images.append(file_path)
    if shuffle:
        np.random.shuffle(images)
    return images


def get_images_path(path):
    image_extension = [".jpg", ".png", ".jpeg"]
    path_split = path.split("/")
    if path_split[-1].endswith(tuple(image_extension)):
        list_images_path = [path]
    else:
        list_images_path = sorted(list_files(directory=path, shuffle=False, extensions=[".jpg", ".png", ".jpeg"]))
    return list_images_path


def draw_polygon_v2(img, polygon: List[Point], color=None, isclosed=True, thickness=2):
    """
    Params:
    img:np.ndarray
    polygon:List[Point], list of polygon points
    color:must be in BGR format i.e.(0,255,0) draws green line
    """
    if color is None:
        color = list(np.random.random(size=3) * 256)
    len_polygon = len(polygon)
    for i in range(len_polygon - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        cv2.line(img, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), color=color, thickness=thickness)

    # enclosed polygon by drawing last line
    if isclosed:
        p1 = polygon[0]
        p2 = polygon[len_polygon - 1]
        cv2.line(img, (int(p1.x), int(p1.y)), (int(p2.x), int(p2.y)), color=color, thickness=thickness)
    return img


if __name__ == '__main__':
    engine = PaddleOCR()
    img_path = "/home/dell/Downloads/embosed_number_plate/embosed_license_plate_ocr_test_images/plates"
    list_images = get_images_path(img_path)

    for im_path in list_images:
        image_file_name = im_path.split("/")[-1]
        image = cv2.imread(im_path)

        # text detection and text recognition
        results = engine.ocr(image, det=True, rec=True)
        if results is not None:
            for paddleocr_result in results[0]:
                word_coordinates1 = paddleocr_result[0]
                list_points1 = []
                print(f"Results {image_file_name}")
                for xy_point1 in word_coordinates1:
                    _x, _y = list(map(int, xy_point1))
                    point1 = Point()
                    point1.x = _x
                    point1.y = _y
                    list_points1.append(point1)
                image = draw_polygon_v2(image, list_points1, thickness=1)
                print(paddleocr_result)
        cv2.namedWindow(f"{image_file_name}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"{image_file_name}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
