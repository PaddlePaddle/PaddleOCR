"""
-- Created by Pravesh Budhathoki
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2023-03-10
"""
from ocr.paddle_ocr import PaddleOCR
from paddleocr import logger

if __name__ == '__main__':
    engine = PaddleOCR()
    img_path = "/home/dell/Pictures/license_plate_data/car1.jpg"

    result = engine.ocr(img_path)
    if result is not None:
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                logger.info(line)
