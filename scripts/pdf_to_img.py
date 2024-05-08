import fitz  # PyMuPDF
import os

from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)


def pdf_to_images(pdf_path, output_folder):
    images = convert_from_path(pdf_path, poppler_path=r'D:\poppler-24.02.0\Library\bin')
    input_name = os.path.basename(pdf_path).split(".")[0]

    for idx, image in enumerate(images):        
        # 生成图像文件名
        image_filename = f"{output_folder}/{input_name}_page_{idx + 1}.png"
        
        # 保存图像
        image.save(image_filename)
        print(f"Page {idx + 1} saved as {image_filename}")



if __name__ == "__main__":
    # 输入 PDF 文件路径和输出文件夹路径
    pdf_path = r"E:\projects\ballooning\data\pdf_unlabeled\33.pdf"
    output_folder = r"E:\projects\ballooning\data\image_unlabeled"

    # 调用函数将 PDF 转换为图像
    pdf_to_images(pdf_path, output_folder)
