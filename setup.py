from setuptools import setup
from io import open

with open('requirments.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()
    requirements.append('tqdm')

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README

setup(
    name='ppocr',
    packages=['ppocr','tools/infer'],
    include_package_data=True,
    entry_points={"console_scripts": ["ppocr= ppocr.pp_ocr:main"]},
    version='0.0.1',
    install_requires=requirements,
    license='Apache License 2.0',
    description='Awesome OCR toolkits based on PaddlePaddle （8.6M ultra-lightweight pre-trained model, support training and deployment among server, mobile, embeded and IoT devices',
    long_description=readme(),
    author='Baidu PaddlePaddle',
    url='https://github.com/PaddlePaddle/PaddleOCR',
    download_url='https://github.com/PaddlePaddle/PaddleOCR.git',
    keywords=['ocr textdetection textrecognition paddleocr crnn east star-net rosetta ocrlite db chineseocr chinesetextdetection chinesetextrecognition'],
    classifiers = [
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Natural Language :: Chinese (Simplified)',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.5',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Utilities'
      ],
)