# PageNet: Towards End-to-End Weakly Supervised Page-Level Handwritten Chinese Text Recognition

The official implementation of [PageNet: Towards End-to-End Weakly Supervised Page-Level Handwritten Chinese Text Recognition](https://arxiv.org/abs/2207.14807) (IJCV 2022). 

## Environment
We recommend using [Anaconda](https://www.anaconda.com/) to manage environments.
```
conda create -n pagenet python=3.7 -y 
conda activate pagenet
git clone https://github.com/shannanyinxiang/PageNet
cd PageNet
pip install -r requirements.txt
```

## Dataset
- ICDAR2013 Competition Dataset: [BaiduNetDisk](https://pan.baidu.com/s/1uM2u1O9cByZtOdXyBUs6lw?pwd=uqxp) or [Google Drive](https://drive.google.com/drive/folders/120phawO79BxCSgzwaBl1vO6iYXexzZeB?usp=share_link)
- SCUT-HCCDoc: Please apply for this dataset at [SCUT-HCCDoc_Dataset_Release](https://github.com/HCIILAB/SCUT-HCCDoc_Dataset_Release).
- MTHv2: [BaiduNetDisk](https://pan.baidu.com/s/1fDU1zlynG1UpQThf2-2LKA?pwd=9c53) or [Google Drive](https://drive.google.com/drive/folders/1UfU4CA3HE-zq2AjY26_QTfKaTtk2p1jw?usp=share_link)

Download the datasets and put them into the `datasets` folder following the file structure below.
```
datasets
├─IC13Comp
├─MTHv2_test
└─raw
   └─SCUT-HCCDoc
      │  hccdoc_test.json
      │  hccdoc_train.json
      └─image
```

Then run the following command to generate the SCUT-HCCDoc dataset in lmdb format.
```
python tools/convert_hccdoc_to_lmdb.py \
  --image_root datasets/raw/SCUT-HCCDoc/image/ \
  --annotation_file datasets/raw/SCUT-HCCDoc/hccdoc_test.json \
  --dict_path dicts/scut-hccdoc.txt \
  --lmdb_root datasets/SCUT-HCCDoc_test
```

## Inference

### ICDAR2013 Competition Dataset 

1. Download the pretrained weights from [BaiduNetDisk](https://pan.baidu.com/s/1FjgZIn0FiK1FU5NxUxPeig?pwd=b3ym) or [Google Drive](https://drive.google.com/file/d/1YxDbrCm0WNjJ05LK4uN7W4VMEzxf7LNg/view?usp=share_link) and put it into the `outputs/casia-hwdb/checkpoints` folder.

2. Run the following command:
```
python main.py --config configs/casia-hwdb.yaml 
```
The results will be saved at `outputs/casia-hwdb/val_log.txt`.

### SCUT-HCCDoc 

1. Download the pretrained weights from [BaiduNetDisk](https://pan.baidu.com/s/1nYcZk9ektLMVIynMORewOg?pwd=dgvh) or [Google Drive](https://drive.google.com/file/d/1ZVuR-qJ9Opj9HC1tuv_5zqvaGkpeic5f/view?usp=share_link) and put it into the `outputs/scut-hccdoc/checkpoints` folder.

2. Run the following command:
```
python main.py --config configs/scut-hccdoc.yaml
```
The results will be saved at `outputs/scut-hccdoc/val_log.txt`.

### MTHv2 

1. Download the pretrained weights from [BaiduNetDisk](https://pan.baidu.com/s/1zRNkUCJnltE0XExlWhbyLg?pwd=0gsw) or [Google Drive](https://drive.google.com/file/d/15NVsNq4gXaSEW2S2Am3tcd0dYti10at8/view?usp=share_link) and put it into the `outputs/mthv2/checkpoints` folder.

2. Run the following command:
```
python main.py --config configs/mthv2.yaml
```
The results will be saved at `outputs/mthv2/val_log.txt`.

### Model Performance

The performance of the provided models on these datasets should be:

| Dataset | $AR^*$ | $CR^*$ |
| :---    | :---:  | :---:  |
| ICDAR2013 Competition Dataset | 92.87 | 93.34 |
| SCUT-HCCDoc | 78.70 | 84.29 |
| MTHv2 | 93.76 | 96.03 | 

## Training
Currently the training codes are not available. For questions about model training, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn) and Mr. Dezhi Peng (eedzpeng@mail.scut.edu.cn).

Note: In the spatial matching of the weakly supervised learning, we found it better to simply delete the matching pairs whose IoUs are equal to zero.

## Citation
```
@article{peng2022pagenet,
  title={PageNet: Towards End-to-End Weakly Supervised Page-Level Handwritten Chinese Text Recognition},
  author={Peng, Dezhi and Jin, Lianwen and Liu, Yuliang and Luo, Canjie and Lai, Songxuan},
  journal={International Journal of Computer Vision},
  pages={2623--2645},
  year={2022},
  volume={130},
  number={11},
  doi={10.1007/s11263-022-01654-0},
}
```

## License

This repository should be used and distributed under [Creative Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) License](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial research purposes.

## Copyright
This repository can only be used for non-commercial research purpose.

For commercial use, please contact Prof. Lianwen Jin (eelwjin@scut.edu.cn).

Copyright 2022, [Deep Learning and Vision Computing Lab](http://www.dlvc-lab.net), South China University of Technology. 