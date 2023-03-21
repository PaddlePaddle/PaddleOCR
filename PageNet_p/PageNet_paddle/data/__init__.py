from paddle.vision.transforms import Compose
from paddle.io import DataLoader
from .lmdb_dataset import LmdbDataset
from .transforms import RandomResize, SizeAjust, ToTensor

def build_dataset(cfg, image_set):
    data_root = cfg['DATA'][image_set.upper()]['DATA_ROOT']
    image_mode = cfg['DATA'][image_set.upper()]['IMAGE_MODE']
    transforms = build_transforms(cfg, image_set)
    return LmdbDataset(data_root, image_mode, transforms)

def build_transforms(cfg, image_set):
    tfm_cfgs = cfg['DATA'][image_set.upper()]

    transforms = []
    force_resize = tfm_cfgs['FORCE_RESIZE'] if 'FORCE_RESIZE' in tfm_cfgs else True
    transforms.append(RandomResize(tfm_cfgs['WIDTHS'], tfm_cfgs['MAX_HEIGHT'], force_resize))
    transforms.append(SizeAjust(tfm_cfgs['SIZE_STRIDE']))
    transforms.append(ToTensor())
    if len(transforms) == 0:
        return None 
    return Compose(transforms)  

def build_dataloader(dataset, image_set, cfg):
    dataloader = DataLoader(  
        dataset=dataset,
        batch_size=cfg['DATA'][image_set.upper()]['BATCH_SIZE'],
        shuffle=(image_set == 'train'),
        num_workers=cfg['DATA']['NUM_WORKER'],
    )
    return dataloader