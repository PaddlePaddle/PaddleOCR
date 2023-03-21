import sys 
sys.path.append('.')

import os 
import cv2
import yaml
from tqdm import tqdm
from utils.converter import Converter
from utils.data import image_tensor_to_opencv, split_to_lines
from data import build_dataset, build_dataloader

def main(cfg, args):
    dataset = build_dataset(cfg, args.image_set)
    dataloader = build_dataloader(dataset, args.image_set, cfg)
    converter = Converter(cfg['DATA']['DICT'])

    os.makedirs(args.save_folder, exist_ok=True)
    for sample in tqdm(dataloader):
        images, labels, num_chars, filenames = sample['image'], sample['label'], sample['num_char_per_line'], sample['filename']
        for image, label, num_char, filename in zip(images, labels, num_chars, filenames):
            image = image_tensor_to_opencv(image)
            
            lines = split_to_lines(label, num_char)
            line_strs = [converter.decode(line[:, 0]) for line in lines] 

            image_path = os.path.join(args.save_folder, filename + '.jpg')
            label_path = os.path.join(args.save_folder, filename + '.txt')
            cv2.imwrite(image_path, image)
            with open(label_path, 'w') as f:
                f.write('\n'.join(line_strs))


if __name__ == '__main__':
    from utils.parser import default_parser

    parser = default_parser()
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--image_set', type=str, required=True, choices=['train', 'val'])
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    
    main(cfg, args)