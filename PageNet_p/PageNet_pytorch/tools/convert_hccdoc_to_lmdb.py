import sys 
sys.path.append('.')

import os
import cv2
import lmdb
import json
import numpy as np

from tqdm import tqdm

from utils.converter import Converter

def create_lmdb_dataset(samples, lmdb_root):
    os.makedirs(os.path.dirname(lmdb_root), exist_ok=True)
    env = lmdb.open(lmdb_root, map_size=1099511627776)

    cache = {}
    count = 0
    for sample in tqdm(samples):
        count += 1

        filename_key = f"filename-{count:06d}".encode()
        cache[filename_key] = sample['filename'].encode()

        image_key = f"image-{count:06d}".encode()
        image = open(sample['file_path'], 'rb').read()
        cache[image_key] = image

        line_label_indices = sample['line labels'].astype(np.float32)[:, np.newaxis]
        line_label_indices = np.pad(line_label_indices, ((0, 0), (0, 4)), mode='constant', constant_values=0)
        char_nums = sample['char nums'].astype(np.int32)
        label_key = f'label-{count:06d}'.encode()
        cache[label_key] = line_label_indices.tobytes()
        char_num_key = f'numchar-{count:06d}'.encode()
        cache[char_num_key] = char_nums.tobytes()

        if count % 1000 == 0:
            writeCache(env, cache)
            cache = {}
    
    num_sample_key = 'num-samples'.encode()
    cache[num_sample_key] = str(count).encode()
    writeCache(env, cache)
    print(f'Create dataset with {count} samples')

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)

def read_json(json_path, image_root, converter):
    label = json.load(open(json_path, 'r'))
    
    samples = []
    for _, annotations in label['annotations'].items():
        for anno in annotations:
            file_path = os.path.join(image_root, anno['file_path'])
            filename = os.path.splitext(os.path.basename(file_path))[0]
            char_nums = []
            line_labels = []
            for text_ins in anno['gt']:
                char_indices = converter.encode(text_ins['text'])
                char_nums.append(len(char_indices))
                line_labels.extend(char_indices)

            samples.append(
                {'filename': filename,
                 'file_path': file_path,
                 'line labels': np.array(line_labels),
                 'char nums': np.array(char_nums)}
            )

    return samples

def main(args):
    converter = Converter(args.dict_path)

    print('Read annotation file...')
    samples = read_json(args.annotation_file, args.image_root, converter)
    
    print('Create LMDB dataset...')
    create_lmdb_dataset(samples, args.lmdb_root)

if __name__ == '__main__':
    import argparse 

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', type=str)
    parser.add_argument('--annotation_file', type=str)
    parser.add_argument('--dict_path', type=str)
    parser.add_argument('--lmdb_root', type=str)
    args = parser.parse_args()

    main(args)