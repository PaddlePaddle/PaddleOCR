#!/usr/bin/env python3
"""Convert the b6 jersey-number .h5 datasets into PaddleOCR recognition format.

Each h5 sample is a group <idx> holding:
  <idx>/image : (H, W, 3) uint8, RGB order
  <idx>/label : (2,)      int64, [first_digit, second_digit]
                second_digit == 10 means "no second digit" (single-digit number)

Output (relative to --out, default ./train_data):
  train_data/rec/<split>/<idx>.png   cropped images (written BGR so cv2.imread
                                      round-trips to the BGR the model expects)
  train_data/<split>_list.txt        "<relpath>\t<text>" lines, paths relative to data_dir

The recognition model is loaded by PaddleOCR's DecodeImage(img_mode="BGR"), i.e.
cv2.imread, which yields BGR arrays. The h5 stores RGB, so we swap RGB->BGR before
cv2.imwrite; the file is then a correct-looking colour image AND reads back as BGR,
matching the convention the pretrained model was trained on.
"""
import argparse
import os

import cv2
import h5py


def decode_label(lab):
    d0, d1 = int(lab[0]), int(lab[1])
    s = str(d0)
    if d1 != 10:
        s += str(d1)
    return s


def convert_split(h5_path, split, out_dir):
    img_subdir = os.path.join("rec", split)              # path stored in list file
    img_abs_dir = os.path.join(out_dir, img_subdir)
    os.makedirs(img_abs_dir, exist_ok=True)
    list_path = os.path.join(out_dir, f"{split}_list.txt")

    n = 0
    with h5py.File(h5_path, "r") as f, open(list_path, "w", encoding="utf-8") as lst:
        # sort numerically so output order is stable/readable
        keys = sorted(f.keys(), key=lambda k: int(k))
        for k in keys:
            rgb = f[k]["image"][()]                       # (H, W, 3) RGB uint8
            text = decode_label(f[k]["label"][()])
            bgr = rgb[:, :, ::-1]                          # RGB -> BGR channel swap
            rel = os.path.join(img_subdir, f"{k}.png")
            cv2.imwrite(os.path.join(out_dir, rel), bgr)
            lst.write(f"{rel}\t{text}\n")
            n += 1
    print(f"  {split:5s}: {n:6d} samples -> {list_path}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", default=os.path.dirname(os.path.abspath(__file__)),
                    help="directory containing train.h5 / val.h5 / test.h5")
    ap.add_argument("--out", default="train_data",
                    help="output data_dir (PaddleOCR data_dir)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    splits = {"train": "train.h5", "val": "val.h5", "test": "test.h5"}
    print(f"Converting -> {os.path.abspath(args.out)}")
    total = 0
    for split, fname in splits.items():
        h5_path = os.path.join(args.h5_dir, fname)
        if not os.path.exists(h5_path):
            print(f"  {split:5s}: SKIP (missing {h5_path})")
            continue
        total += convert_split(h5_path, split, args.out)
    print(f"Done. {total} images written.")


if __name__ == "__main__":
    main()
