#!/usr/bin/env python3
"""Validate the exported rec.onnx against ground-truth test labels.

Replicates PaddleOCR's PP-OCRv5 rec preprocessing (BGR, resize to h=48 keeping
aspect ratio padded to W=320, normalise (x/255 - 0.5)/0.5) and CTC-decodes the
output using ppocrv5_en_dict.txt. Reports exact-match accuracy on a sample.
"""
import sys

import cv2
import numpy as np
import onnxruntime as ort

ONNX = "output/b6_jersey_rec/rec.onnx"
DICT = "ppocr/utils/dict/ppocrv5_en_dict.txt"
LIST = "train_data/test_list.txt"
DATA_DIR = "train_data"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
IMG_H, IMG_W = 48, 320


def load_charset():
    # CTCLabelDecode: ['blank'] + dict_lines + [' '] (use_space_char: true)
    with open(DICT, encoding="utf-8") as f:
        chars = [l.rstrip("\n") for l in f]
    return ["blank"] + chars + [" "]


def preprocess(path):
    img = cv2.imread(path)  # BGR, matches DecodeImage(img_mode=BGR)
    h, w = img.shape[:2]
    ratio = w / float(h)
    rw = min(IMG_W, int(np.ceil(IMG_H * ratio)))
    resized = cv2.resize(img, (rw, IMG_H)).astype("float32")
    resized = (resized / 255.0 - 0.5) / 0.5
    resized = resized.transpose(2, 0, 1)  # CHW
    padded = np.zeros((3, IMG_H, IMG_W), dtype="float32")
    padded[:, :, :rw] = resized
    return padded


def ctc_decode(logits, charset):
    idx = logits.argmax(axis=1)  # (T,)
    out, prev = [], -1
    for i in idx:
        if i != prev and i != 0:  # collapse repeats, drop blank(0)
            out.append(charset[i])
        prev = i
    return "".join(out)


def main():
    charset = load_charset()
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    lines = [l.rstrip("\n").split("\t") for l in open(LIST, encoding="utf-8")]
    lines = lines[:N]
    correct = 0
    misses = []
    for rel, gt in lines:
        x = preprocess(f"{DATA_DIR}/{rel}")[None]  # (1,3,48,320)
        out = sess.run(None, {inp: x})[0]  # (1, T, C)
        pred = ctc_decode(out[0], charset)
        if pred == gt:
            correct += 1
        elif len(misses) < 15:
            misses.append((rel, gt, pred))
    print(f"ONNX exact-match accuracy on {len(lines)} test samples: "
          f"{correct}/{len(lines)} = {100*correct/len(lines):.2f}%")
    if misses:
        print("sample mismatches (path, gt, pred):")
        for m in misses:
            print("  ", m)


if __name__ == "__main__":
    main()
