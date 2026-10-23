"""Crop XFUND German word boxes into PaddleOCR recognition training data.

Run from the repo root:
    python train/prepare_data.py --src <path to XFUND/de> --out train/data

Output: <out>/train/*.jpg, <out>/val/*.jpg, <out>/train_list.txt, <out>/eval_list.txt
"""
import argparse
import json
import os

from PIL import Image

MAX_LEN = 25  # must match max_text_length in latin_de_finetune.yml


def convert(src, out, split, list_name):
    with open(f"{src}/de.{split}.json", encoding="utf-8") as f:
        ann = json.load(f)
    os.makedirs(f"{out}/{split}", exist_ok=True)
    lines = []
    for doc in ann["documents"]:
        fname = doc["img"]["fname"]
        img = Image.open(f"{src}/de.{split}/{fname}").convert("RGB")
        stem = os.path.splitext(fname)[0]
        for item in doc["document"]:
            for i, w in enumerate(item["words"]):
                text = w["text"].strip()
                x1, y1, x2, y2 = w["box"]
                if not text or len(text) > MAX_LEN or x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                name = f"{split}/{stem}_{item['id']}_{i}.jpg"
                img.crop((x1, y1, x2, y2)).save(f"{out}/{name}", quality=95)
                lines.append(f"{name}\t{text}")
    with open(f"{out}/{list_name}", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{split}: {len(lines)} crops")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="train_data/XFUND/de")
    p.add_argument("--out", default="train/data")
    a = p.parse_args()
    convert(a.src, a.out, "train", "train_list.txt")
    convert(a.src, a.out, "val", "eval_list.txt")
