# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import json
import os


def poly_to_string(poly):
    if len(poly.shape) > 1:
        poly = np.array(poly).flatten()

    string = "\t".join(str(i) for i in poly)
    return string


def convert_label(label_dir, mode="gt", save_dir="./save_results/"):
    if not os.path.exists(label_dir):
        raise ValueError(f"The file {label_dir} does not exist!")

    assert label_dir != save_dir, "hahahhaha"

    label_file = open(label_dir, 'r')
    data = label_file.readlines()

    gt_dict = {}

    for line in data:
        try:
            tmp = line.split('\t')
            assert len(tmp) == 2, ""
        except:
            tmp = line.strip().split('    ')

        gt_lists = []

        if tmp[0].split('/')[0] is not None:
            img_path = tmp[0]
            anno = json.loads(tmp[1])
            gt_collect = []
            for dic in anno:
                #txt = dic['transcription'].replace(' ', '')  # ignore blank
                txt = dic['transcription']
                if 'score' in dic and float(dic['score']) < 0.5:
                    continue
                if u'\u3000' in txt: txt = txt.replace(u'\u3000', u' ')
                #while ' ' in txt:
                #    txt = txt.replace(' ', '')
                poly = np.array(dic['points']).flatten()
                if txt == "###":
                    txt_tag = 1  ## ignore 1
                else:
                    txt_tag = 0
                if mode == "gt":
                    gt_label = poly_to_string(poly) + "\t" + str(
                        txt_tag) + "\t" + txt + "\n"
                else:
                    gt_label = poly_to_string(poly) + "\t" + txt + "\n"

                gt_lists.append(gt_label)

            gt_dict[img_path] = gt_lists
        else:
            continue

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_name in gt_dict.keys():
        save_name = img_name.split("/")[-1]
        save_file = os.path.join(save_dir, save_name + ".txt")
        with open(save_file, "w") as f:
            f.writelines(gt_dict[img_name])

    print("The convert label saved in {}".format(save_dir))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="args")
    parser.add_argument("--label_path", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--mode", type=str, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    convert_label(args.label_path, args.mode, args.save_folder)
