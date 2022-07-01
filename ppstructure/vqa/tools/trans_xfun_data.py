# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import json


def transfer_xfun_data(json_path=None, output_file=None):
    with open(json_path, "r", encoding='utf-8') as fin:
        lines = fin.readlines()

    json_info = json.loads(lines[0])
    documents = json_info["documents"]
    with open(output_file, "w", encoding='utf-8') as fout:
        for idx, document in enumerate(documents):
            label_info = []
            img_info = document["img"]
            document = document["document"]
            image_path = img_info["fname"]

            for doc in document:
                x1, y1, x2, y2 = doc["box"]
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                label_info.append({
                    "transcription": doc["text"],
                    "label": doc["label"],
                    "points": points,
                    "id": doc["id"],
                    "linking": doc["linking"]
                })

            fout.write(image_path + "\t" + json.dumps(
                label_info, ensure_ascii=False) + "\n")

    print("===ok====")


def parser_args():
    import argparse
    parser = argparse.ArgumentParser(description="args for paddleserving")
    parser.add_argument(
        "--ori_gt_path", type=str, required=True, help='origin xfun gt path')
    parser.add_argument(
        "--output_path", type=str, required=True, help='path to save')
    args = parser.parse_args()
    return args


args = parser_args()
transfer_xfun_data(args.ori_gt_path, args.output_path)
