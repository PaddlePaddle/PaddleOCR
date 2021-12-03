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
    with open(json_path, "r") as fin:
        lines = fin.readlines()

    json_info = json.loads(lines[0])
    documents = json_info["documents"]
    label_info = {}
    with open(output_file, "w") as fout:
        for idx, document in enumerate(documents):
            img_info = document["img"]
            document = document["document"]
            image_path = img_info["fname"]

            label_info["height"] = img_info["height"]
            label_info["width"] = img_info["width"]

            label_info["ocr_info"] = []

            for doc in document:
                label_info["ocr_info"].append({
                    "text": doc["text"],
                    "label": doc["label"],
                    "bbox": doc["box"],
                    "id": doc["id"],
                    "linking": doc["linking"],
                    "words": doc["words"]
                })

            fout.write(image_path + "\t" + json.dumps(
                label_info, ensure_ascii=False) + "\n")

    print("===ok====")


transfer_xfun_data("./xfun/zh.val.json", "./xfun_normalize_val.json")
