# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json


def gen_rec_label(input_path, out_label):
    with open(out_label, "w") as out_file:
        with open(input_path, "r") as f:
            for line in f.readlines():
                tmp = line.strip("\n").replace(" ", "").split(",")
                img_path, label = tmp[0], tmp[1]
                label = label.replace('"', "")
                out_file.write(img_path + "\t" + label + "\n")


def gen_det_label(root_path, input_dir, out_label):
    with open(out_label, "w") as out_file:
        for label_file in os.listdir(input_dir):
            img_path = os.path.join(root_path, label_file[3:-4] + ".jpg")
            label = []
            with open(
                os.path.join(input_dir, label_file), "r", encoding="utf-8-sig"
            ) as f:
                for line in f.readlines():
                    tmp = line.strip("\n\r").replace("\xef\xbb\xbf", "").split(",")
                    points = tmp[:8]
                    s = []
                    for i in range(0, len(points), 2):
                        b = points[i : i + 2]
                        b = [int(t) for t in b]
                        s.append(b)
                    result = {"transcription": tmp[8], "points": s}
                    label.append(result)

            out_file.write(
                img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="rec",
        help="Generate rec_label or det_label, can be set rec or det",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default=".",
        help="The root directory of images.Only takes effect when mode=det ",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=".",
        help="Input_label or input path to be converted",
    )
    parser.add_argument(
        "--output_label", type=str, default="out_label.txt", help="Output file name"
    )

    args = parser.parse_args()
    if args.mode == "rec":
        print("Generate rec label")
        gen_rec_label(args.input_path, args.output_label)
    elif args.mode == "det":
        gen_det_label(args.root_path, args.input_path, args.output_label)
