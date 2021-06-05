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
import logging

from tools.infer.utility import str2bool, init_args as infer_args


def init_args():
    parser = infer_args()

    # params for output
    parser.add_argument("--output", type=str, default='./output/table')
    # params for table structure
    parser.add_argument("--structure_max_len", type=int, default=488)
    parser.add_argument("--structure_max_text_length", type=int, default=100)
    parser.add_argument("--structure_max_elem_length", type=int, default=800)
    parser.add_argument("--structure_max_cell_num", type=int, default=500)
    parser.add_argument("--structure_model_dir", type=str)
    parser.add_argument("--structure_char_type", type=str, default='en')
    parser.add_argument("--structure_char_dict_path", type=str, default="../ppocr/utils/dict/table_structure_dict.txt")

    # params for layout detector
    parser.add_argument("--layout_model_dir", type=str)
    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()
