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

import yaml
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os.path
import logging
logging.basicConfig(level=logging.INFO)

support_list = {
    'it':'italian', 'xi':'spanish', 'pu':'portuguese', 'ru':'russian', 'ar':'arabic',
    'ta':'tamil', 'ug':'uyghur', 'fa':'persian', 'ur':'urdu', 'rs':'serbian latin',
    'oc':'occitan', 'rsc':'serbian cyrillic', 'bg':'bulgarian', 'uk':'ukranian', 'be':'belarusian',
    'te':'telugu', 'ka':'kannada', 'chinese_cht':'chinese tradition','hi':'hindi','mr':'marathi',
    'ne':'nepali',
}
assert(
    os.path.isfile("./rec_multi_language_lite_train.yml")
    ),"Loss basic configuration file rec_multi_language_lite_train.yml.\
You can download it from \
https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/configs/rec/multi_language/"
 
global_config = yaml.load(open("./rec_multi_language_lite_train.yml", 'rb'), Loader=yaml.Loader)
project_path = os.path.abspath(os.path.join(os.getcwd(), "../../../"))

class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(
            formatter_class=RawDescriptionHelpFormatter)
        self.add_argument(
            "-o", "--opt", nargs='+', help="set configuration options")
        self.add_argument(
            "-l", "--language", nargs='+', help="set language type, support {}".format(support_list))
        self.add_argument(
            "--train",type=str,help="you can use this command to change the train dataset default path")
        self.add_argument(
            "--val",type=str,help="you can use this command to change the eval dataset default path")
        self.add_argument(
            "--dict",type=str,help="you can use this command to change the dictionary default path")
        self.add_argument(
            "--data_dir",type=str,help="you can use this command to change the dataset default root path")

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        args.opt = self._parse_opt(args.opt)
        args.language = self._set_language(args.language)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split('=')
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config

    def _set_language(self, type):
        assert(type),"please use -l or --language to choose language type"
        assert(
                type[0] in support_list.keys()
               ),"the sub_keys(-l or --language) can only be one of support list: \n{},\nbut get: {}, " \
                 "please check your running command".format(support_list, type)
        global_config['Global']['character_dict_path'] = 'ppocr/utils/dict/{}_dict.txt'.format(type[0])
        global_config['Global']['save_model_dir'] = './output/rec_{}_lite'.format(type[0])
        global_config['Train']['dataset']['label_file_list'] = ["train_data/{}_train.txt".format(type[0])]
        global_config['Eval']['dataset']['label_file_list'] = ["train_data/{}_val.txt".format(type[0])]
        global_config['Global']['character_type'] = type[0]
        assert(
                os.path.isfile(os.path.join(project_path,global_config['Global']['character_dict_path']))
              ),"Loss default dictionary file {}_dict.txt.You can download it from \
https://github.com/PaddlePaddle/PaddleOCR/tree/dygraph/ppocr/utils/dict/".format(type[0])
        return type[0]


def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
                    
def loss_file(path):
    assert(
            os.path.exists(path)
          ),"There is no such file:{},Please do not forget to put in the specified file".format(path)

        
if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    merge_config(FLAGS.opt)
    save_file_path = 'rec_{}_lite_train.yml'.format(FLAGS.language)
    if os.path.isfile(save_file_path):
        os.remove(save_file_path)
        
    if FLAGS.train:
        global_config['Train']['dataset']['label_file_list'] = [FLAGS.train]
        train_label_path = os.path.join(project_path,FLAGS.train)
        loss_file(train_label_path)
    if FLAGS.val:
        global_config['Eval']['dataset']['label_file_list'] = [FLAGS.val]
        eval_label_path = os.path.join(project_path,FLAGS.val)
        loss_file(Eval_label_path)
    if FLAGS.dict:
        global_config['Global']['character_dict_path'] = FLAGS.dict
        dict_path = os.path.join(project_path,FLAGS.dict)
        loss_file(dict_path)
    if FLAGS.data_dir:
        global_config['Eval']['dataset']['data_dir'] = FLAGS.data_dir
        global_config['Train']['dataset']['data_dir'] = FLAGS.data_dir
        data_dir = os.path.join(project_path,FLAGS.data_dir)
        loss_file(data_dir)
        
    with open(save_file_path, 'w') as f:
        yaml.dump(dict(global_config), f, default_flow_style=False, sort_keys=False)
    logging.info("Project path is          :{}".format(project_path))
    logging.info("Train list path set to   :{}".format(global_config['Train']['dataset']['label_file_list'][0]))
    logging.info("Eval list path set to    :{}".format(global_config['Eval']['dataset']['label_file_list'][0]))
    logging.info("Dataset root path set to :{}".format(global_config['Eval']['dataset']['data_dir']))
    logging.info("Dict path set to         :{}".format(global_config['Global']['character_dict_path']))
    logging.info("Config file set to       :configs/rec/multi_language/{}".format(save_file_path))
