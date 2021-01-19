import yaml
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import os.path

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
            "--dataset_root_path",type=str,help="you can use this command to change the dataset default root path")

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
        assert(type),"please use -t or --type to choose language type"
        assert(
                type[0] in support_list.keys()
               ),"the sub_keys(-t or --type) can only be one of support list: \n{},\nbut get: {}, " \
                 "please check your running command".format(support_list, type)
        global_config['Global']['character_dict_path'] = 'ppocr/utils/dict/{}_dict.txt'.format(type[0])
        global_config['Global']['save_model_dir'] = './output/rec_{}_lite'.format(type[0])
        global_config['Train']['dataset']['label_file_list'] = ["train_data/{}_train.txt".format(type[0])]
        global_config['Eval']['dataset']['label_file_list'] = ["train_data/{}_val.txt".format(type[0])]
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

if __name__ == '__main__':
    FLAGS = ArgsParser().parse_args()
    merge_config(FLAGS.opt)
    if FLAGS.train:
        global_config['Train']['dataset']['label_file_list'] = [FLAGS.train]
    if FLAGS.val:
        global_config['Eval']['dataset']['label_file_list'] = [FLAGS.val]
    if FLAGS.dict:
        global_config['Global']['character_dict_path'] = FLAGS.dict
    if FLAGS.dataset_root_path:
        global_config['Eval']['dataset']['data_dir'] = FLAGS.dataset_root_path
        global_config['Train']['dataset']['data_dir'] = FLAGS.dataset_root_path

    save_file_path = 'rec_{}_lite_train.yml'.format(FLAGS.language)
    if os.path.isfile(save_file_path):
        os.remove(save_file_path)
    with open(save_file_path, 'w') as f:
        yaml.dump(dict(global_config), f, default_flow_style=False, sort_keys=False)
    print("Train list path set to   :{}".format(global_config['Train']['dataset']['label_file_list'][0]))
    print("Eval list path set to    :{}".format(global_config['Eval']['dataset']['label_file_list'][0]))
    print("Dataset root path set to :{}".format(global_config['Eval']['dataset']['data_dir']))
    print("Dict path set to         :{}".format(global_config['Global']['character_dict_path']))
    print("Config file set to       :configs/rec/multi_language/{}".format(save_file_path))
