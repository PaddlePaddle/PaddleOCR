# -*- coding: utf-8 -*-
# @Time : 2023/3/31 11:52
# @Author : JianjinL
# @eMail : jianjinlv@163.com
# @File : run
# @Software : PyCharm
# @Dscription: 批量执行测试

import os

# 待执行的任务列表
tasks = [
    # {"language": "zh", "api": ["peredoc", ], "dataset_type": ["view"], "filter": "true"},
    # {"language": "ms", "api": ["peredoc", ], "dataset_type": ["doc", "view"], "filter": "true"},
    # {"language": "id", "api": ["peredoc", ], "dataset_type": ["doc", "view"], "filter": "true"},
    # {"language": "vi", "api": ["peredoc", ], "dataset_type": ["doc", "view"], "filter": "true"},
    # {"language": "ar", "api": ["peredoc", ], "dataset_type": ["doc", "view"], "filter": "true"},
    # {"language": "ug", "api": ["peredoc", ], "dataset_type": ["doc", "view"], "filter": "true"},
    # {"language": "ru", "api": ["peredoc", ], "dataset_type": ["doc", "view"], "filter": "true"},
    {"language": "kk", "api": ["peredoc", ], "dataset_type": ["view"], "filter": "true"},
    # {"language": "hi", "api": ["peredoc", ], "dataset_type": ["doc", ], "filter": "true"},
    # {"language": "th", "api": ["peredoc", ], "dataset_type": ["doc", "view"], "filter": "true"},
    # {"language": "my", "api": ["peredoc", ], "dataset_type": ["doc", "view"], "filter": "true"},
    # {"language": "bo", "api": ["peredoc", ], "dataset_type": ["doc", "view"], "filter": "true"},
]

for task in tasks:
    root = "C:/Users/lvjia/Pictures/dataset/test"
    language = task["language"]
    filter = task["filter"]
    for api in task["api"]:
        for dataset_type in task["dataset_type"]:
            dataset = os.path.join(root, "2023/image", task["language"] + "_" + dataset_type).replace("\\", "/")
            output_dir = os.path.join(root, "filter", task["language"] + "_" + dataset_type).replace("\\", "/")
            print(f"开始执行，语言：{language}, 类型: {dataset_type} ......")
            line = f"python tools/eval2api/eval_api.py --dataset {dataset} --language {language} --api {api} --filter {filter} --output_dir {output_dir} --pred_label true"
            os.system(
                f"python tools/eval2api/eval_api.py --dataset {dataset} --language {language} --api {api} --filter {filter} --output_dir {output_dir} --pred_label true")
            print("+++++++++++最终精度++++++++++")
            os.system(
                f"python tools/eval2api/eval_api.py --dataset {output_dir} --language {language} --api {api}")
