# coding:utf-8
# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import contextlib
from collections import deque
import warnings
import paddle
from ..utils.tools import get_env_device
from ..transformers import ErnieCtmWordtagModel, ErnieCtmTokenizer
from .knowledge_mining import WordTagTask, NPTagTask
from .named_entity_recognition import NERTask
from .sentiment_analysis import SentaTask, SkepTask
from .lexical_analysis import LacTask
from .word_segmentation import WordSegmentationTask
from .pos_tagging import POSTaggingTask
from .text_generation import TextGenerationTask
from .poetry_generation import PoetryGenerationTask
from .question_answering import QuestionAnsweringTask
from .dependency_parsing import DDParserTask
from .text_correction import CSCTask
from .text_similarity import TextSimilarityTask
from .dialogue import DialogueTask

warnings.simplefilter(action='ignore', category=Warning, lineno=0, append=False)

TASKS = {
    "knowledge_mining": {
        "models": {
            "wordtag": {
                "task_class": WordTagTask,
                "task_flag": 'knowledge_mining-wordtag',
            },
            "nptag": {
                "task_class": NPTagTask,
                "task_flag": 'knowledge_mining-nptag',
            },
        },
        "default": {
            "model": "wordtag"
        }
    },
    "ner": {
        "models": {
            "wordtag": {
                "task_class": NERTask,
                "task_flag": 'ner-wordtag',
                "linking": False,
            }
        },
        "default": {
            "model": "wordtag"
        }
    },
    "poetry_generation": {
        "models": {
            "gpt-cpm-large-cn": {
                "task_class": PoetryGenerationTask,
                "task_flag": 'poetry_generation-gpt-cpm-large-cn',
            },
        },
        "default": {
            "model": "gpt-cpm-large-cn",
        }
    },
    "question_answering": {
        "models": {
            "gpt-cpm-large-cn": {
                "task_class": QuestionAnsweringTask,
                "task_flag": 'question_answering-gpt-cpm-large-cn',
            },
        },
        "default": {
            "model": "gpt-cpm-large-cn",
        }
    },
    "lexical_analysis": {
        "models": {
            "lac": {
                "task_class": LacTask,
                "hidden_size": 128,
                "emb_dim": 128,
                "task_flag": 'lexical_analysis-gru_crf',
            }
        },
        "default": {
            "model": "lac"
        }
    },
    "word_segmentation": {
        "models": {
            "lac": {
                "task_class": WordSegmentationTask,
                "hidden_size": 128,
                "emb_dim": 128,
                "task_flag": 'word_segmentation-gru_crf',
            }
        },
        "default": {
            "model": "lac"
        }
    },
    "pos_tagging": {
        "models": {
            "lac": {
                "task_class": POSTaggingTask,
                "hidden_size": 128,
                "emb_dim": 128,
                "task_flag": 'pos_tagging-gru_crf',
            }
        },
        "default": {
            "model": "lac"
        }
    },
    'sentiment_analysis': {
        "models": {
            "bilstm": {
                "task_class": SentaTask,
                "task_flag": 'sentiment_analysis-bilstm',
            },
            "skep_ernie_1.0_large_ch": {
                "task_class": SkepTask,
                "task_flag": 'sentiment_analysis-skep_ernie_1.0_large_ch',
            }
        },
        "default": {
            "model": "bilstm"
        }
    },
    'dependency_parsing': {
        "models": {
            "ddparser": {
                "task_class": DDParserTask,
                "task_flag": 'dependency_parsing-biaffine',
            },
            "ddparser-ernie-1.0": {
                "task_class": DDParserTask,
                "task_flag": 'dependency_parsing-ernie-1.0',
            },
            "ddparser-ernie-gram-zh": {
                "task_class": DDParserTask,
                "task_flag": 'dependency_parsing-ernie-gram-zh',
            },
        },
        "default": {
            "model": "ddparser"
        }
    },
    'text_correction': {
        "models": {
            "csc-ernie-1.0": {
                "task_class": CSCTask,
                "task_flag": "text_correction-csc-ernie-1.0"
            },
        },
        "default": {
            "model": "csc-ernie-1.0"
        }
    },
    'text_similarity': {
        "models": {
            "simbert-base-chinese": {
                "task_class": TextSimilarityTask,
                "task_flag": "text_similarity-simbert-base-chinese"
            },
        },
        "default": {
            "model": "simbert-base-chinese"
        }
    },
    'dialogue': {
        "models": {
            "plato-mini": {
                "task_class": DialogueTask,
                "task_flag": "dialogue-plato-mini"
            },
        },
        "default": {
            "model": "plato-mini"
        }
    },
}


class Taskflow(object):
    """
    The Taskflow is the end2end inferface that could convert the raw text to model result, and decode the model result to task result. The main functions as follows:
        1) Convert the raw text to task result.
        2) Convert the model to the inference model.
        3) Offer the usage and help message.
    Args:
        task (str): The task name for the Taskflow, and get the task class from the name.
        model (str, optional): The model name in the task, if set None, will use the default model.  
        device_id (int, optional): The device id for the gpu, xpu and other devices, the defalut value is 0.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task. 

    """

    def __init__(self, task, model=None, device_id=0, **kwargs):
        assert task in TASKS, "The task name:{} is not in Taskflow list, please check your task name.".format(
            task)
        self.task = task
        if model is not None:
            assert model in set(TASKS[task]['models'].keys(
            )), "The model name:{} is not in task:[{}]".format(model, task)
        else:
            model = TASKS[task]['default']['model']
        # Set the device for the task
        device = get_env_device()
        if device == 'cpu' or device_id == -1:
            paddle.set_device('cpu')
        else:
            paddle.set_device(device + ":" + str(device_id))

        self.model = model
        # Update the task config to kwargs
        config_kwargs = TASKS[self.task]['models'][self.model]
        kwargs['device_id'] = device_id
        kwargs.update(config_kwargs)
        self.kwargs = kwargs
        task_class = TASKS[self.task]['models'][self.model]['task_class']
        self.task_instance = task_class(
            model=self.model, task=self.task, **self.kwargs)
        task_list = TASKS.keys()
        Taskflow.task_list = task_list

    def __call__(self, *inputs):
        """
        The main work function in the taskflow.
        """
        results = self.task_instance(inputs)
        return results

    def help(self):
        """
        Return the task usage message.
        """
        return self.task_instance.help()

    def task_path(self):
        """
        Return the path of current task
        """
        return self.task_instance._task_path

    @staticmethod
    def tasks():
        """
        Return the available task list.
        """
        task_list = list(TASKS.keys())
        return task_list

    def from_segments(self, *inputs):
        results = self.task_instance.from_segments(inputs)
        return results

    def interactive_mode(self, max_turn):
        with self.task_instance.interactive_mode(max_turn=3):
            while True:
                human = input("[Human]:").strip()
                if human.lower() == "exit":
                    exit()
                robot = self.task_instance(human)[0]
                print("[Bot]:%s" % robot)
