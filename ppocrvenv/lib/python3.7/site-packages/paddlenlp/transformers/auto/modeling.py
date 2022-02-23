# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
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
import os
import io
import importlib
import json
from collections import OrderedDict
from paddlenlp.transformers import *
from paddlenlp.utils.downloader import COMMUNITY_MODEL_PREFIX, get_path_from_url
from paddlenlp.utils.env import MODEL_HOME
from paddlenlp.utils.log import logger

__all__ = [
    "AutoModel", "AutoModelForPretraining",
    "AutoModelForSequenceClassification", "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering", "AutoModelForMultipleChoice",
    "AutoModelForMaskedLM", "AutoModelForCausalLM", "AutoEncoder",
    "AutoDecoder", "AutoGenerator", "AutoDiscriminator",
    "AutoModelForConditionalGeneration"
]

MAPPING_NAMES = OrderedDict([
    # Base model mapping
    ("Albert", "albert"),
    ("BigBird", "bigbird"),
    ("BlenderbotSmall", "blenderbot_small"),
    ("Blenderbot", "blenderbot"),
    ("ConvBert", "convbert"),
    ("MobileBert", "mobilebert"),
    ("ChineseBert", "chinesebert"),
    ("CTRL", "ctrl"),
    ("DistilBert", "distilbert"),
    ("Electra", "electra"),
    ("Skep", "skep"),
    ("ErnieCtm", "ernie_ctm"),
    ("ErnieDoc", "ernie_doc"),
    ("ErnieGram", "ernie_gram"),
    ("ErnieGen", "ernie_gen"),
    ("Ernie", "ernie"),
    ("ErnieM", "ernie_m"),
    ("GPT", "gpt"),
    ("LayoutXLM", "layoutxlm"),
    ("LayoutLMv2", "layoutlmv2"),
    ("LayoutLM", "layoutlm"),
    ("MBart", "mbart"),
    ("MPNet", "mpnet"),
    ("NeZha", "nezha"),
    ("Roberta", "roberta"),
    ("RoFormer", "roformer"),
    ("Reformer", "reformer"),
    ("SqueezeBert", "squeezebert"),
    ("T5", "t5"),
    ("TinyBert", "tinybert"),
    ("Bert", "bert"),
    ("Bart", "bart"),
    ("UNIMO", "unimo"),
    ("UnifiedTransformer", "unified_transformer"),
    ("XLNet", "xlnet"),
])


def get_name_mapping(task='Model'):
    '''
    Task can be 'Model', 'ForPretraining', 'ForSequenceClassification', 'ForTokenClassification',
    'ForQuestionAnswering', 'ForMultipleChoice', 'ForMaskedLM', 'ForCausalLM', 'Encoder', 'Decoder',
    'Generator', 'Discriminator', 'ForConditionalGeneration'.
    '''
    NAME_MAPPING = OrderedDict()
    for key, value in MAPPING_NAMES.items():
        import_class = key + task
        new_key = key + 'Model_Import_Class'
        NAME_MAPPING[new_key] = import_class
        NAME_MAPPING[import_class] = value

    return NAME_MAPPING


def get_init_configurations():
    CONFIGURATION_MODEL_MAPPING = OrderedDict()
    for key, class_name in MAPPING_NAMES.items():
        import_class = importlib.import_module(
            f"paddlenlp.transformers.{class_name}.modeling")
        model_name = getattr(import_class, key + 'Model')
        if key == 'ErnieGen':
            name = tuple(
                model_name.ernie_gen_pretrained_init_configuration.keys())
        else:
            name = tuple(model_name.pretrained_init_configuration.keys())
        CONFIGURATION_MODEL_MAPPING[name] = key + 'Model'

    return CONFIGURATION_MODEL_MAPPING


class _BaseAutoModelClass:
    # Base class for auto models.
    _pretrained_model_dict = None
    _name_mapping = None
    _task_choice = False
    model_config_file = "model_config.json"

    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            f"{self.__class__.__name__} is designed to be instantiated "
            f"using the `{self.__class__.__name__}.from_pretrained(pretrained_model_name_or_path).`"
        )

    @classmethod
    def _from_pretrained(cls,
                         pretrained_model_name_or_path,
                         task=None,
                         *model_args,
                         **kwargs):
        if task:
            if cls._task_choice == True:
                cls._name_mapping = get_name_mapping(task)
            else:
                print('We only support task choice for AutoModel.')

        all_model_names = []
        for pretrained_model_names, model_name in cls._pretrained_model_dict.items(
        ):
            for name in pretrained_model_names:
                all_model_names.append(name)

        # From built-in pretrained models
        if pretrained_model_name_or_path in all_model_names:
            for pretrained_model_names, model_name in cls._pretrained_model_dict.items(
            ):
                # From built-in pretrained models
                for pattern in pretrained_model_names:
                    if pattern == pretrained_model_name_or_path:
                        init_class = cls._name_mapping[model_name +
                                                       '_Import_Class']
                        class_name = cls._name_mapping[init_class]
                        import_class = importlib.import_module(
                            f"paddlenlp.transformers.{class_name}.modeling")
                        model_class = getattr(import_class, init_class)
                        return model_class.from_pretrained(
                            pretrained_model_name_or_path, *model_args,
                            **kwargs)
        # From local dir path
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       cls.model_config_file)
            if os.path.exists(config_file):
                with io.open(config_file, encoding="utf-8") as f:
                    init_kwargs = json.load(f)
                # class name corresponds to this configuration
                init_class = init_kwargs.pop("init_class", None)
                if init_class:
                    for model_flag, name in MAPPING_NAMES.items():
                        if model_flag in init_class:
                            model_name = model_flag + 'Model'
                            break
                else:
                    # From pretrained_model_name_or_path
                    for model_flag, name in MAPPING_NAMES.items():
                        if name in pretrained_model_name_or_path.lower():
                            model_name = model_flag + 'Model'
                            break
                init_class = cls._name_mapping[model_name + '_Import_Class']
                class_name = cls._name_mapping[init_class]
                import_class = importlib.import_module(
                    f"paddlenlp.transformers.{class_name}.modeling")
                model_name = getattr(import_class, init_class)
                return model_name.from_pretrained(pretrained_model_name_or_path,
                                                  *model_args, **kwargs)
        # Assuming from community-contributed pretrained models
        else:
            community_config_path = os.path.join(COMMUNITY_MODEL_PREFIX,
                                                 pretrained_model_name_or_path,
                                                 cls.model_config_file)

            default_root = os.path.join(MODEL_HOME,
                                        pretrained_model_name_or_path)

            try:
                resolved_vocab_file = get_path_from_url(community_config_path,
                                                        default_root)
            except RuntimeError as err:
                logger.error(err)
                raise RuntimeError(
                    f"Can't load weights for '{pretrained_model_name_or_path}'.\n"
                    f"Please make sure that '{pretrained_model_name_or_path}' is:\n"
                    "- a correct model-identifier of built-in pretrained models,\n"
                    "- or a correct model-identifier of community-contributed pretrained models,\n"
                    "- or the correct path to a directory containing relevant modeling files(model_weights and model_config).\n"
                )

            if os.path.exists(resolved_vocab_file):
                with io.open(resolved_vocab_file, encoding="utf-8") as f:
                    init_kwargs = json.load(f)
                # class name corresponds to this configuration
                init_class = init_kwargs.pop("init_class", None)
                if init_class:
                    for model_flag, name in MAPPING_NAMES.items():
                        if model_flag in init_class:
                            model_name = model_flag + 'Model'
                            break
                else:
                    # From pretrained_model_name_or_path
                    for model_flag, name in MAPPING_NAMES.items():
                        if name in pretrained_model_name_or_path.lower():
                            model_name = model_flag + 'Model'
                            break
                init_class = cls._name_mapping[model_name + '_Import_Class']
                class_name = cls._name_mapping[init_class]
                import_class = importlib.import_module(
                    f"paddlenlp.transformers.{class_name}.modeling")
                model_name = getattr(import_class, init_class)
                return model_name.from_pretrained(pretrained_model_name_or_path,
                                                  *model_args, **kwargs)


class AutoModel(_BaseAutoModelClass):
    """
    AutoClass can help you automatically retrieve the relevant model given the provided
    pretrained weights/vocabulary.
    AutoModel is a generic model class that will be instantiated as one of the base model classes
    when created with the from_pretrained() classmethod.
    """
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('Model')
    _task_choice = True

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        task=None,
                        *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoModel`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model or dir path
                to load from. The string can be:

                - Name of a built-in pretrained model
                - Name of a community-contributed pretrained model.
                - Local directory path which contains model weights file("model_state.pdparams")
                  and model config file ("model_config.json").
            task (str): Specify a downstream task. Task can be 'Model', 'ForPretraining',
                'ForSequenceClassification', 'ForTokenClassification', 'ForQuestionAnswering',
                'ForMultipleChoice', 'ForMaskedLM', 'ForCausalLM', 'Encoder', 'Decoder',
                'Generator', 'Discriminator', 'ForConditionalGeneration'.
                We only support specify downstream tasks in AutoModel. Defaults to `None`.
            *args (tuple): Position arguments for model `__init__`. If provided,
                use these as position argument values for model initialization.
            **kwargs (dict): Keyword arguments for model `__init__`. If provided,
                use these to update pre-defined keyword argument values for model
                initialization. If the keyword is in `__init__` argument names of
                base model, update argument values of the base model; else update
                argument values of derived model.

        Returns:
            PretrainedModel: An instance of `AutoModel`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModel

                # Name of built-in pretrained model
                model = AutoModel.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModel'>

                # Name of community-contributed pretrained model
                model = AutoModel.from_pretrained('yingyibiao/bert-base-uncased-sst-2-finetuned')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModel'>

                # Load from local directory path
                model = AutoModel.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModel'>

                # choose task
                model = AutoModel.from_pretrained('bert-base-uncased', task='ForPretraining')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertForPretraining'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, task,
                                    *model_args, **kwargs)


class AutoModelForPretraining(_BaseAutoModelClass):
    """
    AutoModelForPretraining.
    """
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('ForPretraining')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoModelForPretraining`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForPretraining`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModelForPretraining

                # Name of built-in pretrained model
                model = AutoModelForPretraining.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForPretraining'>

                # Name of community-contributed pretrained model
                model = AutoModelForPretraining.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForPretraining'>

                # Load from local directory path
                model = AutoModelForPretraining.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForPretraining'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    '''
    AutoModelForSequenceClassification.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('ForSequenceClassification')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoModelForSequenceClassification`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForSequenceClassification`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModelForSequenceClassification

                # Name of built-in pretrained model
                model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForSequenceClassification'>

                # Name of community-contributed pretrained model
                model = AutoModelForSequenceClassification.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForSequenceClassification'>

                # Load from local directory path
                model = AutoModelForSequenceClassification.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForSequenceClassification'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoModelForTokenClassification(_BaseAutoModelClass):
    '''
    AutoModelForTokenClassification.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('ForTokenClassification')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoModelForTokenClassification`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForTokenClassification`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModelForTokenClassification

                # Name of built-in pretrained model
                model = AutoModelForTokenClassification.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForTokenClassification'>

                # Name of community-contributed pretrained model
                model = AutoModelForTokenClassification.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForTokenClassification'>

                # Load from local directory path
                model = AutoModelForTokenClassification.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForTokenClassification'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    '''
    AutoModelForQuestionAnswering.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('ForQuestionAnswering')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoModelForQuestionAnswering`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForQuestionAnswering`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModelForQuestionAnswering

                # Name of built-in pretrained model
                model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForQuestionAnswering'>

                # Name of community-contributed pretrained model
                model = AutoModelForQuestionAnswering.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForQuestionAnswering'>

                # Load from local directory path
                model = AutoModelForQuestionAnswering.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForQuestionAnswering'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    '''
    AutoModelForMultipleChoice.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('ForMultipleChoice')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoModelForMultipleChoice`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForMultipleChoice`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModelForMultipleChoice

                # Name of built-in pretrained model
                model = AutoModelForMultipleChoice.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForMultipleChoice'>

                # Name of community-contributed pretrained model
                model = AutoModelForMultipleChoice.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForMultipleChoice'>

                # Load from local directory path
                model = AutoModelForMultipleChoice.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForMultipleChoice'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoModelForMaskedLM(_BaseAutoModelClass):
    '''
    AutoModelForMaskedLM.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('ForMaskedLM')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoModelForMaskedLM`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForMaskedLM`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModelForMaskedLM

                # Name of built-in pretrained model
                model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForMaskedLM'>

                # Name of community-contributed pretrained model
                model = AutoModelForMaskedLM.from_pretrained('iverxin/bert-base-japanese')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForMaskedLM'>

                # Load from local directory path
                model = AutoModelForMaskedLM.from_pretrained('./my_bert/')
                print(type(model))
                # <class 'paddlenlp.transformers.bert.modeling.BertModelForMaskedLM'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoModelForCausalLM(_BaseAutoModelClass):
    '''
    AutoModelForCausalLM.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('ForCausalLM')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoModelForCausalLM`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForCausalLM`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModelForCausalLM

                # Name of built-in pretrained model
                model = AutoModelForCausalLM.from_pretrained('gpt2-en')
                print(type(model))
                # <class 'paddlenlp.transformers.gpt.modeling.GPTLMHeadModel'>

                # Name of community-contributed pretrained model
                model = AutoModelForCausalLM.from_pretrained('junnyu/distilgpt2')
                print(type(model))
                # <class 'paddlenlp.transformers.gpt.modeling.GPTLMHeadModel'>

                # Load from local directory path
                model = AutoModelForCausalLM.from_pretrained('./my_gpt/')
                print(type(model))
                # <class 'paddlenlp.transformers.gpt.modeling.GPTLMHeadModel'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoEncoder(_BaseAutoModelClass):
    '''
    AutoEncoder.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('Encoder')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoEncoder`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoEncoder`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoEncoder

                # Name of built-in pretrained model
                model = AutoEncoder.from_pretrained('bart-base',vocab_size=20000)
                print(type(model))
                # <class 'paddlenlp.transformers.bart.modeling.BartEncoder'>

                # Load from local directory path
                model = AutoEncoder.from_pretrained('./my_bart/')
                print(type(model))
                # <class 'paddlenlp.transformers.bart.modeling.BartEncoder'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoDecoder(_BaseAutoModelClass):
    '''
    AutoDecoder.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('Decoder')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoDecoder`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoDecoder`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoDecoder

                # Name of built-in pretrained model
                model = AutoDecoder.from_pretrained('bart-base', vocab_size=20000)
                print(type(model))
                # <class 'paddlenlp.transformers.bart.modeling.BartEncoder'>

                # Load from local directory path
                model = AutoDecoder.from_pretrained('./my_bart/')
                print(type(model))
                # <class 'paddlenlp.transformers.bart.modeling.BartEncoder'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoGenerator(_BaseAutoModelClass):
    '''
    AutoGenerator.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('Generator')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoGenerator`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoGenerator`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoGenerator

                # Name of built-in pretrained model
                model = AutoGenerator.from_pretrained('electra-small')
                print(type(model))
                # <class 'paddlenlp.transformers.electra.modeling.ElectraGenerator'>

                # Name of community-contributed pretrained model
                model = AutoGenerator.from_pretrained('junnyu/hfl-chinese-legal-electra-small-generator')
                print(type(model))
                # <class 'paddlenlp.transformers.electra.modeling.ElectraGenerator'>

                # Load from local directory path
                model = AutoGenerator.from_pretrained('./my_electra/')
                print(type(model))
                # <class 'paddlenlp.transformers.electra.modeling.ElectraGenerator'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoDiscriminator(_BaseAutoModelClass):
    '''
    AutoDiscriminator.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('Discriminator')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoDiscriminator`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoDiscriminator`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoDiscriminator

                # Name of built-in pretrained model
                model = AutoDiscriminator.from_pretrained('electra-small')
                print(type(model))
                # <class 'paddlenlp.transformers.electra.modeling.ElectraDiscriminator'>

                # Name of community-contributed pretrained model
                model = AutoDiscriminator.from_pretrained('junnyu/hfl-chinese-legal-electra-small-generator')
                print(type(model))
                # <class 'paddlenlp.transformers.electra.modeling.ElectraDiscriminator'>

                # Load from local directory path
                model = AutoDiscriminator.from_pretrained('./my_electra/')
                print(type(model))
                # <class 'paddlenlp.transformers.electra.modeling.ElectraDiscriminator'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)


class AutoModelForConditionalGeneration(_BaseAutoModelClass):
    '''
    AutoModelForConditionalGeneration.
    '''
    CONFIGURATION_MODEL_MAPPING = get_init_configurations()
    _pretrained_model_dict = CONFIGURATION_MODEL_MAPPING
    _name_mapping = get_name_mapping('ForConditionalGeneration')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args,
                        **kwargs):
        '''
        Creates an instance of `AutoModelForConditionalGeneration`. Model weights are loaded
        by specifying name of a built-in pretrained model, or a community contributed model,
        or a local file directory path.

        Args:
            pretrained_model_name_or_path (str): See :class:`AutoModel`.
            *args (tuple): See :class:`AutoModel`.
            **kwargs (dict): See :class:`AutoModel`.

        Returns:
            PretrainedModel: An instance of `AutoModelForConditionalGeneration`.

        Example:
            .. code-block::

                from paddlenlp.transformers import AutoModelForConditionalGeneration

                # Name of built-in pretrained model
                model = AutoModelForConditionalGeneration.from_pretrained('bart-base')
                print(type(model))
                # <class 'paddlenlp.transformers.bart.modeling.BartForConditionalGeneration'>


                # Load from local directory path
                model = AutoModelForConditionalGeneration.from_pretrained('./my_bart/')
                print(type(model))
                # <class 'paddlenlp.transformers.bart.modeling.BartForConditionalGeneration'>
        '''
        return cls._from_pretrained(pretrained_model_name_or_path, *model_args,
                                    **kwargs)
