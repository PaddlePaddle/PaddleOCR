# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2021 The HuggingFace Inc. team.
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

from .. import register_base_model
import math
from dataclasses import dataclass
from dataclasses import fields
from collections import OrderedDict
import numpy as np
from paddle import nn
from paddle.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, LayerNorm
from .. import PretrainedModel as PreTrainedModel
import paddle
import os
import json
import copy
import logging
from collections import Iterable

FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "funnel-transformer/small",  # B4-4-4H768
    "funnel-transformer/small-base",  # B4-4-4H768, no decoder
    "funnel-transformer/medium",  # B6-3x2-3x2H768
    "funnel-transformer/medium-base",  # B6-3x2-3x2H768, no decoder
    "funnel-transformer/intermediate",  # B6-6-6H768
    "funnel-transformer/intermediate-base",  # B6-6-6H768, no decoder
    "funnel-transformer/large",  # B8-8-8H1024
    "funnel-transformer/large-base",  # B8-8-8H1024, no decoder
    "funnel-transformer/xlarge-base",  # B10-10-10H1024
    "funnel-transformer/xlarge",  # B10-10-10H1024, no decoder
]

__all__ = [
    "FunnelModel", "FunnelForSequenceClassification",
    "FunnelForTokenClassification", "FunnelForQuestionAnswering"
]
dtype_float = paddle.get_default_dtype()
logger = logging.getLogger(__name__)
CONFIG_NAME = "model_config.json"
INF = 1e6


class PretrainedConfig(dict):
    r"""
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    Note:
        A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to
        initialize a model does **not** load the model weights. It only affects the model's configuration.

    Class attributes (overridden by derived classes)

        - **model_type** (:obj:`str`) -- An identifier for the model type, serialized into the JSON file, and used to
          recreate the correct object in :class:`~hf_paddle.AutoConfig`.
        - **is_composition** (:obj:`bool`) -- Whether the config class is composed of multiple sub-configs. In this
          case the config has to be initialized from two or more configs of type
          :class:`~hf_paddle.PretrainedConfig` like: :class:`~hf_paddle.EncoderDecoderConfig` or
          :class:`~RagConfig`.
        - **keys_to_ignore_at_inference** (:obj:`List[str]`) -- A list of keys to ignore by default when looking at
          dictionary outputs of the model during inference.

    Common attributes (present in all subclasses)

        - **vocab_size** (:obj:`int`) -- The number of tokens in the vocabulary, which is also the first dimension of
          the embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
        - **hidden_size** (:obj:`int`) -- The hidden size of the model.
        - **num_attention_heads** (:obj:`int`) -- The number of attention heads used in the multi-head attention layers
          of the model.
        - **num_hidden_layers** (:obj:`int`) -- The number of blocks in the model.

    Args:
        name_or_path (:obj:`str`, `optional`, defaults to :obj:`""`):
            Store the string that was passed to :func:`~hf_paddle.PreTrainedModel.from_pretrained` or
            :func:`~hf_paddle.TFPreTrainedModel.from_pretrained` as ``pretrained_model_name_or_path`` if the
            configuration was created with such a method.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should return all hidden-states.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the model should returns all attentions.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return a :class:`~hf_paddle.file_utils.ModelOutput` instead of a plain
            tuple.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as an encoder/decoder or not.
        is_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether the model is used as decoder or not (in which case it's used as an encoder).
        add_cross_attention (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether cross-attention layers should be added to the model. Note, this option is only relevant for models
            that can be used as decoder models within the `:class:~hf_paddle.EncoderDecoderModel` class, which
            consists of all models in ``AUTO_MODELS_FOR_CAUSAL_LM``.
        tie_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`False`)
            Whether all encoder weights should be tied to their equivalent decoder weights. This requires the encoder
            and decoder model to have the exact same parameter names.
        prune_heads (:obj:`Dict[int, List[int]]`, `optional`, defaults to :obj:`{}`):
            Pruned heads of the model. The keys are the selected layer indices and the associated values, the list of
            heads to prune in said layer.

            For instance ``{1: [0, 2], 2: [2, 3]}`` will prune heads 0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        chunk_size_feed_forward (:obj:`int`, `optional`, defaults to :obj:`0`):
            The chunk size of all feed forward layers in the residual attention blocks. A chunk size of :obj:`0` means
            that the feed forward layer is not chunked. A chunk size of n means that the feed forward layer processes
            :obj:`n` < sequence_length embeddings at a time. For more information on feed forward chunking, see `How
            does Feed Forward Chunking work? <../glossary.html#feed-forward-chunking>`__ .

    Parameters for sequence generation

        - **max_length** (:obj:`int`, `optional`, defaults to 20) -- Maximum length that will be used by default in the
          :obj:`generate` method of the model.
        - **min_length** (:obj:`int`, `optional`, defaults to 10) -- Minimum length that will be used by default in the
          :obj:`generate` method of the model.
        - **do_sample** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by default in the
          :obj:`generate` method of the model. Whether or not to use sampling ; use greedy decoding otherwise.
        - **early_stopping** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Flag that will be used by default
          in the :obj:`generate` method of the model. Whether to stop the beam search when at least ``num_beams``
          sentences are finished per batch or not.
        - **num_beams** (:obj:`int`, `optional`, defaults to 1) -- Number of beams for beam search that will be used by
          default in the :obj:`generate` method of the model. 1 means no beam search.
        - **num_beam_groups** (:obj:`int`, `optional`, defaults to 1) -- Number of groups to divide :obj:`num_beams`
          into in order to ensure diversity among different groups of beams that will be used by default in the
          :obj:`generate` method of the model. 1 means no group beam search.
        - **diversity_penalty** (:obj:`float`, `optional`, defaults to 0.0) -- Value to control diversity for group
          beam search. that will be used by default in the :obj:`generate` method of the model. 0 means no diversity
          penalty. The higher the penalty, the more diverse are the outputs.
        - **temperature** (:obj:`float`, `optional`, defaults to 1) -- The value used to module the next token
          probabilities that will be used by default in the :obj:`generate` method of the model. Must be strictly
          positive.
        - **top_k** (:obj:`int`, `optional`, defaults to 50) -- Number of highest probability vocabulary tokens to keep
          for top-k-filtering that will be used by default in the :obj:`generate` method of the model.
        - **top_p** (:obj:`float`, `optional`, defaults to 1) -- Value that will be used by default in the
          :obj:`generate` method of the model for ``top_p``. If set to float < 1, only the most probable tokens with
          probabilities that add up to ``top_p`` or higher are kept for generation.
        - **repetition_penalty** (:obj:`float`, `optional`, defaults to 1) -- Parameter for repetition penalty that
          will be used by default in the :obj:`generate` method of the model. 1.0 means no penalty.
        - **length_penalty** (:obj:`float`, `optional`, defaults to 1) -- Exponential penalty to the length that will
          be used by default in the :obj:`generate` method of the model.
        - **no_repeat_ngram_size** (:obj:`int`, `optional`, defaults to 0) -- Value that will be used by default in the
          :obj:`generate` method of the model for ``no_repeat_ngram_size``. If set to int > 0, all ngrams of that size
          can only occur once.
        - **encoder_no_repeat_ngram_size** (:obj:`int`, `optional`, defaults to 0) -- Value that will be used by
          default in the :obj:`generate` method of the model for ``encoder_no_repeat_ngram_size``. If set to int > 0,
          all ngrams of that size that occur in the ``encoder_input_ids`` cannot occur in the ``decoder_input_ids``.
        - **bad_words_ids** (:obj:`List[int]`, `optional`) -- List of token ids that are not allowed to be generated
          that will be used by default in the :obj:`generate` method of the model. In order to get the tokens of the
          words that should not appear in the generated text, use :obj:`tokenizer.encode(bad_word,
          add_prefix_space=True)`.
        - **num_return_sequences** (:obj:`int`, `optional`, defaults to 1) -- Number of independently computed returned
          sequences for each element in the batch that will be used by default in the :obj:`generate` method of the
          model.
        - **output_scores** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether the model should return the
          logits when used for generation
        - **return_dict_in_generate** (:obj:`bool`, `optional`, defaults to :obj:`False`) -- Whether the model should
          return a :class:`~hf_paddle.file_utils.ModelOutput` instead of a :obj:`paddle.Tensor`
        - **forced_bos_token_id** (:obj:`int`, `optional`) -- The id of the token to force as the first generated token
          after the :obj:`decoder_start_token_id`. Useful for multilingual models like :doc:`mBART
          <../model_doc/mbart>` where the first generated token needs to be the target language token.
        - **forced_eos_token_id** (:obj:`int`, `optional`) -- The id of the token to force as the last generated token
          when :obj:`max_length` is reached.
        - **remove_invalid_values** (:obj:`bool`, `optional`) -- Whether to remove possible `nan` and `inf` outputs of
          the model to prevent the generation method to crash. Note that using ``remove_invalid_values`` can slow down
          generation.


    Parameters for fine-tuning tasks

        - **architectures** (:obj:`List[str]`, `optional`) -- Model architectures that can be used with the model
          pretrained weights.
        - **finetuning_task** (:obj:`str`, `optional`) -- Name of the task used to fine-tune the model. This can be
          used when converting from an original (TensorFlow or PyTorch) checkpoint.
        - **id2label** (:obj:`Dict[int, str]`, `optional`) -- A map from index (for instance prediction index, or
          target index) to label.
        - **label2id** (:obj:`Dict[str, int]`, `optional`) -- A map from label to index for the model.
        - **num_labels** (:obj:`int`, `optional`) -- Number of labels to use in the last layer added to the model,
          typically for a classification task.
        - **task_specific_params** (:obj:`Dict[str, Any]`, `optional`) -- Additional keyword arguments to store for the
          current task.
        - **problem_type** (:obj:`str`, `optional`) -- Problem type for :obj:`XxxForSequenceClassification` models. Can
          be one of (:obj:`"regression"`, :obj:`"single_label_classification"`, :obj:`"multi_label_classification"`).
          Please note that this parameter is only available in the following models: `AlbertForSequenceClassification`,
          `BertForSequenceClassification`, `BigBirdForSequenceClassification`, `ConvBertForSequenceClassification`,
          `DistilBertForSequenceClassification`, `ElectraForSequenceClassification`, `FunnelForSequenceClassification`,
          `LongformerForSequenceClassification`, `MobileBertForSequenceClassification`,
          `ReformerForSequenceClassification`, `RobertaForSequenceClassification`,
          `SqueezeBertForSequenceClassification`, `XLMForSequenceClassification` and `XLNetForSequenceClassification`.

    Parameters linked to the tokenizer

        - **tokenizer_class** (:obj:`str`, `optional`) -- The name of the associated tokenizer class to use (if none is
          set, will use the tokenizer associated to the model by default).
        - **prefix** (:obj:`str`, `optional`) -- A specific prompt that should be added at the beginning of each text
          before calling the model.
        - **bos_token_id** (:obj:`int`, `optional`)) -- The id of the `beginning-of-stream` token.
        - **pad_token_id** (:obj:`int`, `optional`)) -- The id of the `padding` token.
        - **eos_token_id** (:obj:`int`, `optional`)) -- The id of the `end-of-stream` token.
        - **sep_token_id** (:obj:`int`, `optional`)) -- The id of the `separation` token.

    """
    model_type: str = ""
    is_composition: bool = False

    def __init__(self, **kwargs):
        # Attributes with defaults
        self.return_dict = kwargs.pop("return_dict", True)
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        self.output_attentions = kwargs.pop("output_attentions", False)
        self.use_bfloat16 = kwargs.pop("use_bfloat16", False)
        self.pruned_heads = kwargs.pop("pruned_heads", {})
        self.tie_word_embeddings = kwargs.pop(
            "tie_word_embeddings", True
        )  # Whether input and output word embeddings should be tied for all MLM, LM and Seq2Seq models.

        # Is decoder is used in encoder-decoder models to differentiate encoder from decoder
        self.is_encoder_decoder = kwargs.pop("is_encoder_decoder", False)
        self.is_decoder = kwargs.pop("is_decoder", False)
        self.add_cross_attention = kwargs.pop("add_cross_attention", False)
        self.tie_encoder_decoder = kwargs.pop("tie_encoder_decoder", False)

        # Parameters for sequence generation
        self.max_length = kwargs.pop("max_length", 20)
        self.min_length = kwargs.pop("min_length", 0)
        self.do_sample = kwargs.pop("do_sample", False)
        self.early_stopping = kwargs.pop("early_stopping", False)
        self.num_beams = kwargs.pop("num_beams", 1)
        self.num_beam_groups = kwargs.pop("num_beam_groups", 1)
        self.diversity_penalty = kwargs.pop("diversity_penalty", 0.0)
        self.temperature = kwargs.pop("temperature", 1.0)
        self.top_k = kwargs.pop("top_k", 50)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        self.length_penalty = kwargs.pop("length_penalty", 1.0)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        self.encoder_no_repeat_ngram_size = kwargs.pop(
            "encoder_no_repeat_ngram_size", 0)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.num_return_sequences = kwargs.pop("num_return_sequences", 1)
        self.chunk_size_feed_forward = kwargs.pop("chunk_size_feed_forward", 0)
        self.output_scores = kwargs.pop("output_scores", False)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate",
                                                  False)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", False)

        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.finetuning_task = kwargs.pop("finetuning_task", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.id2label is not None:
            kwargs.pop("num_labels", None)
            self.id2label = dict((int(key), value)
                                 for key, value in self.id2label.items())
            # Keys are always strings in JSON so convert ids to int here.
        else:
            self.num_labels = kwargs.pop("num_labels", 2)

        # Tokenizer arguments TODO: eventually tokenizer and models should share the same config
        self.tokenizer_class = kwargs.pop("tokenizer_class", None)
        self.prefix = kwargs.pop("prefix", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        self.sep_token_id = kwargs.pop("sep_token_id", None)

        # task specific arguments
        self.task_specific_params = kwargs.pop("task_specific_params", None)

        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification",
                                 "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError(
                f"The config parameter `problem_type` was not understood: received {self.problem_type}"
                "but only 'regression', 'single_label_classification' and 'multi_label_classification' are valid."
            )

        # TPU arguments
        if kwargs.pop("xla_device", None) is not None:
            logger.warning(
                "The `xla_device` argument has been deprecated in v4.4.0 of Transformers. It is ignored and you can "
                "safely remove it from your `config.json` file.")

        # Name or path to the pretrained checkpoint
        self._name_or_path = str(kwargs.pop("name_or_path", ""))

        # Drop the hf_paddle version info
        self.hf_paddle_version = kwargs.pop("hf_paddle_version", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @property
    def name_or_path(self) -> str:
        return self._name_or_path

    @name_or_path.setter
    def name_or_path(self, value):
        self._name_or_path = str(
            value)  # Make sure that name_or_path is a string (for JSON encoding)

    @property
    def use_return_dict(self) -> bool:
        """
        :obj:`bool`: Whether or not return :class:`~hf_paddle.file_utils.ModelOutput` instead of tuples.
        """
        return self.return_dict

    @property
    def num_labels(self) -> int:
        """
        :obj:`int`: The number of labels for classification models.
        """
        return len(self.id2label)

    @num_labels.setter
    def num_labels(self, num_labels: int):
        if self.id2label is None or len(self.id2label) != num_labels:
            self.id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
            self.label2id = dict(
                zip(self.id2label.values(), self.id2label.keys()))

    def save_pretrained(self, save_directory, push_to_hub: bool=False,
                        **kwargs):
        """
        Save a configuration object to the directory ``save_directory``, so that it can be re-loaded using the
        :func:`~hf_paddle.PretrainedConfig.from_pretrained` class method.

        Args:
            save_directory (:obj:`str` or :obj:`os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to push your model to the Hugging Face model hub after saving it.

                .. warning::

                    Using :obj:`push_to_hub=True` will synchronize the repository you are pushing to with
                    :obj:`save_directory`, which requires :obj:`save_directory` to be a local clone of the repo you are
                    pushing to if it's an existing folder. Pass along :obj:`temp_dir=True` to use a temporary directory
                    instead.

            kwargs:
                Additional key word arguments passed along to the
                :meth:`~hf_paddle.file_utils.PushToHubMixin.push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path,
                        **kwargs) -> "PretrainedConfig":
        r"""
        Instantiate a :class:`~hf_paddle.PretrainedConfig` (or a derived class) from a pretrained model
        configuration.

        Args:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                This can be either:

                - a string, the `model id` of a pretrained model configuration hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like ``bert-base-uncased``, or
                  namespaced under a user or organization name, like ``dbmdz/bert-base-german-cased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~hf_paddle.PretrainedConfig.save_pretrained` method, e.g., ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.,
                  ``./my_model_directory/configuration.json``.
            cache_dir (:obj:`str` or :obj:`os.PathLike`, `optional`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            resume_download (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (:obj:`Dict[str, str]`, `optional`):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., :obj:`{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (:obj:`str` or `bool`, `optional`):
                The token to use as HTTP bearer authorization for remote files. If :obj:`True`, will use the token
                generated when running :obj:`hf_paddle-cli login` (stored in :obj:`~/.huggingface`).
            revision(:obj:`str`, `optional`, defaults to :obj:`"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so ``revision`` can be any
                identifier allowed by git.
            return_unused_kwargs (:obj:`bool`, `optional`, defaults to :obj:`False`):
                If :obj:`False`, then this function returns just the final configuration object.

                If :obj:`True`, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e.,
                the part of ``kwargs`` which has not been used to update ``config`` and is otherwise ignored.
            kwargs (:obj:`Dict[str, Any]`, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the ``return_unused_kwargs`` keyword parameter.

        .. note::

            Passing :obj:`use_auth_token=True` is required when you want to use a private model.


        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from this pretrained model.

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from huggingface.co and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True, foo=False)
            assert config.output_attentions == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attentions=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attentions == True
            assert unused_kwargs == {'foo': False}

        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path,
                                                  **kwargs)
        if "model_type" in config_dict and hasattr(
                cls,
                "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warn(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)

    def items(self):
        A = self.to_dict()
        for key in A:
            self[key] = A[key]
        return A.items()
        # return self.to_dict().items()

    def __str__(self):
        return self.to_json_string()

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        """
        From a ``pretrained_model_name_or_path``, resolve to a dictionary of parameters, to be used for instantiating a
        :class:`~hf_paddle.PretrainedConfig` using ``from_dict``.



        Parameters:
            pretrained_model_name_or_path (:obj:`str` or :obj:`os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the configuration object.

        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path

        try:
            # Load from URL or cache if already cached
            resolved_config_file = cache_dir + "/model_config.json"
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)

        except EnvironmentError as err:
            logger.error(err)
            msg = (
                f"Can't load config for '{pretrained_model_name_or_path}'. Make sure that:\n\n"
                f"- '{pretrained_model_name_or_path}' is a correct model identifier listed on 'https://huggingface.co/models'\n\n"
                f"- or '{pretrained_model_name_or_path}' is the correct path to a directory containing a {CONFIG_NAME} file\n\n"
            )
            raise EnvironmentError(msg)

        except json.JSONDecodeError:
            msg = (
                f"Couldn't reach server at '{config_file}' to download configuration file or "
                "configuration file is not a valid JSON file. "
                f"Please check network or file content here: {resolved_config_file}."
            )
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            logger.info(f"loading configuration file {config_file}")
        else:
            logger.info(
                f"loading configuration file {config_file} from cache at {resolved_config_file}"
            )

        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict, **kwargs) -> "PretrainedConfig":
        """
        Instantiates a :class:`~hf_paddle.PretrainedConfig` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                :func:`~hf_paddle.PretrainedConfig.get_config_dict` method.
            kwargs (:obj:`Dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)

        if hasattr(config, "pruned_heads"):
            config.pruned_heads = dict(
                (int(key), value) for key, value in config.pruned_heads.items())

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Model config {config}")
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file) -> "PretrainedConfig":
        """
        Instantiates a :class:`~hf_paddle.PretrainedConfig` from the path to a JSON file of parameters.

        Args:
            json_file (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`PretrainedConfig`: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_diff_dict(self):
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        # get the default config dict
        default_config_dict = PretrainedConfig().to_dict()

        # get class specific config dict
        class_config_dict = self.__class__().to_dict(
        ) if not self.is_composition else {}

        serializable_config_dict = {}

        # only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (key not in default_config_dict or key == "hf_paddle_version" or
                    value != default_config_dict[key] or
                (key in class_config_dict and value != class_config_dict[key])):
                serializable_config_dict[key] = value

        return serializable_config_dict

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        # Transformers version when serializing the model
        output["hf_paddle_version"] = "0.0.1"

        return output

    def to_json_string(self, use_diff: bool=True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON string.

        Returns:
            :obj:`str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path, use_diff: bool=True):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (:obj:`str` or :obj:`os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (:obj:`bool`, `optional`, defaults to :obj:`True`):
                If set to ``True``, only the difference between the config instance and the default
                ``PretrainedConfig()`` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff))

    def update(self, config_dict):
        """
        Updates attributes of this class with attributes from ``config_dict``.

        Args:
            config_dict (:obj:`Dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from ``update_str``.

        The expected format is ints, floats and strings as is, and for booleans use ``true`` or ``false``. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (:obj:`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(
                        f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise ValueError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)


class FunnelConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~hf_paddle.FunnelModel` or a
    :class:`~hf_paddle.TFBertModel`. It is used to instantiate a Funnel Transformer model according to the specified
    arguments, defining the model architecture. Instantiating a configuration with the defaults will yield a similar
    configuration to that of the Funnel Transformer `funnel-transformer/small
    <https://huggingface.co/funnel-transformer/small>`__ architecture.

    Configuration objects inherit from :class:`~hf_paddle.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~hf_paddle.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the Funnel transformer. Defines the number of different tokens that can be represented
            by the :obj:`inputs_ids` passed when calling :class:`~hf_paddle.FunnelModel` or
            :class:`~hf_paddle.TFFunnelModel`.
        block_sizes (:obj:`List[int]`, `optional`, defaults to :obj:`[4, 4, 4]`):
            The sizes of the blocks used in the model.
        block_repeats (:obj:`List[int]`, `optional`):
            If passed along, each layer of each block is repeated the number of times indicated.
        num_decoder_layers (:obj:`int`, `optional`, defaults to 2):
            The number of layers in the decoder (when not using the base model).
        d_model (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the model's hidden states.
        n_head (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        d_head (:obj:`int`, `optional`, defaults to 64):
            Dimensionality of the model's heads.
        d_inner (:obj:`int`, `optional`, defaults to 3072):
            Inner dimension in the feed-forward blocks.
        hidden_act (:obj:`str` or :obj:`callable`, `optional`, defaults to :obj:`"gelu_new"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probability used between the two layers of the feed-forward blocks.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 3):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~hf_paddle.FunnelModel` or
            :class:`~hf_paddle.TFFunnelModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.1):
            The standard deviation of the `uniform initializer` for initializing all weight matrices in attention
            layers.
        initializer_std (:obj:`float`, `optional`):
            The standard deviation of the `normal initializer` for initializing the embedding matrix and the weight of
            linear layers. Will default to 1 for the embedding matrix and the value given by Xavier initialization for
            linear layers.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-9):
            The epsilon used by the layer normalization layers.
        pooling_type (:obj:`str`, `optional`, defaults to :obj:`"mean"`):
            Possible values are ``"mean"`` or ``"max"``. The way pooling is performed at the beginning of each block.
        attention_type (:obj:`str`, `optional`, defaults to :obj:`"relative_shift"`):
            Possible values are ``"relative_shift"`` or ``"factorized"``. The former is faster on CPU/GPU while the
            latter is faster on TPU.
        separate_cls (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to separate the cls token when applying pooling.
        truncate_seq (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When using ``separate_cls``, whether or not to truncate the last token when pooling, to avoid getting a
            sequence length that is not a multiple of 2.
        pool_q_only (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to apply the pooling only to the query or to query, key and values for the attention layers.
    """
    model_type = "funnel"

    def __init__(self,
                 vocab_size=30522,
                 block_sizes=[4, 4, 4],
                 block_repeats=None,
                 num_decoder_layers=2,
                 d_model=768,
                 n_head=12,
                 d_head=64,
                 d_inner=3072,
                 hidden_act="gelu_new",
                 hidden_dropout=0.1,
                 attention_dropout=0.1,
                 activation_dropout=0.0,
                 max_position_embeddings=512,
                 type_vocab_size=3,
                 initializer_range=0.1,
                 initializer_std=None,
                 layer_norm_eps=1e-9,
                 pooling_type="mean",
                 attention_type="relative_shift",
                 separate_cls=True,
                 truncate_seq=True,
                 pool_q_only=True,
                 **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.block_sizes = block_sizes
        self.block_repeats = [1] * len(
            block_sizes) if block_repeats is None else block_repeats
        assert len(block_sizes) == len(
            self.block_repeats
        ), "`block_sizes` and `block_repeats` should have the same length."
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.initializer_std = initializer_std
        self.layer_norm_eps = layer_norm_eps
        assert pooling_type in [
            "mean",
            "max",
        ], f"Got {pooling_type} for `pooling_type` but only 'mean' and 'max' are supported."
        self.pooling_type = pooling_type
        assert attention_type in [
            "relative_shift",
            "factorized",
        ], f"Got {attention_type} for `attention_type` but only 'relative_shift' and 'factorized' are supported."
        self.attention_type = attention_type
        self.separate_cls = separate_cls
        self.truncate_seq = truncate_seq
        self.pool_q_only = pool_q_only

        self.items()  ##strange way to initialize the dict

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def num_attention_heads(self):
        return self.n_head

    @property
    def num_hidden_layers(self):
        return sum(self.block_sizes)

    @property
    def num_blocks(self):
        return len(self.block_sizes)


def expand(self, *sizes):
    if isinstance(sizes[0], Iterable):
        sizes = sizes[0]
    ##handle -1 case
    if len(sizes) > len(self.shape):
        for _ in range(len(sizes) - len(self.shape)):
            self = self.unsqueeze(axis=0)
    x = paddle.expand(self, sizes, name=None)
    return x


def repeat_interleave(x, repeats, dim=None):
    orig_shape = list(x.shape)
    if dim is None:
        dim = 1
        x = paddle.reshape(x, (-1, 1))  # x.reshape(-1,1)
        size = [1] * len(x.shape)
        size[dim] = repeats
        x = paddle.tile(x, size)
        return paddle.reshape(x, (-1))
    else:
        if len(orig_shape) == dim + 1:
            x = x.unsqueeze(-1)
        # x=x.reshape(-1,1)
        size = [1] * len(orig_shape)
        size[-1] = repeats
        x = paddle.tile(x, size)
        orig_shape[dim] = -1
        return paddle.reshape(x, orig_shape)


def gather(x, dim, index):
    index_shape = index.shape
    index_flatten = index.flatten()
    if dim < 0:
        dim = len(x.shape) + dim
    nd_index = []
    for k in range(len(x.shape)):
        if k == dim:
            nd_index.append(index_flatten)
        else:
            reshape_shape = [1] * len(x.shape)
            reshape_shape[k] = x.shape[k]
            dim_index = paddle.expand(
                paddle.reshape(
                    paddle.arange(
                        x.shape[k], dtype=index.dtype), reshape_shape),
                index_shape).flatten()
            nd_index.append(dim_index)

    ind2 = paddle.transpose(paddle.stack(nd_index), [1, 0])
    paddle_out = paddle.gather_nd(x, ind2).reshape(index_shape)
    return paddle_out


def split(x, batch_size, dim=0):
    if isinstance(batch_size, int):
        if batch_size > x.shape[dim]:
            return [x]  # do nothing
        return [y for y in paddle.split(x, x.shape[dim] // batch_size, dim)]
    else:
        return [y for y in paddle.split(x, batch_size, dim)]


def normal_(x, m=0, std=1):
    y = paddle.randn(x.shape) * std + m
    paddle.assign(y, x)
    return x


def uniform_(x, a=0, b=1.):
    temp_value = paddle.uniform(min=a, max=b, shape=x.shape)
    x.set_value(temp_value)
    return x


def constant_(x, val):
    temp_value = paddle.full_like(x, fill_value=val)
    x.set_value(temp_value)
    return x


def _gelu_python(x):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    paddle.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * paddle.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + paddle.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


gelu = nn.functional.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + paddle.tanh(x * 0.7978845608 *
                                        (1.0 + 0.044715 * x * x)))


def quick_gelu(x):
    return x * paddle.sigmoid(1.702 * x)


silu = nn.functional.silu


def _mish_python(x):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """
    return x * paddle.tanh(nn.functional.softplus(x))


def linear_act(x):
    return x


ACT2FN = {
    "relu": nn.functional.relu,
    "silu": silu,
    "swish": silu,
    "gelu": gelu,
    "tanh": paddle.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "quick_gelu": quick_gelu,
    "linear": linear_act,
    "sigmoid": paddle.nn.functional.sigmoid,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(
            f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}"
        )


class FunnelEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id)
        self.layer_norm = LayerNorm(
            config.d_model, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        embeddings = self.layer_norm(inputs_embeds)
        embeddings = self.dropout(embeddings)

        return embeddings


def pad(input, pad, mode='constant', value=0):
    pad2 = []
    for _ in range(len(input.shape) * 2 - len(pad)):
        pad2.append(0)
    if isinstance(pad, tuple):
        pad = list(pad)
    pad2 = pad2 + pad
    return paddle.nn.functional.pad(input, pad2, mode=mode, value=value)


class FunnelAttentionStructure(nn.Layer):
    """
    Contains helpers for `FunnelRelMultiheadAttention `.
    """

    cls_token_type_id: int = 2

    def __init__(self, config):
        super().__init__()
        self.config2 = config
        self.sin_dropout = nn.Dropout(config.hidden_dropout)
        self.cos_dropout = nn.Dropout(config.hidden_dropout)
        # Track where we are at in terms of pooling from the original input, e.g., by how much the sequence length was
        # divided.
        self.pooling_mult = None

    def init_attention_inputs(self,
                              inputs_embeds,
                              attention_mask=None,
                              token_type_ids=None):
        """Returns the attention inputs associated to the inputs of the model."""
        # inputs_embeds has shape batch_size x seq_len x d_model
        # attention_mask and token_type_ids have shape batch_size x seq_len
        self.pooling_mult = 1
        self.seq_len = seq_len = inputs_embeds.shape[1]
        position_embeds = self.get_position_embeds(seq_len, inputs_embeds.dtype)
        token_type_mat = self.token_type_ids_to_mat(
            token_type_ids) if token_type_ids is not None else None
        cls_mask = (
            pad(
                paddle.ones(
                    [seq_len - 1, seq_len - 1],
                    dtype=inputs_embeds.dtype), (1, 0, 1, 0)
            )  # nn.functional.pad(inputs_embeds.new_ones([seq_len - 1, seq_len - 1]), (1, 0, 1, 0))
            if self.config2.separate_cls else None)
        return (position_embeds, token_type_mat, attention_mask, cls_mask)

    def token_type_ids_to_mat(self, token_type_ids):
        """Convert `token_type_ids` to `token_type_mat`."""
        # token_type_mat = token_type_ids[:, :, None] == token_type_ids[:, None]
        token_type_mat = token_type_ids.unsqueeze(
            2) == token_type_ids.unsqueeze(1)
        # Treat <cls> as in the same segment as both A & B
        cls_ids = token_type_ids == self.cls_token_type_id
        # cls_mat = cls_ids[:, :, None] | cls_ids[:, None]
        cls_mat = paddle.logical_or(cls_ids.unsqueeze(2), cls_ids.unsqueeze(1))
        return paddle.logical_or(cls_mat, token_type_mat)

    def get_position_embeds(self, seq_len, dtype):
        """
        Create and cache inputs related to relative position encoding. Those are very different depending on whether we
        are using the factorized or the relative shift attention:

        For the factorized attention, it returns the matrices (phi, pi, psi, omega) used in the paper, appendix A.2.2,
        final formula.

        For the relative shift attention, it returns all possible vectors R used in the paper, appendix A.2.1, final
        formula.

        Paper link: https://arxiv.org/abs/2006.03236
        """
        d_model = self.config2.d_model
        if self.config2.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula.
            # We need to create and return the matrices phi, psi, pi and omega.
            pos_seq = paddle.arange(0, seq_len, 1.0, dtype=dtype)
            freq_seq = paddle.arange(0, d_model // 2, 1.0, dtype=dtype)
            inv_freq = 1 / (10000**(freq_seq / (d_model // 2)))
            sinusoid = pos_seq.unsqueeze(1) * inv_freq.unsqueeze(0)
            sin_embed = paddle.sin(sinusoid)
            sin_embed_d = self.sin_dropout(sin_embed)
            cos_embed = paddle.cos(sinusoid)
            cos_embed_d = self.cos_dropout(cos_embed)
            # This is different from the formula on the paper...
            phi = paddle.concat([sin_embed_d, sin_embed_d], axis=-1)
            psi = paddle.concat([cos_embed, sin_embed], axis=-1)
            pi = paddle.concat([cos_embed_d, cos_embed_d], axis=-1)
            omega = paddle.concat([-sin_embed, cos_embed], axis=-1)
            return (phi, pi, psi, omega)
        else:
            # Notations from the paper, appending A.2.1, final formula.
            # We need to create and return all the possible vectors R for all blocks and shifts.
            freq_seq = paddle.arange(0, d_model // 2, 1, dtype=dtype)
            inv_freq = 1 / (10000**(freq_seq / (d_model // 2)))
            # Maximum relative positions for the first input
            rel_pos_id = paddle.arange(
                -seq_len * 2, seq_len * 2, 1, dtype=dtype)
            zero_offset = seq_len * 2
            sinusoid = rel_pos_id.unsqueeze(1) * inv_freq.unsqueeze(0)
            sin_embed = self.sin_dropout(paddle.sin(sinusoid))
            cos_embed = self.cos_dropout(paddle.cos(sinusoid))
            pos_embed = paddle.concat([sin_embed, cos_embed], axis=-1)

            pos = paddle.arange(0, seq_len, dtype=dtype)
            pooled_pos = pos
            position_embeds_list = []
            for block_index in range(0, self.config2.num_blocks):
                # For each block with block_index > 0, we need two types position embeddings:
                #   - Attention(pooled-q, unpooled-kv)
                #   - Attention(pooled-q, pooled-kv)
                # For block_index = 0 we only need the second one and leave the first one as None.

                # First type
                if block_index == 0:
                    position_embeds_pooling = None
                else:
                    pooled_pos = self.stride_pool_pos(pos, block_index)

                    # construct rel_pos_id
                    stride = 2**(block_index - 1)
                    rel_pos = self.relative_pos(
                        pos, stride, pooled_pos, shift=2)
                    rel_pos = rel_pos.unsqueeze(1) + zero_offset
                    rel_pos = expand(rel_pos, (rel_pos.shape[0], d_model))
                    position_embeds_pooling = gather(pos_embed, 0, rel_pos)

                # Second type
                pos = pooled_pos
                stride = 2**block_index
                rel_pos = self.relative_pos(pos, stride)

                rel_pos = rel_pos.unsqueeze(1) + zero_offset
                rel_pos = expand(rel_pos, (rel_pos.shape[0], d_model))
                position_embeds_no_pooling = gather(pos_embed, 0, rel_pos)

                position_embeds_list.append(
                    [position_embeds_no_pooling, position_embeds_pooling])
            return position_embeds_list

    def stride_pool_pos(self, pos_id, block_index):
        """
        Pool `pos_id` while keeping the cls token separate (if `config.separate_cls=True`).
        """
        if self.config2.separate_cls:
            # Under separate <cls>, we treat the <cls> as the first token in
            # the previous block of the 1st real block. Since the 1st real
            # block always has position 1, the position of the previous block
            # will be at `1 - 2 ** block_index`.
            cls_pos = paddle.to_tensor(
                [-(2**block_index) + 1]).astype(pos_id.dtype)
            pooled_pos_id = pos_id[
                1:-1] if self.config2.truncate_seq else pos_id[1:]
            return paddle.concat([cls_pos, pooled_pos_id[::2]], axis=0)
        else:
            return pos_id[::2]

    def relative_pos(self, pos, stride, pooled_pos=None, shift=1):
        """
        Build the relative positional vector between `pos` and `pooled_pos`.
        """
        if pooled_pos is None:
            pooled_pos = pos

        ref_point = pooled_pos[0] - pos[0]
        num_remove = shift * len(pooled_pos)
        max_dist = ref_point + num_remove * stride
        min_dist = pooled_pos[0] - pos[-1]

        return paddle.arange(
            max_dist, min_dist - 1, -stride, dtype=paddle.int64)

    def stride_pool(self, tensor, axis):
        """
        Perform pooling by stride slicing the tensor along the given axis.
        """
        if tensor is None:
            return None
        tensor = tensor.astype("float32")
        # Do the stride pool recursively if axis is a list or a tuple of ints.
        if isinstance(axis, (list, tuple)):
            for ax in axis:
                tensor = self.stride_pool(tensor, ax)
            return tensor

        # Do the stride pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.stride_pool(x, axis) for x in tensor)

        # Deal with negative axis
        axis %= tensor.ndim

        axis_slice = (slice(None, -1, 2) if self.config2.separate_cls and
                      self.config2.truncate_seq else slice(None, None, 2))
        enc_slice = [slice(None)] * axis + [axis_slice]
        if self.config2.separate_cls:
            cls_slice = [slice(None)] * axis + [slice(None, 1)]
            # tensor = paddle.cat([tensor[cls_slice], tensor], axis=axis)
            if axis == 1:
                tensor = paddle.concat([tensor[:, :1], tensor], axis=axis)
            if axis == 2:
                tensor = paddle.concat([tensor[:, :, :1], tensor], axis=axis)
            if axis == 0:
                tensor = paddle.concat([tensor[:1], tensor], axis=axis)
        if axis == 1:
            return tensor[:, 0:-1:2].astype("bool")
        if axis == 0:
            return tensor[0:-1:2].astype("bool")
        if axis == 2:
            return tensor[:, :, 0:-1:2].astype("bool")

    def pool_tensor(self, tensor, mode="mean", stride=2):
        """Apply 1D pooling to a tensor of size [B x T (x H)]."""
        if tensor is None:
            return None

        # Do the pool recursively if tensor is a list or tuple of tensors.
        if isinstance(tensor, (tuple, list)):
            return type(tensor)(self.pool_tensor(
                tensor, mode=mode, stride=stride) for x in tensor)

        if self.config2.separate_cls:
            suffix = tensor[:, :-1] if self.config2.truncate_seq else tensor
            tensor = paddle.concat([tensor[:, :1], suffix], axis=1)

        ndim = tensor.ndim
        if ndim == 2:
            tensor = tensor.unsqueeze(1).unsqueeze(3)  # [:, None, :, None]
        elif ndim == 3:
            tensor = tensor.unsqueeze(1)  # [:, None, :, :]
        # Stride is applied on the second-to-last dimension.
        stride = (stride, 1)

        if mode == "mean":
            tensor = nn.functional.avg_pool2d(
                tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "max":
            tensor = nn.functional.max_pool2d(
                tensor, stride, stride=stride, ceil_mode=True)
        elif mode == "min":
            tensor = -nn.functional.max_pool2d(
                -tensor, stride, stride=stride, ceil_mode=True)
        else:
            raise NotImplementedError(
                "The supported modes are 'mean', 'max' and 'min'.")

        if ndim == 2:
            return tensor[:, 0, :, 0]
        elif ndim == 3:
            return tensor[:, 0]
        return tensor

    def pre_attention_pooling(self, output, attention_inputs):
        """Pool `output` and the proper parts of `attention_inputs` before the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config2.pool_q_only:
            if self.config2.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds[:2],
                                                   0) + position_embeds[2:]
            token_type_mat = self.stride_pool(token_type_mat, 1)
            cls_mask = self.stride_pool(cls_mask, 0)
            output = self.pool_tensor(output, mode=self.config2.pooling_type)

        else:
            self.pooling_mult *= 2
            if self.config2.attention_type == "factorized":
                position_embeds = self.stride_pool(position_embeds, 0)
            token_type_mat = self.stride_pool(token_type_mat, [1, 2])
            cls_mask = self.stride_pool(cls_mask, [1, 2])
            attention_mask = self.pool_tensor(attention_mask, mode="min")
            output = self.pool_tensor(output, mode=self.config2.pooling_type)

        attention_inputs = (position_embeds, token_type_mat, attention_mask,
                            cls_mask)
        return output, attention_inputs

    def post_attention_pooling(self, attention_inputs):
        """Pool the proper parts of `attention_inputs` after the attention layer."""
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs
        if self.config2.pool_q_only:
            self.pooling_mult *= 2
            if self.config2.attention_type == "factorized":
                position_embeds = position_embeds[:2] + self.stride_pool(
                    position_embeds[2:], 0)
            token_type_mat = self.stride_pool(token_type_mat, 2)
            cls_mask = self.stride_pool(cls_mask, 1)
            attention_mask = self.pool_tensor(attention_mask, mode="min")
        attention_inputs = (position_embeds, token_type_mat, attention_mask,
                            cls_mask)
        return attention_inputs


def _relative_shift_gather(positional_attn, context_len, shift):
    batch_size, n_head, seq_len, max_rel_len = positional_attn.shape
    # max_rel_len = 2 * context_len + shift -1 is the numbers of possible relative positions i-j

    # What's next is the same as doing the following gather, which might be clearer code but less efficient.
    # idxs = context_len + paddle.arange(0, context_len).unsqueeze(0) - paddle.arange(0, seq_len).unsqueeze(1)
    # # matrix of context_len + i-j
    # return positional_attn.gather(3, idxs.expand([batch_size, n_head, context_len, context_len]))

    positional_attn = paddle.reshape(
        positional_attn, [batch_size, n_head, max_rel_len, seq_len])
    positional_attn = positional_attn[:, :, shift:, :]
    positional_attn = paddle.reshape(
        positional_attn, [batch_size, n_head, seq_len, max_rel_len - shift])
    positional_attn = positional_attn[:, :, :, :context_len]
    return positional_attn


def Parameter(shape_or_tensor, fill_value=None, requires_grad=True):
    if isinstance(shape_or_tensor, paddle.Tensor):
        X = Parameter(shape_or_tensor.shape, 0.0)
        paddle.assign(shape_or_tensor.astype("float32"), X)
    else:
        if isinstance(shape_or_tensor, int):
            shape_or_tensor = [shape_or_tensor]

        X = paddle.create_parameter(
            shape=shape_or_tensor,
            dtype="float32",
            attr=paddle.ParamAttr(
                name=None,
                initializer=paddle.nn.initializer.Constant(value=fill_value)),
            is_bias=False)
    if not requires_grad:
        X.stop_gradient = True

    return X


class FunnelRelMultiheadAttention(nn.Layer):
    def __init__(self, config, block_index):
        super().__init__()
        self.config2 = config
        self.block_index = block_index
        d_model, n_head, d_head = config.d_model, config.n_head, config.d_head

        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

        self.q_head = nn.Linear(d_model, n_head * d_head, bias_attr=False)
        self.k_head = nn.Linear(d_model, n_head * d_head)
        self.v_head = nn.Linear(d_model, n_head * d_head)

        self.r_w_bias = Parameter(paddle.zeros([n_head, d_head]))
        self.r_r_bias = Parameter(paddle.zeros([n_head, d_head]))
        self.r_kernel = Parameter(paddle.zeros([d_model, n_head, d_head]))
        self.r_s_bias = Parameter(paddle.zeros([n_head, d_head]))
        self.seg_embed = Parameter(paddle.zeros([2, n_head, d_head]))

        self.post_proj = nn.Linear(n_head * d_head, d_model)
        self.layer_norm = LayerNorm(d_model, epsilon=config.layer_norm_eps)
        self.scale = 1.0 / (d_head**0.5)

    def relative_positional_attention(self,
                                      position_embeds,
                                      q_head,
                                      context_len,
                                      cls_mask=None):
        """Relative attention score for the positional encodings"""
        # q_head has shape batch_size x sea_len x n_head x d_head
        if self.config2.attention_type == "factorized":
            # Notations from the paper, appending A.2.2, final formula (https://arxiv.org/abs/2006.03236)
            # phi and pi have shape seq_len x d_model, psi and omega have shape context_len x d_model
            phi, pi, psi, omega = position_embeds
            # Shape n_head x d_head
            u = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape batch_size x sea_len x n_head x d_model
            q_r_attention = paddle.einsum("binh,dnh->bind", q_head + u, w_r)
            q_r_attention_1 = q_r_attention * phi.unsqueeze(1)  # [:, None]
            q_r_attention_2 = q_r_attention * pi.unsqueeze(1)  # [:, None]

            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = paddle.einsum(
                "bind,jd->bnij", q_r_attention_1, psi) + paddle.einsum(
                    "bind,jd->bnij", q_r_attention_2, omega)
        else:
            shift = 2 if q_head.shape[1] != context_len else 1
            # Notations from the paper, appending A.2.1, final formula (https://arxiv.org/abs/2006.03236)
            # Grab the proper positional encoding, shape max_rel_len x d_model
            r = position_embeds[self.block_index][shift - 1]
            # Shape n_head x d_head
            v = self.r_r_bias * self.scale
            # Shape d_model x n_head x d_head
            w_r = self.r_kernel

            # Shape max_rel_len x n_head x d_model
            r_head = paddle.einsum("td,dnh->tnh", r, w_r)
            # Shape batch_size x n_head x seq_len x max_rel_len
            positional_attn = paddle.einsum("binh,tnh->bnit", q_head + v,
                                            r_head)
            # Shape batch_size x n_head x seq_len x context_len
            positional_attn = _relative_shift_gather(positional_attn,
                                                     context_len, shift)

        if cls_mask is not None:
            positional_attn *= cls_mask
        return positional_attn

    def relative_token_type_attention(self,
                                      token_type_mat,
                                      q_head,
                                      cls_mask=None):
        """Relative attention score for the token_type_ids"""
        if token_type_mat is None:
            return 0
        batch_size, seq_len, context_len = token_type_mat.shape
        # q_head has shape batch_size x seq_len x n_head x d_head
        # Shape n_head x d_head
        r_s_bias = self.r_s_bias * self.scale

        # Shape batch_size x n_head x seq_len x 2
        token_type_bias = paddle.einsum("bind,snd->bnis", q_head + r_s_bias,
                                        self.seg_embed)

        # Shape batch_size x n_head x seq_len x context_len
        # token_type_mat = token_type_mat[:, None].expand([batch_size, q_head.shape[2], seq_len, context_len])
        token_type_mat = expand(
            token_type_mat.unsqueeze(1),
            ([batch_size, q_head.shape[2], seq_len, context_len]))
        # Shapes batch_size x n_head x seq_len
        diff_token_type, same_token_type = split(token_type_bias, 1, dim=-1)
        # Shape batch_size x n_head x seq_len x context_len
        token_type_attn = paddle.where(
            token_type_mat,
            expand(same_token_type, (token_type_mat.shape)),
            expand(diff_token_type, (token_type_mat.shape)))

        if cls_mask is not None:
            token_type_attn *= cls_mask
        return token_type_attn

    def forward(self,
                query,
                key,
                value,
                attention_inputs,
                output_attentions=False):
        # query has shape batch_size x seq_len x d_model
        # key and value have shapes batch_size x context_len x d_model
        position_embeds, token_type_mat, attention_mask, cls_mask = attention_inputs

        batch_size, seq_len, _ = query.shape
        context_len = key.shape[1]
        n_head, d_head = self.config2.n_head, self.config2.d_head

        # Shape batch_size x seq_len x n_head x d_head
        q_head = paddle.reshape(
            self.q_head(query), (batch_size, seq_len, n_head, d_head)
        )  # self.q_head(query).reshape(batch_size, seq_len, n_head, d_head)
        # Shapes batch_size x context_len x n_head x d_head
        k_head = paddle.reshape(
            self.k_head(key), (batch_size, context_len, n_head, d_head)
        )  # self.k_head(key).reshape(batch_size, context_len, n_head, d_head)
        v_head = paddle.reshape(
            self.v_head(value), (batch_size, context_len, n_head, d_head))

        q_head = q_head * self.scale
        # Shape n_head x d_head
        r_w_bias = self.r_w_bias * self.scale
        # Shapes batch_size x n_head x seq_len x context_len

        content_score = paddle.einsum("bind,bjnd->bnij", q_head + r_w_bias,
                                      k_head)

        positional_attn = self.relative_positional_attention(
            position_embeds, q_head, context_len, cls_mask)
        token_type_attn = self.relative_token_type_attention(token_type_mat,
                                                             q_head, cls_mask)

        # merge attention scores
        attn_score = content_score + positional_attn + token_type_attn

        # precision safe in case of mixed precision training
        dtype = attn_score.dtype
        attn_score = attn_score.astype("float32")
        # perform masking
        if attention_mask is not None:
            # attn_score = attn_score - INF * (1 - attention_mask[:, None, None].float())
            attn_score = attn_score - INF * (
                1 - attention_mask.unsqueeze(1).unsqueeze(2).astype("float32"))
        # attention probability
        attn_prob = paddle.nn.functional.softmax(
            attn_score, axis=-1, dtype=dtype)
        attn_prob = self.attention_dropout(attn_prob)

        # attention output, shape batch_size x seq_len x n_head x d_head
        attn_vec = paddle.einsum("bnij,bjnd->bind", attn_prob, v_head)

        # Shape shape batch_size x seq_len x d_model
        attn_out = self.post_proj(
            attn_vec.reshape((batch_size, seq_len, n_head * d_head)))
        attn_out = self.hidden_dropout(attn_out)

        output = self.layer_norm(query + attn_out)
        return (output, attn_prob) if output_attentions else (output, )


class FunnelPositionwiseFFN(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.d_model, config.d_inner)
        self.activation_function = ACT2FN[config.hidden_act]
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.linear_2 = nn.Linear(config.d_inner, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = LayerNorm(
            config.d_model, epsilon=config.layer_norm_eps)

    def forward(self, hidden):
        h = self.linear_1(hidden)
        h = self.activation_function(h)
        h = self.activation_dropout(h)
        h = self.linear_2(h)
        h = self.dropout(h)
        return self.layer_norm(hidden + h)


class FunnelLayer(nn.Layer):
    def __init__(self, config, block_index):
        super().__init__()
        self.attention = FunnelRelMultiheadAttention(config, block_index)
        self.ffn = FunnelPositionwiseFFN(config)

    def forward(self,
                query,
                key,
                value,
                attention_inputs,
                output_attentions=False):
        attn = self.attention(
            query,
            key,
            value,
            attention_inputs,
            output_attentions=output_attentions)
        output = self.ffn(attn[0])
        return (output, attn[1]) if output_attentions else (output, )


class FunnelEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config2 = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.blocks = nn.LayerList([
            nn.LayerList(
                [FunnelLayer(config, block_index) for _ in range(block_size)])
            for block_index, block_size in enumerate(config.block_sizes)
        ])

    def forward(
            self,
            inputs_embeds,
            attention_mask=None,
            token_type_ids=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True, ):
        # The pooling is not implemented on long tensors, so we convert this mask.
        attention_mask = attention_mask.astype(inputs_embeds.dtype)
        attention_inputs = self.attention_structure.init_attention_inputs(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, )
        hidden = inputs_embeds

        all_hidden_states = (inputs_embeds, ) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for block_index, block in enumerate(self.blocks):
            pooling_flag = hidden.shape[1] > (2 if self.config2.separate_cls
                                              else 1)
            pooling_flag = pooling_flag and block_index > 0
            if pooling_flag:
                pooled_hidden, attention_inputs = self.attention_structure.pre_attention_pooling(
                    hidden, attention_inputs)
            for (layer_index, layer) in enumerate(block):
                for repeat_index in range(self.config2.block_repeats[
                        block_index]):
                    do_pooling = (repeat_index == 0) and (
                        layer_index == 0) and pooling_flag
                    if do_pooling:
                        query = pooled_hidden
                        key = value = hidden if self.config2.pool_q_only else pooled_hidden
                    else:
                        query = key = value = hidden
                    # if layer_index==8 and block_index==0 and repeat_index==0 :
                    #     print(block_index,layer_index,repeat_index,layer,query.mean(), key.mean(), value.mean())
                    layer_output = layer(
                        query,
                        key,
                        value,
                        attention_inputs,
                        output_attentions=output_attentions)

                    hidden = layer_output[0]

                    if do_pooling:
                        attention_inputs = self.attention_structure.post_attention_pooling(
                            attention_inputs)

                    if output_attentions:
                        all_attentions = all_attentions + layer_output[1:]
                    if output_hidden_states:
                        all_hidden_states = all_hidden_states + (hidden, )
        if not return_dict:
            return tuple(v
                         for v in [hidden, all_hidden_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden,
            hidden_states=all_hidden_states,
            attentions=all_attentions)


def upsample(x, stride, target_len, separate_cls=True, truncate_seq=False):
    """
    Upsample tensor `x` to match `target_len` by repeating the tokens `stride` time on the sequence length dimension.
    """
    if stride == 1:
        return x
    if separate_cls:
        cls = x[:, :1]
        x = x[:, 1:]
    output = repeat_interleave(x, repeats=stride, dim=1)
    if separate_cls:
        if truncate_seq:
            output = pad(output, (0, 0, 0, stride - 1, 0, 0))
        output = output[:, :target_len - 1]
        output = paddle.concat([cls, output], axis=1)
    else:
        output = output[:, :target_len]
    return output


class FunnelDecoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config2 = config
        self.attention_structure = FunnelAttentionStructure(config)
        self.layers = nn.LayerList(
            [FunnelLayer(config, 0) for _ in range(config.num_decoder_layers)])

    def forward(
            self,
            final_hidden,
            first_block_hidden,
            attention_mask=None,
            token_type_ids=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True, ):
        upsampled_hidden = upsample(
            final_hidden,
            stride=2**(len(self.config2.block_sizes) - 1),
            target_len=first_block_hidden.shape[1],
            separate_cls=self.config2.separate_cls,
            truncate_seq=self.config2.truncate_seq, )

        hidden = upsampled_hidden + first_block_hidden
        all_hidden_states = (hidden, ) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        attention_inputs = self.attention_structure.init_attention_inputs(
            hidden,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, )

        for layer in self.layers:
            layer_output = layer(
                hidden,
                hidden,
                hidden,
                attention_inputs,
                output_attentions=output_attentions)
            hidden = layer_output[0]

            if output_attentions:
                all_attentions = all_attentions + layer_output[1:]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden, )

        if not return_dict:
            return tuple(v
                         for v in [hidden, all_hidden_states, all_attentions]
                         if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden,
            hidden_states=all_hidden_states,
            attentions=all_attentions)


class FunnelDiscriminatorPredictions(nn.Layer):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()
        self.config2 = config
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.dense_prediction = nn.Linear(config.d_model, 1)

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = ACT2FN[self.config2.hidden_act](hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze()
        return logits


class FunnelPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        "funnel-transformer/small": {},  # B4-4-4H768
        "funnel-transformer/small-base": {},  # B4-4-4H768, no decoder
        "funnel-transformer/medium": {},  # B6-3x2-3x2H768
        "funnel-transformer/medium-base": {},  # B6-3x2-3x2H768, no decoder
        "funnel-transformer/intermediate": {},  # B6-6-6H768
        "funnel-transformer/intermediate-base": {},  # B6-6-6H768, no decoder
        "funnel-transformer/large": {},  # B8-8-8H1024
        "funnel-transformer/large-base": {},  # B8-8-8H1024, no decoder
        "funnel-transformer/xlarge-base": {},  # B10-10-10H1024
        "funnel-transformer/xlarge": {},  # B10-10-10H1024, no decoder
    }
    resource_files_names = {
        "model_state": "model_state.pdparams",
        "model_config": "model_config.json"
    }
    pretrained_resource_files_map = {
        "model_state": {
            "funnel-transformer/small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small/model_state.pdparams",
            "funnel-transformer/small-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small-base/model_state.pdparams",
            "funnel-transformer/medium":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium/model_state.pdparams",
            "funnel-transformer/medium-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium-base/model_state.pdparams",
            "funnel-transformer/intermediate":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate/model_state.pdparams",
            "funnel-transformer/intermediate-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate-base/model_state.pdparams",
            "funnel-transformer/large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large/model_state.pdparams",
            "funnel-transformer/large-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large-base/model_state.pdparams",
            "funnel-transformer/xlarge-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge-base/model_state.pdparams",
            "funnel-transformer/xlarge":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge/model_state.pdparams",
        },
        "model_config": {
            "funnel-transformer/small":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small/model_config.json",
            "funnel-transformer/small-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/small-base/model_config.json",
            "funnel-transformer/medium":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium/model_config.json",
            "funnel-transformer/medium-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/medium-base/model_config.json",
            "funnel-transformer/intermediate":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate/model_config.json",
            "funnel-transformer/intermediate-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/intermediate-base/model_config.json",
            "funnel-transformer/large":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large/model_config.json",
            "funnel-transformer/large-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/large-base/model_config.json",
            "funnel-transformer/xlarge-base":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge-base/model_config.json",
            "funnel-transformer/xlarge":
            "https://bj.bcebos.com/paddlenlp/models/transformers/funnel-transformer/xlarge/model_config.json",
        },
    }

    config_class = FunnelConfig
    base_model_prefix = "funnel"

    def _init_weights(self, module):
        classname = module.__class__.__name__
        if classname.find("Linear") != -1:
            if getattr(module, "weight", None) is not None:
                if self.config2.initializer_std is None:
                    fan_out, fan_in = module.weight.shape
                    std = np.sqrt(1.0 / float(fan_in + fan_out))
                else:
                    std = self.config2.initializer_std
                normal_(module.weight, std=std)
            if getattr(module, "bias", None) is not None:
                constant_(module.bias, 0.0)
        elif classname == "FunnelRelMultiheadAttention":
            uniform_(module.r_w_bias, b=self.config2.initializer_range)
            uniform_(module.r_r_bias, b=self.config2.initializer_range)
            uniform_(module.r_kernel, b=self.config2.initializer_range)
            uniform_(module.r_s_bias, b=self.config2.initializer_range)
            uniform_(module.seg_embed, b=self.config2.initializer_range)
        elif classname == "FunnelEmbeddings":
            std = 1.0 if self.config2.initializer_std is None else self.config2.initializer_std
            normal_(module.word_embeddings.weight, std=std)
            if module.word_embeddings._padding_idx is not None:
                module.word_embeddings.weight.data[module._padding_idx].zero_()

    def init_weights(self):
        """
        If needed prunes and maybe initializes weights.
        """
        # Prune heads if needed
        if self.config2.pruned_heads:
            self.prune_heads(self.config2.pruned_heads)
        _init_weights = True
        if _init_weights:
            # Initialize weights
            self.apply(self._init_weights)

            # Tie weights should be skipped when not initializing all weights
            # since from_pretrained(...) calls tie weights anyways
            # self.tie_weights()

    def prune_heads(self, heads_to_prune):
        """
        Prunes heads of the base model.

        Arguments:
            heads_to_prune (:obj:`Dict[int, List[int]]`):
                Dictionary with keys being selected layer indices (:obj:`int`) and associated values being the list of
                heads to prune in said layer (list of :obj:`int`). For instance {1: [0, 2], 2: [2, 3]} will prune heads
                0 and 2 on layer 1 and heads 2 and 3 on layer 2.
        """
        # save new sets of pruned heads as union of previously stored pruned heads and newly pruned heads
        for layer, heads in heads_to_prune.items():
            union_heads = set(self.config2.pruned_heads.get(layer, [])) | set(
                heads)
            self.config2.pruned_heads[layer] = list(
                union_heads
            )  # Unfortunately we have to store it as list for JSON

        self.base_model._prune_heads(heads_to_prune)


class FunnelClassificationHead(nn.Layer):
    def __init__(self, config, n_labels):
        super().__init__()
        self.linear_hidden = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.linear_out = nn.Linear(config.d_model, n_labels)

    def forward(self, hidden):
        hidden = self.linear_hidden(hidden)
        hidden = paddle.tanh(hidden)
        hidden = self.dropout(hidden)
        return self.linear_out(hidden)


class FunnelForPreTrainingOutput(OrderedDict):
    """
    Output type of :class:`~hf_paddle.FunnelForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``paddle.Tensor`` of shape :obj:`(1,)`):
            Total loss of the ELECTRA-style objective.
        logits (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Prediction scores of the head (scores for each token before SoftMax).
        hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss = None
    logits = None
    hidden_states = None
    attentions = None


class FunnelBaseModel(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, PreTrainedModel):
            config = config.config
        if isinstance(config, dict):
            config = FunnelConfig(**config)
        self.config2 = config
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        output_attentions = output_attentions if output_attentions is not None else self.config2.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config2.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

        # TODO: deal with head_mask
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        return encoder_outputs


@register_base_model
class FunnelModel(FunnelPreTrainedModel):
    base_model_prefix = "model"

    def __init__(self, **config):
        super().__init__()
        if isinstance(config, PreTrainedModel):
            config = config.config
        if isinstance(config, dict):
            config = FunnelConfig(**config)
        self.config2 = config
        # self.config2.hidden_dropout=0
        # self.config2.attention_dropout=0
        # self.config2.activation_dropout=0
        self.embeddings = FunnelEmbeddings(config)
        self.encoder = FunnelEncoder(config)
        self.decoder = FunnelDecoder(config)

        # self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):

        output_attentions = output_attentions if output_attentions is not None else self.config2.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else
                                self.config2.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if attention_mask is None:
            attention_mask = paddle.ones(input_shape)
        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)
        else:
            token_type_ids = token_type_ids.astype("int64")

        # TODO: deal with head_mask
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
            encoder_outputs = self.encoder(
                inputs_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict, )

        decoder_outputs = self.decoder(
            final_hidden=encoder_outputs.last_hidden_state,
            first_block_hidden=encoder_outputs.hidden_states[
                self.config2.block_sizes[0]],
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        if not return_dict:
            idx = 0
            outputs = (decoder_outputs.last_hidden_state, )
            if output_hidden_states:
                idx += 1
                outputs = outputs + (encoder_outputs.hidden_states +
                                     decoder_outputs[idx], )
            if output_attentions:
                idx += 1
                outputs = outputs + (encoder_outputs.attentions +
                                     decoder_outputs[idx], )
            return outputs

        return BaseModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            hidden_states=(
                encoder_outputs.hidden_states + decoder_outputs.hidden_states)
            if output_hidden_states else None,
            attentions=(encoder_outputs.attentions + decoder_outputs.attentions)
            if output_attentions else None, )


class FunnelForPreTraining(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.funnel = FunnelModel(config)
        self.discriminator_predictions = FunnelDiscriminatorPredictions(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        labels (``paddle.Tensor`` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the ELECTRA-style loss. Input should be a sequence of tokens (see :obj:`input_ids`
            docstring) Indices should be in ``[0, 1]``:

            - 0 indicates the token is an original token,
            - 1 indicates the token was replaced.

        Returns:


        """
        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        discriminator_hidden_states = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            if attention_mask is not None:
                active_loss = attention_mask.reshape(
                    -1, discriminator_sequence_output.shape[1]) == 1
                active_logits = logits.reshape(
                    -1, discriminator_sequence_output.shape[1])[active_loss]
                active_labels = labels[active_loss]
                loss = loss_fct(active_logits, active_labels.astype("float32"))
            else:
                loss = loss_fct(
                    logits.reshape(-1, discriminator_sequence_output.shape[1]),
                    labels.astype("float32"))

        if not return_dict:
            output = (logits, ) + discriminator_hidden_states[1:]
            return ((loss, ) + output) if loss is not None else output

        return FunnelForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions, )


class FunnelForMaskedLM(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, PreTrainedModel):
            config = config.config
        if isinstance(config, dict):
            config = FunnelConfig(**config)

        self.funnel = FunnelModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        last_hidden_state = outputs[0]
        prediction_logits = self.lm_head(last_hidden_state)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_logits.reshape(-1, self.config2.vocab_size),
                labels.reshape(-1))

        if not return_dict:
            output = (prediction_logits, ) + outputs[1:]
            return ((masked_lm_loss, ) + output
                    ) if masked_lm_loss is not None else output

        return prediction_logits


class FunnelForSequenceClassification(FunnelPreTrainedModel):
    base_model_class = FunnelModel

    def __init__(self, basemodel, num_classes=2):
        super().__init__()
        config = basemodel.init_config
        self.num_classes = num_classes
        if isinstance(config, PreTrainedModel):
            config = config.config
        if isinstance(config, dict):
            config = FunnelConfig(**config)
        self.num_labels = config.num_labels
        self.config2 = config
        # self.config2.hidden_dropout=0
        # self.config2.attention_dropout=0
        # self.config2.activation_dropout=0

        self.funnel = FunnelBaseModel(config)
        self.classifier = FunnelClassificationHead(config, config.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config2.problem_type is None:
                if self.num_labels == 1:
                    self.config2.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == paddle.int64 or
                                              labels.dtype == paddle.int32):
                    self.config2.problem_type = "single_label_classification"
                else:
                    self.config2.problem_type = "multi_label_classification"

            if self.config2.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config2.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.reshape(-1, self.num_labels), labels.reshape(-1))
            elif self.config2.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return logits


class FunnelForMultipleChoice(FunnelPreTrainedModel):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, PreTrainedModel):
            config = config.config
        if isinstance(config, dict):
            config = FunnelConfig(**config)

        self.funnel = FunnelBaseModel(config)
        self.classifier = FunnelClassificationHead(config, 1)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)
        """

        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict
        num_choices = input_ids.shape[
            1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.reshape(
            -1, input_ids.shape[-1]) if input_ids is not None else None
        attention_mask = attention_mask.reshape(
            -1,
            attention_mask.shape[-1]) if attention_mask is not None else None
        token_type_ids = token_type_ids.reshape(
            -1,
            token_type_ids.shape[-1]) if token_type_ids is not None else None
        inputs_embeds = (inputs_embeds.reshape(-1, inputs_embeds.shape[-2],
                                               inputs_embeds.shape[-1])
                         if inputs_embeds is not None else None)

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        last_hidden_state = outputs[0]
        pooled_output = last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.reshape(-1, num_choices)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits, ) + outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return reshaped_logits


class FunnelForTokenClassification(FunnelPreTrainedModel):
    base_model_class = FunnelModel

    def __init__(self, basemodel, num_classes=2):
        config = basemodel.init_config
        super().__init__()
        if isinstance(config, PreTrainedModel):
            config = config.config
        if isinstance(config, dict):
            config = FunnelConfig(**config)
        self.num_labels = config.num_labels

        self.funnel = FunnelModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.num_classes = num_classes
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        labels (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """

        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        last_hidden_state = outputs[0]
        last_hidden_state = self.dropout(last_hidden_state)
        logits = self.classifier(last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.reshape(-1) == 1
                active_logits = logits.reshape(-1, self.num_labels)
                active_labels = paddle.where(
                    active_loss,
                    labels.reshape(-1),
                    paddle.tensor(loss_fct.ignore_index).astype(labels.dtype))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(
                    logits.reshape(-1, self.num_labels),
                    paddle.reshape(labels, -1))

        if not return_dict:
            output = (logits, ) + outputs[1:]
            return ((loss, ) + output) if loss is not None else output

        return logits


class FunnelForQuestionAnswering(FunnelPreTrainedModel):
    base_model_class = FunnelModel

    def __init__(self, basemodel):
        config = basemodel.init_config
        super().__init__()
        if isinstance(config, PreTrainedModel):
            config = config.config
        if isinstance(config, dict):
            config = FunnelConfig(**config)
        self.config2 = config
        self.num_labels = config.num_labels

        self.funnel = FunnelModel(**config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None, ):
        r"""
        start_positions (:obj:`paddle.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`paddle.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """

        return_dict = return_dict if return_dict is not None else self.config2.use_return_dict

        outputs = self.funnel(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, )

        last_hidden_state = outputs[0]

        logits = self.qa_outputs(last_hidden_state)
        start_logits, end_logits = split(logits, 1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.shape) > 1:
                start_positions = start_positions.squeze(-1)
            if len(end_positions.shape) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.shape[1]
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return (
                (total_loss, ) + output) if total_loss is not None else output

        return start_logits, end_logits


def is_tensor(x):
    """
    Tests if ``x`` is a :obj:`paddle.Tensor`,   or
    :obj:`np.ndarray`.
    """

    if isinstance(x, paddle.Tensor):
        return True

    return isinstance(x, np.ndarray)


class ModelOutput(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        class_fields = fields(self)

        # Safety and consistency checks
        assert len(class_fields), f"{self.__class__.__name__} has no fields."
        assert all(
            field.default is None for field in class_fields[1:]
        ), f"{self.__class__.__name__} should not have more than one required field."

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(
            getattr(self, field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            try:
                iterator = iter(first_field)
                first_field_iterator = True
            except TypeError:
                first_field_iterator = False

            # if we provided an iterator as first field and the iterator is a (key, value) iterator
            # set the associated fields
            if first_field_iterator:
                for element in iterator:
                    if (not isinstance(element, (list, tuple)) or
                            not len(element) == 2 or
                            not isinstance(element[0], str)):
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = getattr(self, field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance."
        )

    def setdefault(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance."
        )

    def pop(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(
            f"You cannot use ``update`` on a {self.__class__.__name__} instance."
        )

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        if value is not None:
            # Don't call self.__setitem__ to avoid recursion errors
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def to_tuple(self):
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


@dataclass
class BaseModelOutput(ModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (:obj:`paddle.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`paddle.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(paddle.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`paddle.Tensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state = None
    hidden_states = None
    attentions = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
