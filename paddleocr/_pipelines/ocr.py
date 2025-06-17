# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

# TODO: Should we use a third-party CLI library to auto-generate command-line
# arguments from the pipeline class, to reduce boilerplate and improve
# maintainability?

import sys
import warnings

from .._utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
    str2bool,
)
from .._utils.deprecation import (
    DeprecatedOptionAction,
    deprecated,
    warn_deprecated_param,
)
from .._utils.logging import logger
from .base import PaddleXPipelineWrapper, PipelineCLISubcommandExecutor
from .utils import create_config_from_structure

_DEPRECATED_PARAM_NAME_MAPPING = {
    "det_model_dir": "text_detection_model_dir",
    "det_limit_side_len": "text_det_limit_side_len",
    "det_limit_type": "text_det_limit_type",
    "det_db_thresh": "text_det_thresh",
    "det_db_box_thresh": "text_det_box_thresh",
    "det_db_unclip_ratio": "text_det_unclip_ratio",
    "rec_model_dir": "text_recognition_model_dir",
    "rec_batch_num": "text_recognition_batch_size",
    "use_angle_cls": "use_textline_orientation",
    "cls_model_dir": "textline_orientation_model_dir",
    "cls_batch_num": "textline_orientation_batch_size",
}

_SUPPORTED_OCR_VERSIONS = ["PP-OCRv3", "PP-OCRv4", "PP-OCRv5"]


# Be comptable with PaddleOCR 2.x interfaces
class PaddleOCR(PaddleXPipelineWrapper):
    def __init__(
        self,
        doc_orientation_classify_model_name=None,
        doc_orientation_classify_model_dir=None,
        doc_unwarping_model_name=None,
        doc_unwarping_model_dir=None,
        text_detection_model_name=None,
        text_detection_model_dir=None,
        textline_orientation_model_name=None,
        textline_orientation_model_dir=None,
        textline_orientation_batch_size=None,
        text_recognition_model_name=None,
        text_recognition_model_dir=None,
        text_recognition_batch_size=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_det_input_shape=None,
        text_rec_score_thresh=None,
        text_rec_input_shape=None,
        lang=None,
        ocr_version=None,
        **kwargs,
    ):
        if ocr_version is not None and ocr_version not in _SUPPORTED_OCR_VERSIONS:
            raise ValueError(
                f"Invalid OCR version: {ocr_version}. Supported values are {_SUPPORTED_OCR_VERSIONS}."
            )

        if all(
            map(
                lambda p: p is None,
                (
                    text_detection_model_name,
                    text_detection_model_dir,
                    text_recognition_model_name,
                    text_recognition_model_dir,
                ),
            )
        ):
            if lang is not None or ocr_version is not None:
                det_model_name, rec_model_name = self._get_ocr_model_names(
                    lang, ocr_version
                )
                if det_model_name is None or rec_model_name is None:
                    raise ValueError(
                        f"No models are available for the language {repr(lang)} and OCR version {repr(ocr_version)}."
                    )
                text_detection_model_name = det_model_name
                text_recognition_model_name = rec_model_name
        else:
            if lang is not None or ocr_version is not None:
                warnings.warn(
                    "`lang` and `ocr_version` will be ignored when model names or model directories are not `None`.",
                    stacklevel=2,
                )

        params = {
            "doc_orientation_classify_model_name": doc_orientation_classify_model_name,
            "doc_orientation_classify_model_dir": doc_orientation_classify_model_dir,
            "doc_unwarping_model_name": doc_unwarping_model_name,
            "doc_unwarping_model_dir": doc_unwarping_model_dir,
            "text_detection_model_name": text_detection_model_name,
            "text_detection_model_dir": text_detection_model_dir,
            "textline_orientation_model_name": textline_orientation_model_name,
            "textline_orientation_model_dir": textline_orientation_model_dir,
            "textline_orientation_batch_size": textline_orientation_batch_size,
            "text_recognition_model_name": text_recognition_model_name,
            "text_recognition_model_dir": text_recognition_model_dir,
            "text_recognition_batch_size": text_recognition_batch_size,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "use_textline_orientation": use_textline_orientation,
            "text_det_limit_side_len": text_det_limit_side_len,
            "text_det_limit_type": text_det_limit_type,
            "text_det_thresh": text_det_thresh,
            "text_det_box_thresh": text_det_box_thresh,
            "text_det_unclip_ratio": text_det_unclip_ratio,
            "text_det_input_shape": text_det_input_shape,
            "text_rec_score_thresh": text_rec_score_thresh,
            "text_rec_input_shape": text_rec_input_shape,
        }
        base_params = {}
        for name, val in kwargs.items():
            if name in _DEPRECATED_PARAM_NAME_MAPPING:
                new_name = _DEPRECATED_PARAM_NAME_MAPPING[name]
                warn_deprecated_param(name, new_name)
                assert (
                    new_name in params
                ), f"{repr(new_name)} is not a valid parameter name."
                if params[new_name] is not None:
                    raise ValueError(
                        f"`{name}` and `{new_name}` are mutually exclusive."
                    )
                params[new_name] = val
            else:
                base_params[name] = val

        self._params = params

        super().__init__(**base_params)

    @property
    def _paddlex_pipeline_name(self):
        return "OCR"

    def predict_iter(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_rec_score_thresh=None,
    ):
        return self.paddlex_pipeline.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_textline_orientation=use_textline_orientation,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
        )

    def predict(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_textline_orientation=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_rec_score_thresh=None,
    ):
        return list(
            self.predict_iter(
                input,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_textline_orientation=use_textline_orientation,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_limit_type=text_det_limit_type,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
                text_rec_score_thresh=text_rec_score_thresh,
            )
        )

    @deprecated("Please use `predict` instead.")
    def ocr(self, img, **kwargs):
        return self.predict(img, **kwargs)

    @classmethod
    def get_cli_subcommand_executor(cls):
        return PaddleOCRCLISubcommandExecutor()

    def _get_paddlex_config_overrides(self):
        STRUCTURE = {
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_name": self._params[
                "doc_orientation_classify_model_name"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_dir": self._params[
                "doc_orientation_classify_model_dir"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name": self._params[
                "doc_unwarping_model_name"
            ],
            "SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir": self._params[
                "doc_unwarping_model_dir"
            ],
            "SubModules.TextDetection.model_name": self._params[
                "text_detection_model_name"
            ],
            "SubModules.TextDetection.model_dir": self._params[
                "text_detection_model_dir"
            ],
            "SubModules.TextLineOrientation.model_name": self._params[
                "textline_orientation_model_name"
            ],
            "SubModules.TextLineOrientation.model_dir": self._params[
                "textline_orientation_model_dir"
            ],
            "SubModules.TextLineOrientation.batch_size": self._params[
                "textline_orientation_batch_size"
            ],
            "SubModules.TextRecognition.model_name": self._params[
                "text_recognition_model_name"
            ],
            "SubModules.TextRecognition.model_dir": self._params[
                "text_recognition_model_dir"
            ],
            "SubModules.TextRecognition.batch_size": self._params[
                "text_recognition_batch_size"
            ],
            "SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params[
                "use_doc_orientation_classify"
            ],
            "SubPipelines.DocPreprocessor.use_doc_unwarping": self._params[
                "use_doc_unwarping"
            ],
            "use_textline_orientation": self._params["use_textline_orientation"],
            "SubModules.TextDetection.limit_side_len": self._params[
                "text_det_limit_side_len"
            ],
            "SubModules.TextDetection.limit_type": self._params["text_det_limit_type"],
            "SubModules.TextDetection.thresh": self._params["text_det_thresh"],
            "SubModules.TextDetection.box_thresh": self._params["text_det_box_thresh"],
            "SubModules.TextDetection.unclip_ratio": self._params[
                "text_det_unclip_ratio"
            ],
            "SubModules.TextDetection.input_shape": self._params[
                "text_det_input_shape"
            ],
            "SubModules.TextRecognition.score_thresh": self._params[
                "text_rec_score_thresh"
            ],
            "SubModules.TextRecognition.input_shape": self._params[
                "text_rec_input_shape"
            ],
        }
        return create_config_from_structure(STRUCTURE)

    def _get_ocr_model_names(self, lang, ppocr_version):
        LATIN_LANGS = [
            "af",
            "az",
            "bs",
            "cs",
            "cy",
            "da",
            "de",
            "es",
            "et",
            "fr",
            "ga",
            "hr",
            "hu",
            "id",
            "is",
            "it",
            "ku",
            "la",
            "lt",
            "lv",
            "mi",
            "ms",
            "mt",
            "nl",
            "no",
            "oc",
            "pi",
            "pl",
            "pt",
            "ro",
            "rs_latin",
            "sk",
            "sl",
            "sq",
            "sv",
            "sw",
            "tl",
            "tr",
            "uz",
            "vi",
            "french",
            "german",
        ]
        ARABIC_LANGS = ["ar", "fa", "ug", "ur"]
        CYRILLIC_LANGS = [
            "ru",
            "rs_cyrillic",
            "be",
            "bg",
            "uk",
            "mn",
            "abq",
            "ady",
            "kbd",
            "ava",
            "dar",
            "inh",
            "che",
            "lbe",
            "lez",
            "tab",
        ]
        DEVANAGARI_LANGS = [
            "hi",
            "mr",
            "ne",
            "bh",
            "mai",
            "ang",
            "bho",
            "mah",
            "sck",
            "new",
            "gom",
            "sa",
            "bgc",
        ]
        SPECIFIC_LANGS = [
            "ch",
            "en",
            "korean",
            "japan",
            "chinese_cht",
            "te",
            "ka",
            "ta",
        ]

        if lang is None:
            lang = "ch"

        if ppocr_version is None:
            if lang in ("ch", "chinese_cht", "en", "japan"):
                ppocr_version = "PP-OCRv5"
            elif lang in (
                LATIN_LANGS
                + ARABIC_LANGS
                + CYRILLIC_LANGS
                + DEVANAGARI_LANGS
                + SPECIFIC_LANGS
            ):
                ppocr_version = "PP-OCRv3"
            else:
                # Unknown language specified
                return None, None

        if ppocr_version == "PP-OCRv5":
            if lang in ("ch", "chinese_cht", "en", "japan"):
                return "PP-OCRv5_server_det", "PP-OCRv5_server_rec"
            else:
                return None, None
        elif ppocr_version == "PP-OCRv4":
            if lang == "ch":
                return "PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec"
            elif lang == "en":
                return "PP-OCRv4_mobile_det", "en_PP-OCRv4_mobile_rec"
            else:
                return None, None
        else:
            # PP-OCRv3
            rec_lang = None
            if lang in LATIN_LANGS:
                rec_lang = "latin"
            elif lang in ARABIC_LANGS:
                rec_lang = "arabic"
            elif lang in CYRILLIC_LANGS:
                rec_lang = "cyrillic"
            elif lang in DEVANAGARI_LANGS:
                rec_lang = "devanagari"
            else:
                if lang in SPECIFIC_LANGS:
                    rec_lang = lang

            rec_model_name = None
            if rec_lang == "ch":
                rec_model_name = "PP-OCRv3_mobile_rec"
            elif rec_lang is not None:
                rec_model_name = f"{rec_lang}_PP-OCRv3_mobile_rec"
            return "PP-OCRv3_mobile_det", rec_model_name


class PaddleOCRCLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "ocr"

    def _update_subparser(self, subparser):
        add_simple_inference_args(subparser)

        subparser.add_argument(
            "--doc_orientation_classify_model_name",
            type=str,
            help="Name of the document image orientation classification model.",
        )
        subparser.add_argument(
            "--doc_orientation_classify_model_dir",
            type=str,
            help="Path to the document image orientation classification model directory.",
        )
        subparser.add_argument(
            "--doc_unwarping_model_name",
            type=str,
            help="Name of the text image unwarping model.",
        )
        subparser.add_argument(
            "--doc_unwarping_model_dir",
            type=str,
            help="Path to the image unwarping model directory.",
        )
        subparser.add_argument(
            "--text_detection_model_name",
            type=str,
            help="Name of the text detection model.",
        )
        subparser.add_argument(
            "--text_detection_model_dir",
            type=str,
            help="Path to the text detection model directory.",
        )
        subparser.add_argument(
            "--textline_orientation_model_name",
            type=str,
            help="Name of the text line orientation classification model.",
        )
        subparser.add_argument(
            "--textline_orientation_model_dir",
            type=str,
            help="Path to the text line orientation classification model directory.",
        )
        subparser.add_argument(
            "--textline_orientation_batch_size",
            type=int,
            help="Batch size for the text line orientation classification model.",
        )
        subparser.add_argument(
            "--text_recognition_model_name",
            type=str,
            help="Name of the text recognition model.",
        )
        subparser.add_argument(
            "--text_recognition_model_dir",
            type=str,
            help="Path to the text recognition model directory.",
        )
        subparser.add_argument(
            "--text_recognition_batch_size",
            type=int,
            help="Batch size for the text recognition model.",
        )
        subparser.add_argument(
            "--use_doc_orientation_classify",
            type=str2bool,
            help="Whether to use document image orientation classification.",
        )
        subparser.add_argument(
            "--use_doc_unwarping",
            type=str2bool,
            help="Whether to use text image unwarping.",
        )
        subparser.add_argument(
            "--use_textline_orientation",
            type=str2bool,
            help="Whether to use text line orientation classification.",
        )
        subparser.add_argument(
            "--text_det_limit_side_len",
            type=int,
            help="This sets a limit on the side length of the input image for the text detection model.",
        )
        subparser.add_argument(
            "--text_det_limit_type",
            type=str,
            help="This determines how the side length limit is applied to the input image before feeding it into the text deteciton model.",
        )
        subparser.add_argument(
            "--text_det_thresh",
            type=float,
            help="Detection pixel threshold for the text detection model. Pixels with scores greater than this threshold in the output probability map are considered text pixels.",
        )
        subparser.add_argument(
            "--text_det_box_thresh",
            type=float,
            help="Detection box threshold for the text detection model. A detection result is considered a text region if the average score of all pixels within the border of the result is greater than this threshold.",
        )
        subparser.add_argument(
            "--text_det_unclip_ratio",
            type=float,
            help="Text detection expansion coefficient, which expands the text region using this method. The larger the value, the larger the expansion area.",
        )
        subparser.add_argument(
            "--text_det_input_shape",
            nargs=3,
            type=int,
            metavar=("C", "H", "W"),
            help="Input shape of the text detection model.",
        )
        subparser.add_argument(
            "--text_rec_score_thresh",
            type=float,
            help="Text recognition threshold. Text results with scores greater than this threshold are retained.",
        )
        subparser.add_argument(
            "--text_rec_input_shape",
            nargs=3,
            type=int,
            metavar=("C", "H", "W"),
            help="Input shape of the text recognition model.",
        )
        subparser.add_argument(
            "--lang", type=str, help="Language in the input image for OCR processing."
        )
        subparser.add_argument(
            "--ocr_version",
            type=str,
            choices=_SUPPORTED_OCR_VERSIONS,
            help="PP-OCR version to use.",
        )

        deprecated_arg_types = {
            "det_model_dir": str,
            "det_limit_side_len": int,
            "det_limit_type": str,
            "det_db_thresh": float,
            "det_db_box_thresh": float,
            "det_db_unclip_ratio": float,
            "rec_model_dir": str,
            "rec_batch_num": int,
            "use_angle_cls": str2bool,
            "cls_model_dir": str,
            "cls_batch_num": int,
        }

        for name, new_name in _DEPRECATED_PARAM_NAME_MAPPING.items():
            assert name in deprecated_arg_types, name
            subparser.add_argument(
                "--" + name,
                action=DeprecatedOptionAction,
                type=str,
                help=f"[Deprecated] Please use `--{new_name}` instead.",
            )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        for name, new_name in _DEPRECATED_PARAM_NAME_MAPPING.items():
            assert name in params
            val = params[name]
            new_val = params[new_name]
            if val is not None and new_val is not None:
                logger.error(
                    "`--%s` and `--%s` are mutually exclusive.", name, new_name
                )
                sys.exit(2)
            if val is None:
                params.pop(name)

        perform_simple_inference(PaddleOCR, params)
