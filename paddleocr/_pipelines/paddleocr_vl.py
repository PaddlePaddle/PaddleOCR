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

from .._utils.cli import (
    add_simple_inference_args,
    get_subcommand_args,
    perform_simple_inference,
    str2bool,
)
from .base import PaddleXPipelineWrapper, PipelineCLISubcommandExecutor
from .utils import create_config_from_structure


_SUPPORTED_VL_BACKENDS = ["native", "vllm-server", "sglang-server", "fastdeploy-server"]


class PaddleOCRVL(PaddleXPipelineWrapper):
    def __init__(
        self,
        layout_detection_model_name=None,
        layout_detection_model_dir=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        vl_rec_model_name=None,
        vl_rec_model_dir=None,
        vl_rec_backend=None,
        vl_rec_server_url=None,
        vl_rec_max_concurrency=None,
        vl_rec_api_key=None,
        doc_orientation_classify_model_name=None,
        doc_orientation_classify_model_dir=None,
        doc_unwarping_model_name=None,
        doc_unwarping_model_dir=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_layout_detection=None,
        use_chart_recognition=None,
        format_block_content=None,
        **kwargs,
    ):
        if vl_rec_backend is not None and vl_rec_backend not in _SUPPORTED_VL_BACKENDS:
            raise ValueError(
                f"Invalid backend for the VL recognition module: {vl_rec_backend}. Supported values are {_SUPPORTED_VL_BACKENDS}."
            )

        params = locals().copy()
        params.pop("self")
        params.pop("kwargs")
        self._params = params

        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self):
        return "PaddleOCR-VL"

    def predict_iter(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_layout_detection=None,
        use_chart_recognition=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        use_queues=None,
        prompt_label=None,
        format_block_content=None,
        repetition_penalty=None,
        temperature=None,
        top_p=None,
        min_pixels=None,
        max_pixels=None,
        **kwargs,
    ):
        return self.paddlex_pipeline.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_layout_detection=use_layout_detection,
            use_chart_recognition=use_chart_recognition,
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            use_queues=use_queues,
            prompt_label=prompt_label,
            format_block_content=format_block_content,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            **kwargs,
        )

    def predict(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_layout_detection=None,
        use_chart_recognition=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        use_queues=None,
        prompt_label=None,
        format_block_content=None,
        repetition_penalty=None,
        temperature=None,
        top_p=None,
        min_pixels=None,
        max_pixels=None,
        **kwargs,
    ):
        return list(
            self.predict_iter(
                input,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_layout_detection=use_layout_detection,
                use_chart_recognition=use_chart_recognition,
                layout_threshold=layout_threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                use_queues=use_queues,
                prompt_label=prompt_label,
                format_block_content=format_block_content,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                **kwargs,
            )
        )

    def concatenate_markdown_pages(self, markdown_list):
        return self.paddlex_pipeline.concatenate_markdown_pages(markdown_list)

    @classmethod
    def get_cli_subcommand_executor(cls):
        return PaddleOCRVLCLISubcommandExecutor()

    def _get_paddlex_config_overrides(self):
        STRUCTURE = {
            "SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params[
                "use_doc_orientation_classify"
            ],
            "SubPipelines.DocPreprocessor.use_doc_unwarping": self._params[
                "use_doc_unwarping"
            ],
            "use_doc_preprocessor": self._params["use_doc_orientation_classify"]
            or self._params["use_doc_unwarping"],
            "use_layout_detection": self._params["use_layout_detection"],
            "use_chart_recognition": self._params["use_chart_recognition"],
            "format_block_content": self._params["format_block_content"],
            "SubModules.LayoutDetection.model_name": self._params[
                "layout_detection_model_name"
            ],
            "SubModules.LayoutDetection.model_dir": self._params[
                "layout_detection_model_dir"
            ],
            "SubModules.LayoutDetection.threshold": self._params["layout_threshold"],
            "SubModules.LayoutDetection.layout_nms": self._params["layout_nms"],
            "SubModules.LayoutDetection.layout_unclip_ratio": self._params[
                "layout_unclip_ratio"
            ],
            "SubModules.LayoutDetection.layout_merge_bboxes_mode": self._params[
                "layout_merge_bboxes_mode"
            ],
            "SubModules.VLRecognition.model_name": self._params["vl_rec_model_name"],
            "SubModules.VLRecognition.model_dir": self._params["vl_rec_model_dir"],
            "SubModules.VLRecognition.genai_config.backend": self._params[
                "vl_rec_backend"
            ],
            "SubModules.VLRecognition.genai_config.server_url": self._params[
                "vl_rec_server_url"
            ],
            "SubModules.VLRecognition.genai_config.max_concurrency": self._params[
                "vl_rec_max_concurrency"
            ],
            "SubModules.VLRecognition.genai_config.client_kwargs.api_key": self._params[
                "vl_rec_api_key"
            ],
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
        }
        return create_config_from_structure(STRUCTURE)


class PaddleOCRVLCLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "doc_parser"

    def _update_subparser(self, subparser):
        add_simple_inference_args(subparser)

        subparser.add_argument(
            "--layout_detection_model_name",
            type=str,
            help="Name of the layout detection model.",
        )
        subparser.add_argument(
            "--layout_detection_model_dir",
            type=str,
            help="Path to the layout detection model directory.",
        )
        subparser.add_argument(
            "--layout_threshold",
            type=float,
            help="Score threshold for the layout detection model.",
        )
        subparser.add_argument(
            "--layout_nms",
            type=str2bool,
            help="Whether to use NMS in layout detection.",
        )
        subparser.add_argument(
            "--layout_unclip_ratio",
            type=float,
            help="Expansion coefficient for layout detection.",
        )
        subparser.add_argument(
            "--layout_merge_bboxes_mode",
            type=str,
            help="Overlapping box filtering method.",
        )

        subparser.add_argument(
            "--vl_rec_model_name",
            type=str,
            help="Name of the VL recognition model.",
        )
        subparser.add_argument(
            "--vl_rec_model_dir",
            type=str,
            help="Path to the VL recognition model directory.",
        )
        subparser.add_argument(
            "--vl_rec_backend",
            type=str,
            help="Backend used by the VL recognition module.",
            choices=_SUPPORTED_VL_BACKENDS,
        )
        subparser.add_argument(
            "--vl_rec_server_url",
            type=str,
            help="Server URL used by the VL recognition module.",
        )
        subparser.add_argument(
            "--vl_rec_max_concurrency",
            type=str,
            help="Maximum concurrency for making VLM requests.",
        )
        subparser.add_argument(
            "--vl_rec_api_key",
            type=str,
            help="API key for the VLM server.",
        )

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
            "--use_layout_detection",
            type=str2bool,
            help="Whether to use layout detection.",
        )
        subparser.add_argument(
            "--use_chart_recognition",
            type=str2bool,
            help="Whether to use chart recognition.",
        )
        subparser.add_argument(
            "--format_block_content",
            type=str2bool,
            help="Whether to format block content to Markdown.",
        )
        subparser.add_argument(
            "--use_queues",
            type=str2bool,
            help="Whether to use queues for asynchronous processing.",
        )
        subparser.add_argument(
            "--prompt_label",
            type=str,
            help="Prompt label for the VLM.",
        )
        subparser.add_argument(
            "--repetition_penalty",
            type=float,
            help="Repetition penalty used in sampling for the VLM.",
        )
        subparser.add_argument(
            "--temperature",
            type=float,
            help="Temperature parameter used in sampling for the VLM.",
        )
        subparser.add_argument(
            "--top_p",
            type=float,
            help="Top-p parameter used in sampling for the VLM.",
        )
        subparser.add_argument(
            "--min_pixels",
            type=int,
            help="Minimum pixels for image preprocessing for the VLM.",
        )
        subparser.add_argument(
            "--max_pixels",
            type=int,
            help="Maximum pixels for image preprocessing for the VLM.",
        )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        perform_simple_inference(
            PaddleOCRVL,
            params,
            predict_param_names={
                "use_queues",
                "prompt_label",
                "repetition_penalty",
                "temperature",
                "top_p",
                "min_pixels",
                "max_pixels",
            },
        )
