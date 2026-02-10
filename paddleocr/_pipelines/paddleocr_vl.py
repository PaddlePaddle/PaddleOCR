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


_AVAILABLE_PIPELINE_VERSIONS = ["v1", "v1.5"]
_DEFAULT_PIPELINE_VERSION = "v1.5"
_SUPPORTED_VL_BACKENDS = [
    "native",
    "vllm-server",
    "sglang-server",
    "fastdeploy-server",
    "mlx-vlm-server",
    "llama-cpp-server",
]


class PaddleOCRVL(PaddleXPipelineWrapper):
    def __init__(
        self,
        pipeline_version=_DEFAULT_PIPELINE_VERSION,
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
        vl_rec_api_model_name=None,
        vl_rec_api_key=None,
        doc_orientation_classify_model_name=None,
        doc_orientation_classify_model_dir=None,
        doc_unwarping_model_name=None,
        doc_unwarping_model_dir=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_layout_detection=None,
        use_chart_recognition=None,
        use_seal_recognition=None,
        use_ocr_for_image_block=None,
        format_block_content=None,
        merge_layout_blocks=None,
        markdown_ignore_labels=None,
        use_queues=None,
        **kwargs,
    ):
        if pipeline_version not in _AVAILABLE_PIPELINE_VERSIONS:
            raise ValueError(
                f"Invalid pipeline version: {pipeline_version}. Supported versions are {_AVAILABLE_PIPELINE_VERSIONS}."
            )

        if vl_rec_backend is not None and vl_rec_backend not in _SUPPORTED_VL_BACKENDS:
            raise ValueError(
                f"Invalid backend for the VL recognition module: {vl_rec_backend}. Supported values are {_SUPPORTED_VL_BACKENDS}."
            )

        params = locals().copy()
        params.pop("self")
        params.pop("pipeline_version")
        params.pop("kwargs")
        self._params = params
        self.pipeline_version = pipeline_version

        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self):
        if self.pipeline_version == "v1":
            return "PaddleOCR-VL"
        elif self.pipeline_version == "v1.5":
            return "PaddleOCR-VL-1.5"
        else:
            raise AssertionError(f"Unknown pipeline version: {self.pipeline_version}")

    def predict_iter(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_layout_detection=None,
        use_chart_recognition=None,
        use_seal_recognition=None,
        use_ocr_for_image_block=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        layout_shape_mode="auto",
        use_queues=None,
        prompt_label=None,
        format_block_content=None,
        repetition_penalty=None,
        temperature=None,
        top_p=None,
        min_pixels=None,
        max_pixels=None,
        max_new_tokens=None,
        merge_layout_blocks=None,
        markdown_ignore_labels=None,
        vlm_extra_args=None,
        **kwargs,
    ):
        return self.paddlex_pipeline.predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_layout_detection=use_layout_detection,
            use_chart_recognition=use_chart_recognition,
            use_seal_recognition=use_seal_recognition,
            use_ocr_for_image_block=use_ocr_for_image_block,
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            layout_shape_mode=layout_shape_mode,
            use_queues=use_queues,
            prompt_label=prompt_label,
            format_block_content=format_block_content,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_p=top_p,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_new_tokens=max_new_tokens,
            merge_layout_blocks=merge_layout_blocks,
            markdown_ignore_labels=markdown_ignore_labels,
            vlm_extra_args=vlm_extra_args,
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
        use_seal_recognition=None,
        use_ocr_for_image_block=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        layout_shape_mode="auto",
        use_queues=None,
        prompt_label=None,
        format_block_content=None,
        repetition_penalty=None,
        temperature=None,
        top_p=None,
        min_pixels=None,
        max_pixels=None,
        max_new_tokens=None,
        merge_layout_blocks=None,
        markdown_ignore_labels=None,
        vlm_extra_args=None,
        **kwargs,
    ):
        return list(
            self.predict_iter(
                input,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping,
                use_layout_detection=use_layout_detection,
                use_chart_recognition=use_chart_recognition,
                use_seal_recognition=use_seal_recognition,
                use_ocr_for_image_block=use_ocr_for_image_block,
                layout_threshold=layout_threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                layout_shape_mode=layout_shape_mode,
                use_queues=use_queues,
                prompt_label=prompt_label,
                format_block_content=format_block_content,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_new_tokens=max_new_tokens,
                merge_layout_blocks=merge_layout_blocks,
                markdown_ignore_labels=markdown_ignore_labels,
                vlm_extra_args=vlm_extra_args,
                **kwargs,
            )
        )

    def concatenate_markdown_pages(self, markdown_list):
        return self.paddlex_pipeline.concatenate_markdown_pages(markdown_list)

    def restructure_pages(
        self, res_list, merge_tables=True, relevel_titles=True, concatenate_pages=False
    ):
        return list(
            self.paddlex_pipeline.restructure_pages(
                res_list,
                merge_tables=merge_tables,
                relevel_titles=relevel_titles,
                concatenate_pages=concatenate_pages,
            )
        )

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
            "merge_layout_blocks": self._params["merge_layout_blocks"],
            "markdown_ignore_labels": self._params["markdown_ignore_labels"],
            "use_queues": self._params["use_queues"],
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
            "SubModules.VLRecognition.genai_config.client_kwargs.model_name": self._params[
                "vl_rec_api_model_name"
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
            "use_seal_recognition": self._params["use_seal_recognition"],
            "use_ocr_for_image_block": self._params["use_ocr_for_image_block"],
        }
        return create_config_from_structure(STRUCTURE)


class PaddleOCRVLCLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "doc_parser"

    def _update_subparser(self, subparser):
        add_simple_inference_args(subparser)

        subparser.add_argument(
            "--pipeline_version",
            type=str,
            default=_DEFAULT_PIPELINE_VERSION,
            choices=_AVAILABLE_PIPELINE_VERSIONS,
        )

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
            type=int,
            help="Maximum concurrency for making VLM requests.",
        )
        subparser.add_argument(
            "--vl_rec_api_model_name",
            type=str,
            help="Model name for the VLM server.",
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
            "--use_seal_recognition",
            type=str2bool,
            help="Whether to use seal recognition.",
        )
        subparser.add_argument(
            "--use_ocr_for_image_block",
            type=str2bool,
            help="Whether to use OCR for image blocks.",
        )
        subparser.add_argument(
            "--format_block_content",
            type=str2bool,
            help="Whether to format block content to Markdown.",
        )
        subparser.add_argument(
            "--merge_layout_blocks",
            type=str2bool,
            help="Whether to merge layout blocks.",
        )
        subparser.add_argument(
            "--markdown_ignore_labels",
            type=str,
            nargs="+",
            help="List of layout labels to ignore in Markdown output.",
        )

        subparser.add_argument(
            "--use_queues",
            type=str2bool,
            help="Whether to use queues for asynchronous processing.",
        )

        subparser.add_argument(
            "--layout_shape_mode",
            type=str,
            default="auto",
            help="Mode for layout shape.",
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
        subparser.add_argument(
            "--max_new_tokens",
            type=int,
            help="Maximum number of tokens generated by the VLM.",
        )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        perform_simple_inference(
            PaddleOCRVL,
            params,
            predict_param_names={
                "layout_shape_mode",
                "prompt_label",
                "repetition_penalty",
                "temperature",
                "top_p",
                "min_pixels",
                "max_pixels",
                "max_new_tokens",
            },
        )
