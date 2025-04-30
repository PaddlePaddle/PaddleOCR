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

from ..utils.cli import (
    get_subcommand_args,
    str2bool,
)
from .base import PaddleXPipelineWrapper, PipelineCLISubcommandExecutor
from .utils import create_config_from_structure


class PPChatOCRv4Doc(PaddleXPipelineWrapper):
    def __init__(
        self,
        layout_detection_model_name=None,
        layout_detection_model_dir=None,
        doc_orientation_classify_model_name=None,
        doc_orientation_classify_model_dir=None,
        doc_unwarping_model_name=None,
        doc_unwarping_model_dir=None,
        text_detection_model_name=None,
        text_detection_model_dir=None,
        text_recognition_model_name=None,
        text_recognition_model_dir=None,
        text_recognition_batch_size=None,
        table_structure_recognition_model_name=None,
        table_structure_recognition_model_dir=None,
        seal_text_detection_model_name=None,
        seal_text_detection_model_dir=None,
        seal_text_recognition_model_name=None,
        seal_text_recognition_model_dir=None,
        seal_text_recognition_batch_size=None,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_general_ocr=None,
        use_seal_recognition=None,
        use_table_recognition=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_rec_score_thresh=None,
        seal_det_limit_side_len=None,
        seal_det_limit_type=None,
        seal_det_thresh=None,
        seal_det_box_thresh=None,
        seal_det_unclip_ratio=None,
        seal_rec_score_thresh=None,
        retriever_config=None,
        mllm_chat_bot_config=None,
        chat_bot_config=None,
        **kwargs,
    ):
        params = locals().copy()
        params.pop("self")
        params.pop("kwargs")
        self._params = params

        super().__init__(**kwargs)

    @property
    def _paddlex_pipeline_name(self):
        return "PP-ChatOCRv4-doc"

    def visual_predict(
        self,
        input,
        *,
        use_doc_orientation_classify=None,
        use_doc_unwarping=None,
        use_general_ocr=None,
        use_seal_recognition=None,
        use_table_recognition=None,
        layout_threshold=None,
        layout_nms=None,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        text_det_limit_side_len=None,
        text_det_limit_type=None,
        text_det_thresh=None,
        text_det_box_thresh=None,
        text_det_unclip_ratio=None,
        text_rec_score_thresh=None,
        seal_det_limit_side_len=None,
        seal_det_limit_type=None,
        seal_det_thresh=None,
        seal_det_box_thresh=None,
        seal_det_unclip_ratio=None,
        seal_rec_score_thresh=None,
        **kwargs,
    ):
        result = []
        for res in self.paddlex_pipeline.visual_predict(
            input,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_general_ocr=use_general_ocr,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
            layout_threshold=layout_threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            text_det_limit_side_len=text_det_limit_side_len,
            text_det_limit_type=text_det_limit_type,
            text_det_thresh=text_det_thresh,
            text_det_box_thresh=text_det_box_thresh,
            text_det_unclip_ratio=text_det_unclip_ratio,
            text_rec_score_thresh=text_rec_score_thresh,
            seal_det_limit_side_len=seal_det_limit_side_len,
            seal_det_limit_type=seal_det_limit_type,
            seal_det_thresh=seal_det_thresh,
            seal_det_box_thresh=seal_det_box_thresh,
            seal_det_unclip_ratio=seal_det_unclip_ratio,
            seal_rec_score_thresh=seal_rec_score_thresh,
            **kwargs,
        ):
            result.append(res)
        return result

    def build_vector(
        self,
        visual_info,
        *,
        min_characters=3500,
        block_size=300,
        flag_save_bytes_vector=False,
        retriever_config=None,
    ):
        return self.paddlex_pipeline.build_vector(
            visual_info,
            min_characters=min_characters,
            block_size=block_size,
            flag_save_bytes_vector=flag_save_bytes_vector,
            retriever_config=retriever_config,
        )

    def mllm_pred(self, input, key_list, *, mllm_chat_bot_config=None):
        return self.paddlex_pipeline.mllm_pred(
            input,
            key_list,
            mllm_chat_bot_config=mllm_chat_bot_config,
        )

    def chat(
        self,
        key_list,
        visual_info,
        *,
        use_vector_retrieval=True,
        vector_info=None,
        min_characters=3500,
        text_task_description=None,
        text_output_format=None,
        text_rules_str=None,
        text_few_shot_demo_text_content=None,
        text_few_shot_demo_key_value_list=None,
        table_task_description=None,
        table_output_format=None,
        table_rules_str=None,
        table_few_shot_demo_text_content=None,
        table_few_shot_demo_key_value_list=None,
        mllm_predict_info=None,
        mllm_integration_strategy="integration",
        chat_bot_config=None,
        retriever_config=None,
    ):
        return self.paddlex_pipeline.chat(
            key_list,
            visual_info,
            use_vector_retrieval=use_vector_retrieval,
            vector_info=vector_info,
            min_characters=min_characters,
            text_task_description=text_task_description,
            text_output_format=text_output_format,
            text_rules_str=text_rules_str,
            text_few_shot_demo_text_content=text_few_shot_demo_text_content,
            text_few_shot_demo_key_value_list=text_few_shot_demo_key_value_list,
            table_task_description=table_task_description,
            table_output_format=table_output_format,
            table_rules_str=table_rules_str,
            table_few_shot_demo_text_content=table_few_shot_demo_text_content,
            table_few_shot_demo_key_value_list=table_few_shot_demo_key_value_list,
            mllm_predict_info=mllm_predict_info,
            mllm_integration_strategy=mllm_integration_strategy,
            chat_bot_config=chat_bot_config,
            retriever_config=retriever_config,
        )

    @classmethod
    def get_cli_subcommand_executor(cls):
        return PPChatOCRv4DocCLISubcommandExecutor()

    def _get_paddlex_config_overrides(self):
        STRUCTURE = {
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.model_name": self._params[
                "layout_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.model_dir": self._params[
                "layout_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_name": self._params[
                "doc_orientation_classify_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.SubModules.DocOrientationClassify.model_dir": self._params[
                "doc_orientation_classify_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_name": self._params[
                "doc_unwarping_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.SubModules.DocUnwarping.model_dir": self._params[
                "doc_unwarping_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.model_name": self._params[
                "text_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.model_dir": self._params[
                "text_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextRecognition.model_name": self._params[
                "text_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextRecognition.model_dir": self._params[
                "text_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextRecognition.batch_size": self._params[
                "text_recognition_batch_size"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.TableStructureRecognition.model_name": self._params[
                "table_structure_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.TableRecognition.SubModules.TableStructureRecognition.model_dir": self._params[
                "table_structure_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.model_name": self._params[
                "seal_text_detection_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.model_dir": self._params[
                "seal_text_detection_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.model_name": self._params[
                "seal_text_recognition_model_name"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.model_dir": self._params[
                "seal_text_recognition_model_dir"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.batch_size": self._params[
                "seal_text_recognition_batch_size"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.use_doc_orientation_classify": self._params[
                "use_doc_orientation_classify"
            ],
            "SubPipelines.LayoutParser.SubPipelines.DocPreprocessor.use_doc_unwarping": self._params[
                "use_doc_unwarping"
            ],
            "SubPipelines.LayoutParser.use_general_ocr": self._params[
                "use_general_ocr"
            ],
            "SubPipelines.LayoutParser.use_seal_recognition": self._params[
                "use_seal_recognition"
            ],
            "SubPipelines.LayoutParser.use_table_recognition": self._params[
                "use_table_recognition"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.threshold": self._params[
                "layout_threshold"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.nms": self._params[
                "layout_nms"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.unclip_ratio": self._params[
                "layout_unclip_ratio"
            ],
            "SubPipelines.LayoutParser.SubModules.LayoutDetection.merge_bboxes_mode": self._params[
                "layout_merge_bboxes_mode"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.limit_side_len": self._params[
                "text_det_limit_side_len"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.limit_type": self._params[
                "text_det_limit_type"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.thresh": self._params[
                "text_det_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.box_thresh": self._params[
                "text_det_box_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextDetection.unclip_ratio": self._params[
                "text_det_unclip_ratio"
            ],
            "SubPipelines.LayoutParser.SubPipelines.GeneralOCR.SubModules.TextRecognition.score_thresh": self._params[
                "text_rec_score_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.limit_side_len": self._params[
                "text_det_limit_side_len"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.limit_type": self._params[
                "seal_det_limit_type"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.thresh": self._params[
                "seal_det_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.box_thresh": self._params[
                "seal_det_box_thresh"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextDetection.unclip_ratio": self._params[
                "seal_det_unclip_ratio"
            ],
            "SubPipelines.LayoutParser.SubPipelines.SealRecognition.SubPipelines.SealOCR.SubModules.TextRecognition.score_thresh": self._params[
                "seal_rec_score_thresh"
            ],
            "SubModules.LLM_Retriever": self._params["retriever_config"],
            "SubModules.MLLM_Chat": self._params["mllm_chat_bot_config"],
            "SubModules.LLM_Chat": self._params["chat_bot_config"],
        }
        return create_config_from_structure(STRUCTURE)


class PPChatOCRv4DocCLISubcommandExecutor(PipelineCLISubcommandExecutor):
    @property
    def subparser_name(self):
        return "pp_chatocrv4_doc"

    def _update_subparser(self, subparser):
        subparser.add_argument(
            "-i",
            "--input",
            type=str,
            required=True,
            help="Input path or URL.",
        )
        subparser.add_argument(
            "-k",
            "--keys",
            type=str,
            nargs="+",
            required=True,
            metavar="KEY",
            help="Keys use for information extraction.",
        )
        subparser.add_argument(
            "--save_path",
            type=str,
            default="output",
            help="Path to the output directory.",
        )
        subparser.add_argument(
            "--invoke_mllm",
            type=str2bool,
            default=False,
            help="Whether to invoke the multimodal large language model.",
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
            "--table_structure_recognition_model_name",
            type=str,
            help="Name of the table structure recognition model.",
        )
        subparser.add_argument(
            "--table_structure_recognition_model_dir",
            type=str,
            help="Path to the table structure recognition model directory.",
        )
        subparser.add_argument(
            "--seal_text_detection_model_name",
            type=str,
            help="Name of the seal text detection model.",
        )
        subparser.add_argument(
            "--seal_text_detection_model_dir",
            type=str,
            help="Path to the seal text detection model directory.",
        )
        subparser.add_argument(
            "--seal_text_recognition_model_name",
            type=str,
            help="Name of the seal text recognition model.",
        )
        subparser.add_argument(
            "--seal_text_recognition_model_dir",
            type=str,
            help="Path to the seal text recognition model directory.",
        )
        subparser.add_argument(
            "--seal_text_recognition_batch_size",
            type=int,
            help="Batch size for the seal text recognition model.",
        )
        subparser.add_argument(
            "--use_doc_orientation_classify",
            type=str2bool,
            help="Whether to use the document image orientation classification model.",
        )
        subparser.add_argument(
            "--use_doc_unwarping",
            type=str2bool,
            help="Whether to use the text image unwarping model.",
        )
        subparser.add_argument(
            "--use_general_ocr",
            type=str2bool,
            help="Whether to use general OCR.",
        )
        subparser.add_argument(
            "--use_seal_recognition",
            type=str2bool,
            help="Whether to use seal recognition.",
        )
        subparser.add_argument(
            "--use_table_recognition",
            type=str2bool,
            help="Whether to use table recognition.",
        )
        # TODO: Support dict and list types
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
            "--text_rec_score_thresh",
            type=float,
            help="Text recognition threshold used in general OCR. Text results with scores greater than this threshold are retained.",
        )
        subparser.add_argument(
            "--seal_det_limit_side_len",
            type=int,
            help="This sets a limit on the side length of the input image for the seal text detection model.",
        )
        subparser.add_argument(
            "--seal_det_limit_type",
            type=str,
            help="This determines how the side length limit is applied to the input image before feeding it into the seal text deteciton model.",
        )
        subparser.add_argument(
            "--seal_det_thresh",
            type=float,
            help="Detection pixel threshold for the seal text detection model. Pixels with scores greater than this threshold in the output probability map are considered text pixels.",
        )
        subparser.add_argument(
            "--seal_det_box_thresh",
            type=float,
            help="Detection box threshold for the seal text detection model. A detection result is considered a text region if the average score of all pixels within the border of the result is greater than this threshold.",
        )
        subparser.add_argument(
            "--seal_det_unclip_ratio",
            type=float,
            help="Seal text detection expansion coefficient, which expands the text region using this method. The larger the value, the larger the expansion area.",
        )
        subparser.add_argument(
            "--seal_rec_score_thresh",
            type=float,
            help="Seal text recognition threshold. Text results with scores greater than this threshold are retained.",
        )

        # FIXME: Passing API key through CLI is not secure; consider using
        # environment variables.
        subparser.add_argument(
            "--qianfan_api_key",
            type=str,
            help="Configuration for the embedding model.",
        )
        subparser.add_argument(
            "--pp_docbee_base_url",
            type=str,
            help="Configuration for the multimodal large language model.",
        )

    def execute_with_args(self, args):
        params = get_subcommand_args(args)
        input = params.pop("input")
        keys = params.pop("keys")
        save_path = params.pop("save_path")
        invoke_mllm = params.pop("invoke_mllm")
        qianfan_api_key = params.pop("qianfan_api_key")
        if qianfan_api_key is not None:
            params["retriever_config"] = {
                "module_name": "retriever",
                "model_name": "embedding-v1",
                "base_url": "https://qianfan.baidubce.com/v2",
                "api_type": "qianfan",
                "api_key": qianfan_api_key,
            }
            params["chat_bot_config"] = {
                "module_name": "chat_bot",
                "model_name": "ernie-3.5-8k",
                "base_url": "https://qianfan.baidubce.com/v2",
                "api_type": "openai",
                "api_key": qianfan_api_key,
            }
        pp_docbee_base_url = params.pop("pp_docbee_base_url")
        if pp_docbee_base_url is not None:
            params["mllm_chat_bot_config"] = {
                "module_name": "chat_bot",
                "model_name": "PP-DocBee",
                # PaddleX requires endpoints such as ".../chat/completions",
                # which, as the parameter name suggests, are not base URLs.
                "base_url": pp_docbee_base_url,
                "api_type": "openai",
                "api_key": "fake_key",
            }

        chatocr = PPChatOCRv4Doc(**params)

        result_visual = chatocr.visual_predict(input)

        visual_info_list = []
        for res in result_visual:
            visual_info_list.append(res["visual_info"])
            if save_path:
                res["layout_parsing_result"].save_all(save_path)

        vector_info = chatocr.build_vector(visual_info_list)

        if invoke_mllm:
            result_mllm = chatocr.mllm_pred(input, keys)
            mllm_predict_info = result_mllm["mllm_res"]
        else:
            mllm_predict_info = None

        result_chat = chatocr.chat(
            keys,
            visual_info_list,
            vector_info=vector_info,
            mllm_predict_info=mllm_predict_info,
        )

        # Print the result to stdout
        for k, v in result_chat["chat_res"].items():
            print(f"{k} {v}")
