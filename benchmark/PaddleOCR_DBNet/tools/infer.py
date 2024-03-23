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

import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))

import cv2
import paddle
from paddle import inference
import numpy as np
from PIL import Image

from paddle.vision import transforms
from tools.predict import resize_image
from post_processing import get_post_processing
from utils.util import draw_bbox, save_result


class InferenceEngine(object):
    """InferenceEngine
    
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensor, self.output_tensor = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

        # build transforms
        self.transforms = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # wamrup
        if self.args.warmup > 0:
            for idx in range(args.warmup):
                print(idx)
                x = np.random.rand(1, 3, self.args.crop_size,
                                   self.args.crop_size).astype("float32")
                self.input_tensor.copy_from_cpu(x)
                self.predictor.run()
                self.output_tensor.copy_to_cpu()

        self.post_process = get_post_processing({
            'type': 'SegDetectorRepresenter',
            'args': {
                'thresh': 0.3,
                'box_thresh': 0.7,
                'max_candidates': 1000,
                'unclip_ratio': 1.5
            }
        })

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.use_gpu:
            config.enable_use_gpu(1000, 0)
            if args.use_tensorrt:
                config.enable_tensorrt_engine(
                    workspace_size=1 << 30,
                    precision_mode=precision,
                    max_batch_size=args.max_batch_size,
                    min_subgraph_size=args.
                    min_subgraph_size,  # skip the minmum trt subgraph
                    use_calib_mode=False)

                # collect shape
                trt_shape_f = os.path.join(model_dir, "_trt_dynamic_shape.txt")

                if not os.path.exists(trt_shape_f):
                    config.collect_shape_range_info(trt_shape_f)
                    logger.info(
                        f"collect dynamic shape info into : {trt_shape_f}")
                try:
                    config.enable_tuned_tensorrt_dynamic_shape(trt_shape_f,
                                                               True)
                except Exception as E:
                    logger.info(E)
                    logger.info("Please keep your paddlepaddle-gpu >= 2.3.0!")
        else:
            config.disable_gpu()
            # The thread num should not be greater than the number of cores in the CPU.
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if args.precision == "fp16":
                    config.enable_mkldnn_bfloat16()
                if hasattr(args, "cpu_threads"):
                    config.set_cpu_math_library_num_threads(args.cpu_threads)
                else:
                    # default cpu threads as 10
                    config.set_cpu_math_library_num_threads(10)

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_names = predictor.get_input_names()
        input_tensor = predictor.get_input_handle(input_names[0])

        output_names = predictor.get_output_names()
        output_tensor = predictor.get_output_handle(output_names[0])

        return predictor, config, input_tensor, output_tensor

    def preprocess(self, img_path, short_size):
        """preprocess
        Preprocess to the input.
        Args:
            img_path: Image path.
        Returns: Input data after preprocess.
        """
        img = cv2.imread(img_path, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        img = resize_image(img, short_size)
        img = self.transforms(img)
        img = np.expand_dims(img, axis=0)
        shape_info = {'shape': [(h, w)]}
        return img, shape_info

    def postprocess(self, x, shape_info, is_output_polygon):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            x: Inference engine output.
        Returns: Output data after argmax.
        """
        box_list, score_list = self.post_process(
            shape_info, x, is_output_polygon=is_output_polygon)
        box_list, score_list = box_list[0], score_list[0]
        if len(box_list) > 0:
            if is_output_polygon:
                idx = [x.sum() > 0 for x in box_list]
                box_list = [box_list[i] for i, v in enumerate(idx) if v]
                score_list = [score_list[i] for i, v in enumerate(idx) if v]
            else:
                idx = box_list.reshape(box_list.shape[0], -1).sum(
                    axis=1) > 0  # 去掉全为0的框
                box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        return box_list, score_list

    def run(self, x):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensor.copy_from_cpu(x)
        self.predictor.run()
        output = self.output_tensor.copy_to_cpu()
        return output


def get_args(add_help=True):
    """
    parse args
    """
    import argparse

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(
        description="PaddlePaddle Classification Training", add_help=add_help)

    parser.add_argument("--model_dir", default=None, help="inference model dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--short_size", default=1024, type=int, help="short size")
    parser.add_argument("--img_path", default="./images/demo.jpg")

    parser.add_argument(
        "--benchmark", default=False, type=str2bool, help="benchmark")
    parser.add_argument("--warmup", default=0, type=int, help="warmup iter")
    parser.add_argument(
        '--polygon', action='store_true', help='output polygon or box')

    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
    parser.add_argument("--cpu_threads", type=int, default=10)

    args = parser.parse_args()
    return args


def main(args):
    """
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        class_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="db",
            batch_size=args.batch_size,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.use_gpu else None)

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # preprocess
    img, shape_info = inference_engine.preprocess(args.img_path,
                                                  args.short_size)

    if args.benchmark:
        autolog.times.stamp()

    output = inference_engine.run(img)

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    box_list, score_list = inference_engine.postprocess(output, shape_info,
                                                        args.polygon)

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()

    img = draw_bbox(cv2.imread(args.img_path)[:, :, ::-1], box_list)
    # 保存结果到路径
    os.makedirs('output', exist_ok=True)
    img_path = pathlib.Path(args.img_path)
    output_path = os.path.join('output', img_path.stem + '_infer_result.jpg')
    cv2.imwrite(output_path, img[:, :, ::-1])
    save_result(
        output_path.replace('_infer_result.jpg', '.txt'), box_list, score_list,
        args.polygon)


if __name__ == "__main__":
    args = get_args()
    main(args)
