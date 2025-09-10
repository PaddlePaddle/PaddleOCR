# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import cv2
import numpy as np
import json
import paddle
from PIL import Image, ImageDraw, ImageFont
import math
from paddle import inference

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import tools.infer.utility as utility
from ppocr.postprocess.omniparser_postprocess import OmniParserPostProcess
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read

logger = get_logger()


class OmniParserPredictor(object):
    def __init__(self, args):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.use_onnx = args.use_onnx
        pre_process_list = [{
            'OmniParserDataProcess': {
                'image_shape': [1024, 1024],
                'augmentation': False,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
            }
        }]
        postprocess_params = {
            'name': 'OmniParserPostProcess',
            'mode': args.mode,
            'text_threshold': args.text_threshold,
            'center_threshold': args.center_threshold,
            'border_threshold': args.border_threshold,
            'structure_thresh': args.structure_thresh,
            'boundary_thresh': args.boundary_thresh,
        }
        
        # Initialize post-processor
        self.postprocess_op = OmniParserPostProcess(**postprocess_params)
        
        # Load model
        self.model = self.init_model(args)
        
    def init_model(self, args):
        """Initialize the inference model"""
        if args.use_onnx:
            # ONNX model initialization
            try:
                import onnxruntime as ort
                model_file_path = args.det_model_dir
                if not os.path.exists(model_file_path):
                    raise ValueError("Model file not found: {}".format(model_file_path))
                sess = ort.InferenceSession(model_file_path)
                return sess
            except Exception as e:
                logger.error(f"Failed to initialize ONNX model: {e}")
                raise
        else:
            # Paddle model initialization
            model_file_path = args.det_model_dir
            params_file_path = args.det_model_dir
            
            if not os.path.exists(model_file_path):
                model_file_path = os.path.join(args.det_model_dir, 'inference.pdmodel')
                params_file_path = os.path.join(args.det_model_dir, 'inference.pdiparams')
                
            if not os.path.exists(model_file_path) or not os.path.exists(params_file_path):
                raise ValueError("Model files not found: {} or {}".format(
                    model_file_path, params_file_path))
                    
            config = inference.Config(model_file_path, params_file_path)
            
            if args.use_gpu:
                config.enable_use_gpu(args.gpu_mem, args.gpu_id)
            else:
                config.disable_gpu()
                if args.enable_mkldnn:
                    # Requires PaddlePaddle 2.0+
                    config.enable_mkldnn()
                    config.set_cpu_math_library_num_threads(args.cpu_threads)
                    
            # Enable memory optimization
            config.enable_memory_optim()
            config.disable_glog_info()
            
            # Use zero copy to improve performance
            config.switch_use_feed_fetch_ops(False)
            
            # Create predictor
            predictor = inference.create_predictor(config)
            
            # Get input and output tensors
            input_names = predictor.get_input_names()
            self.input_tensor = predictor.get_input_handle(input_names[0])
            output_names = predictor.get_output_names()
            self.output_tensors = []
            for output_name in output_names:
                self.output_tensors.append(
                    predictor.get_output_handle(output_name))
            
            return predictor
    
    def preprocess(self, img):
        """Preprocess the input image"""
        # Resize image
        h, w = img.shape[:2]
        ratio_h = 1024 / h
        ratio_w = 1024 / w
        
        if self.args.keep_ratio:
            # Keep aspect ratio
            scale = min(ratio_h, ratio_w)
            resize_h = int(h * scale)
            resize_w = int(w * scale)
            
            resize_img = cv2.resize(img, (resize_w, resize_h))
            
            # Create new empty image with target size
            new_img = np.zeros((1024, 1024, 3), dtype=np.float32)
            new_img[:resize_h, :resize_w, :] = resize_img
            
            ratio_h = resize_h / h
            ratio_w = resize_w / w
        else:
            # Direct resize to target size
            resize_img = cv2.resize(img, (1024, 1024))
            new_img = resize_img
        
        # Normalize image
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        
        new_img = new_img.astype(np.float32) / 255.0
        new_img -= mean
        new_img /= std
        
        # Transpose from HWC to CHW format
        new_img = new_img.transpose(2, 0, 1)
        
        # Add batch dimension (NCHW)
        new_img = new_img[np.newaxis, :]
        
        # Return processed image and resize ratios
        return new_img, (ratio_h, ratio_w)
    
    def extract_preds_from_tensors(self, output_tensors):
        """Extract predictions from output tensors"""
        preds = {}
        
        if self.args.mode in ['text', 'all'] and len(output_tensors) >= 3:
            # Text detection outputs
            preds['text_prob'] = output_tensors[0]
            preds['center_prob'] = output_tensors[1]
            preds['border_prob'] = output_tensors[2]
            
        if self.args.mode in ['table', 'all'] and len(output_tensors) >= 5:
            # Table recognition outputs
            preds['structure_pred'] = output_tensors[3]
            preds['boundary_pred'] = output_tensors[4]
            
        if self.args.mode in ['kie', 'all'] and len(output_tensors) >= 6:
            # KIE outputs
            preds['kie_features'] = output_tensors[5]
            
        return preds
            
    def run_onnx(self, img):
        """Run inference with ONNX model"""
        input_data, (ratio_h, ratio_w) = self.preprocess(img)
        
        # Run ONNX inference
        input_name = self.model.get_inputs()[0].name
        output_names = [output.name for output in self.model.get_outputs()]
        outputs = self.model.run(output_names, {input_name: input_data})
        
        # Process outputs
        preds = {}
        for i, output_name in enumerate(output_names):
            if 'text' in output_name:
                preds['text_prob'] = outputs[i]
            elif 'center' in output_name:
                preds['center_prob'] = outputs[i]
            elif 'border' in output_name:
                preds['border_prob'] = outputs[i]
            elif 'structure' in output_name:
                preds['structure_pred'] = outputs[i]
            elif 'boundary' in output_name:
                preds['boundary_pred'] = outputs[i]
            elif 'kie' in output_name:
                preds['kie_features'] = outputs[i]
                
        # Post-process
        data = {'ratio_h': ratio_h, 'ratio_w': ratio_w}
        result = self.postprocess_op(preds, data)
        
        return result
    
    def run_paddle(self, img):
        """Run inference with Paddle model"""
        input_data, (ratio_h, ratio_w) = self.preprocess(img)
        
        # Set input data
        self.input_tensor.copy_from_cpu(input_data)
        
        # Run inference
        self.model.run()
        
        # Get outputs
        outputs = []
        for output_tensor in self.output_tensors:
            output = output_tensor.copy_to_cpu()
            outputs.append(output)
            
        # Process outputs
        preds = self.extract_preds_from_tensors(outputs)
        
        # Post-process
        data = {'ratio_h': ratio_h, 'ratio_w': ratio_w}
        result = self.postprocess_op(preds, data)
        
        return result
    
    def __call__(self, img):
        """Run inference on input image"""
        if isinstance(img, str):
            # Load image from file
            image_file = img
            img, flag, _ = check_and_read(image_file)
            if not flag:
                raise ValueError(f"Error in loading image: {image_file}")
                
        if self.use_onnx:
            result = self.run_onnx(img)
        else:
            result = self.run_paddle(img)
            
        return result
            
    def visualize(self, image, result, output_path):
        """Visualize the prediction results on the image"""
        # Create a copy for visualization
        vis_img = image.copy()
        
        # Draw text boxes
        if 'text_boxes' in result and result['text_boxes']:
            for box in result['text_boxes']:
                x1, y1, x2, y2 = box
                # Draw rectangle for text
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw table structure
        if 'table_structure' in result and result['table_structure']['table_region']:
            table_region = result['table_structure']['table_region']
            row_positions = result['table_structure']['row_positions']
            col_positions = result['table_structure']['col_positions']
            
            # Draw table boundary
            cv2.rectangle(
                vis_img, 
                (table_region[0], table_region[1]), 
                (table_region[2], table_region[3]), 
                (255, 0, 0), 
                2)
                
            # Draw rows
            for y in row_positions:
                cv2.line(
                    vis_img, 
                    (table_region[0], y), 
                    (table_region[2], y), 
                    (0, 0, 255), 
                    1)
                    
            # Draw columns
            for x in col_positions:
                cv2.line(
                    vis_img, 
                    (x, table_region[1]), 
                    (x, table_region[3]), 
                    (0, 0, 255), 
                    1)
        
        # Save visualization
        cv2.imwrite(output_path, vis_img)
        logger.info(f"Visualization saved to {output_path}")
        
        return vis_img


def main():
    import argparse
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input images")
    parser.add_argument("--det_model_dir", type=str, required=True, help="Path to detection model directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    
    # Optional parameters
    parser.add_argument("--mode", type=str, default="all", help="Mode: all, text, table, or kie")
    parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU for inference")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device id")
    parser.add_argument("--gpu_mem", type=int, default=4000, help="GPU memory allocation")
    parser.add_argument("--enable_mkldnn", type=bool, default=False, help="Enable MKLDNN acceleration")
    parser.add_argument("--cpu_threads", type=int, default=10, help="CPU threads for MKLDNN")
    parser.add_argument("--use_onnx", type=bool, default=False, help="Use ONNX for inference")
    parser.add_argument("--det_algorithm", type=str, default="OmniParser", help="Detection algorithm")
    parser.add_argument("--keep_ratio", type=bool, default=True, help="Keep aspect ratio during resizing")
    parser.add_argument("--text_threshold", type=float, default=0.5, help="Text detection threshold")
    parser.add_argument("--center_threshold", type=float, default=0.5, help="Center line detection threshold")
    parser.add_argument("--border_threshold", type=float, default=0.5, help="Border detection threshold")
    parser.add_argument("--structure_thresh", type=float, default=0.5, help="Table structure detection threshold")
    parser.add_argument("--boundary_thresh", type=float, default=0.5, help="Table boundary detection threshold")
    parser.add_argument("--visualize", type=bool, default=True, help="Visualize results")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = OmniParserPredictor(args)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Get image file list
    image_list = get_image_file_list(args.image_dir)
    logger.info(f"Total images: {len(image_list)}")
    
    # Process each image
    for image_path in image_list:
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        img, flag, _ = check_and_read(image_path)
        if not flag:
            logger.warning(f"Error in loading image: {image_path}, skipping...")
            continue
        
        # Run inference
        result = predictor(img)
        
        # Save results
        basename = os.path.basename(image_path)
        basename, ext = os.path.splitext(basename)
        
        # Save JSON results
        json_path = os.path.join(args.output, f"{basename}_result.json")
        with open(json_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_result = {}
            
            if 'text_boxes' in result:
                serializable_result['text_boxes'] = result['text_boxes']
                
            if 'table_structure' in result:
                table_structure = result['table_structure']
                serializable_result['table_structure'] = {
                    'table_region': table_structure['table_region'],
                    'row_positions': table_structure['row_positions'],
                    'col_positions': table_structure['col_positions'],
                    'cells': table_structure['cells']
                }
                
            if 'entities' in result:
                serializable_result['entities'] = result['entities']
                
            json.dump(serializable_result, f, indent=2)
        
        # Visualize results
        if args.visualize:
            vis_path = os.path.join(args.output, f"{basename}_vis{ext}")
            predictor.visualize(img, result, vis_path)
            
    logger.info(f"All images processed. Results saved to {args.output}")


if __name__ == "__main__":
    main()
