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

"""
This script implements inference functionality for MixTeX models
for formula recognition.
"""

import os
import sys
import cv2
import copy
import time
import math
import numpy as np
import argparse

import paddle
from paddle import inference

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

from ppocr.postprocess.mixtex_postprocess import MixTeXPostProcess
from ppocr.data.imaug.mixtex_process import ImageProcessor

# Add import for utility function
from predict_system import get_image_file_list


class TextRecognizer(object):
    def __init__(self, args):
        """Initialize text recognizer
        
        Args:
            args: Command line arguments
        """
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorith = "MixTeX"
        
        # Initialize model predictor
        self.predictor, self.input_tensor, self.output_tensor = self.create_predictor(
            args)
            
        # Initialize image processor
        self.img_processor = ImageProcessor(
            img_h=args.rec_image_shape[1],
            img_w=args.rec_image_shape[2]
        )
        
        # Initialize post-processor
        self.postprocessor = MixTeXPostProcess(args.rec_vocab_file)
        
        # Initialize time metrics
        self.preprocess_time = 0
        self.inference_time = 0
        self.postprocess_time = 0
        self.total_time = 0
        self.count = 0
        

    def create_predictor(self, args):
        """Create PaddlePaddle predictor
        
        Args:
            args: Command line arguments
            
        Returns:
            tuple: (predictor, input_tensor, output_tensor)
        """
        # Set config
        config = inference.Config(args.rec_model_file, args.rec_params_file)
        config.enable_memory_optim()
        
        # Use GPU if available
        if args.use_gpu:
            config.enable_use_gpu(args.gpu_mem, 0)
        else:
            config.disable_gpu()
            if args.enable_mkldnn:
                config.enable_mkldnn()
                config.set_cpu_math_library_num_threads(args.cpu_threads)
        
        # Create predictor
        predictor = inference.create_predictor(config)
        
        # Get input and output tensors
        input_tensor = predictor.get_input_handle(args.rec_input_name)
        output_tensors = {}
        for name in args.rec_output_name:
            output_tensors[name] = predictor.get_output_handle(name)
            
        return predictor, input_tensor, output_tensors
    
    def __call__(self, img_list):
        """Recognize text from images
        
        Args:
            img_list (list): List of images (numpy arrays)
            
        Returns:
            list: List of recognition results, each containing 'text' and 'score'
        """
        self.count += len(img_list)
        
        # Preprocess images
        img_list = copy.deepcopy(img_list)
        start_time = time.time()
        batch_img_input = []
        for img in img_list:
            processed_img = self.img_processor.resize_normalize(img)
            batch_img_input.append(processed_img)
        
        batch_img_tensor = np.concatenate(batch_img_input)
        self.preprocess_time += time.time() - start_time
        
        # Run inference
        start_time = time.time()
        self.input_tensor.copy_from_cpu(batch_img_tensor)
        self.predictor.run()
        outputs = {}
        for name, tensor in self.output_tensor.items():
            outputs[name] = tensor.copy_to_cpu()
        self.inference_time += time.time() - start_time
        
        # Format model outputs for post-processing
        preds = {'pred': outputs['pred']}
        
        # Post-process results
        start_time = time.time()
        process_result = self.postprocessor(preds)
        self.postprocess_time += time.time() - start_time
        
        # Format results
        results = []
        for result in process_result:
            text = result['pred']
            score = result['conf']
            results.append({
                'text': text,
                'score': score
            })
        
        return results
    
    def print_profiling(self):
        """Print profiling information"""
        if self.count == 0:
            return
            
        total_time = self.preprocess_time + self.inference_time + self.postprocess_time
        preprocess_time_avg = self.preprocess_time / self.count * 1000
        inference_time_avg = self.inference_time / self.count * 1000
        postprocess_time_avg = self.postprocess_time / self.count * 1000
        total_time_avg = total_time / self.count * 1000
        
        print("MixTeX Inference Profiling:")
        print(f"Total count: {self.count}")
        print(f"Preprocess time per image: {preprocess_time_avg:.2f} ms")
        print(f"Inference time per image: {inference_time_avg:.2f} ms")
        print(f"Postprocess time per image: {postprocess_time_avg:.2f} ms")
        print(f"Total time per image: {total_time_avg:.2f} ms")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='PaddleOCR MixTeX Inference')
    
    # Model settings
    parser.add_argument('--rec_model_file', type=str, required=True,
                        help='Path to the MixTeX model file (.pdmodel)')
    parser.add_argument('--rec_params_file', type=str, required=True,
                        help='Path to the MixTeX model parameters file (.pdiparams)')
    parser.add_argument('--rec_vocab_file', type=str, required=True,
                        help='Path to the MixTeX vocabulary file')
    
    # Input settings
    parser.add_argument('--image_file', type=str, required=True,
                        help='Path to image file or directory of images')
    parser.add_argument('--rec_batch_num', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--rec_image_shape', type=str, default='1,128,512',
                        help='Image shape in CHW format (e.g. 1,128,512)')
    
    # Runtime settings
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Whether to use GPU for inference')
    parser.add_argument('--gpu_mem', type=int, default=500,
                        help='GPU memory size in MB')
    parser.add_argument('--enable_mkldnn', action='store_true',
                        help='Enable MKL-DNN acceleration for CPU inference')
    parser.add_argument('--cpu_threads', type=int, default=10,
                        help='Number of CPU threads to use')
    parser.add_argument('--warmup', action='store_true',
                        help='Whether to warmup the model before inference')
    
    # Output settings
    parser.add_argument('--rec_input_name', type=str, default='x',
                        help='Input tensor name of the MixTeX model')
    parser.add_argument('--rec_output_name', type=list, default=['pred'],
                        help='Output tensor names of the MixTeX model')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='Path to save inference results')
    
    return parser.parse_args()


def main():
    """Main function for MixTeX inference"""
    # Parse arguments
    args = parse_args()
    
    # Parse image shape
    args.rec_image_shape = [int(v) for v in args.rec_image_shape.split(',')]
    
    # Get image list
    img_file_list = get_image_file_list(args.image_file)
    if len(img_file_list) == 0:
        print(f"No images found in {args.image_file}")
        return
    print(f"Found {len(img_file_list)} images for inference")
    
    # Create text recognizer
    recognizer = TextRecognizer(args)
    
    # Process images in batches
    batch_size = args.rec_batch_num
    total_img_num = len(img_file_list)
    results = []
    
    # Warmup if requested
    if args.warmup:
        for i in range(2):
            dummy_img = np.random.randint(0, 255, (128, 512, 1)).astype(np.uint8)
            recognizer([dummy_img])
        print("Model warmed up")
    
    print("Starting inference...")
    start_time = time.time()
    
    for beg_idx in range(0, total_img_num, batch_size):
        end_idx = min(beg_idx + batch_size, total_img_num)
        batch = []
        
        for idx in range(beg_idx, end_idx):
            img_path = img_file_list[idx]
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image: {img_path}")
                continue
            batch.append(img)
            
        if len(batch) > 0:
            batch_results = recognizer(batch)
            for i, result in enumerate(batch_results):
                img_path = img_file_list[beg_idx + i]
                results.append({
                    'image_path': img_path,
                    'text': result['text'],
                    'score': result['score']
                })
                print(f"Processed {len(results)}/{total_img_num}: {img_path} -> {result['text']}")
    
    # Print profiling information
    recognizer.print_profiling()
    
    # Save results to output file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    output_file = os.path.join(args.output_path, 'mixtex_results.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['image_path']}\t{result['text']}\t{result['score']}\n")
    
    print(f"Inference completed for {len(results)} images")
    print(f"Results saved to {output_file}")
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    main()
