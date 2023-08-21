// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using System;
using System.IO;
using System.Runtime.InteropServices;
using OpenCvSharp;
using fastdeploy;

namespace Test
{
    public class TestPPOCRv3
    {
        public static void Main(string[] args)
        {
            if (args.Length < 6) {
                Console.WriteLine(
                    "Usage: infer_demo path/to/det_model path/to/cls_model " +
                    "path/to/rec_model path/to/rec_label_file path/to/image " +
                    "run_option, " +
                    "e.g ./infer_demo ./ch_PP-OCRv2_det_infer " +
                    "./ch_ppocr_mobile_v2.0_cls_infer ./ch_PP-OCRv2_rec_infer " +
                    "./ppocr_keys_v1.txt ./12.jpg 0"
                );
                Console.WriteLine( "The data type of run_option is int, 0: run with cpu; 1: run with gpu");
                return;
            }
            string det_model_dir = args[0];
            string cls_model_dir = args[1];
            string rec_model_dir = args[2];
            string rec_label_file = args[3];
            string image_path = args[4];
            RuntimeOption runtimeoption = new RuntimeOption();
            int device_option = Int32.Parse(args[5]);
            if(device_option==0){
                runtimeoption.UseCpu();
            }else{
                runtimeoption.UseGpu();
            }
            string sep = "\\";
            string det_model_file = det_model_dir + sep + "inference.pdmodel";
            string det_params_file = det_model_dir + sep + "inference.pdiparams";

            string cls_model_file = cls_model_dir + sep + "inference.pdmodel";
            string cls_params_file = cls_model_dir + sep + "inference.pdiparams";

            string rec_model_file = rec_model_dir + sep + "inference.pdmodel";
            string rec_params_file = rec_model_dir + sep + "inference.pdiparams";

            fastdeploy.vision.ocr.DBDetector dbdetector = new fastdeploy.vision.ocr.DBDetector(det_model_file, det_params_file, runtimeoption, ModelFormat.PADDLE);
            fastdeploy.vision.ocr.Classifier classifier = new fastdeploy.vision.ocr.Classifier(cls_model_file, cls_params_file, runtimeoption, ModelFormat.PADDLE);
            fastdeploy.vision.ocr.Recognizer recognizer = new fastdeploy.vision.ocr.Recognizer(rec_model_file, rec_params_file, rec_label_file, runtimeoption, ModelFormat.PADDLE);
            fastdeploy.pipeline.PPOCRv3 model = new fastdeploy.pipeline.PPOCRv3(dbdetector, classifier, recognizer);
            if(!model.Initialized()){
                Console.WriteLine("Failed to initialize.\n");
            }
            Mat image = Cv2.ImRead(image_path);
            fastdeploy.vision.OCRResult res = model.Predict(image);
            Console.WriteLine(res.ToString());
            Mat res_img = fastdeploy.vision.Visualize.VisOcr(image, res);
            Cv2.ImShow("result.png", res_img);
            Cv2.ImWrite("result.png", res_img);
            Cv2.WaitKey(0);

        }

    }
}