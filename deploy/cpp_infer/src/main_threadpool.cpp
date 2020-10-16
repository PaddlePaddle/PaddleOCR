// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/config.h>
#include <include/ocr_det.h>
#include <include/ocr_rec.h>
#include <include/thread_pool.h> 

using namespace std;
using namespace cv;
using namespace PaddleOCR;

std::vector<std::thread::id> vecID;
std::vector<DBDetector> vecDBD;
std::vector<CRNNRecognizer> vecCRNNRec;
std::vector<Classifier*> vecCLS;
//获取线程ID
void getIDString()
{
  std::thread::id tid = std::this_thread::get_id();
  vecID.push_back(tid);
  std::cout<<"Thread ID: "<<tid<<std::endl;
}
//OCR处理流程
void OCR_processing(cv::Mat srcmat)
{
  std::thread::id tid = std::this_thread::get_id();
  int i=0;
  for(;i<vecID.size();i++)
    if(tid == vecID[i])
      break;
  if(i==vecID.size())
    std::cout<<"ERROR! no thread ID."<<std::endl;
  std::vector<std::vector<std::vector<int>>> boxes;
  vecDBD[i].Run(srcmat, boxes);
  vecCRNNRec[i].Run(boxes, srcmat,vecCLS[i]);

}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "[ERROR] usage: " << argv[0]
              << " configure_filepath image_path\n";
    exit(1);
  }

  Config config(argv[1]);

  config.PrintConfigInfo();
  //init thread pool
  unsigned short threadpool_num = config.threadpool_num;
  std::threadpool executor{threadpool_num};

  std::string image_floder(argv[2]);

  std::vector<cv::String> filenames;
  glob(image_floder,filenames);
  int num = filenames.size();

  //cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);

  for(int i=0;i<threadpool_num;i++)
  { 
    DBDetector det(
                config.det_model_dir, config.use_gpu, config.gpu_id, config.gpu_mem,
                config.cpu_math_library_num_threads, config.use_mkldnn,
                config.use_zero_copy_run, config.max_side_len, config.det_db_thresh,
                config.det_db_box_thresh, config.det_db_unclip_ratio, config.visualize);
    vecDBD.push_back(det);
    Classifier *cls = nullptr;
    if (config.use_angle_cls == true) {
    cls = new Classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
                         config.gpu_mem, config.cpu_math_library_num_threads,
                         config.use_mkldnn, config.use_zero_copy_run,
                         config.cls_thresh);
    };
    vecCLS.push_back(cls);
    CRNNRecognizer rec(config.rec_model_dir, config.use_gpu, config.gpu_id,
                     config.gpu_mem, config.cpu_math_library_num_threads,
                     config.use_mkldnn, config.use_zero_copy_run,
                     config.char_list_file);
    vecCRNNRec.push_back(rec);
  }
  for(int i=0;i<threadpool_num;i++)
  {
    std::future<void> ff = executor.commit(getIDString);
    ff.get();
  }
  //初始化时间
  double start = static_cast<double>( cv::getTickCount());

    for(int i=0;i<num;++i)
    {
        cv::Mat srcmat;
        srcmat = cv::imread(filenames[i]);
        //if(!srcmat.data){
        //    std::cout<<"img empty"<<std::endl;
        //    continue;
        //  }
        //std::cout<<filenames[i]<<std::endl;

    while(executor.tasksSize()>= 2*threadpool_num)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

    //std::future<std::void> ff = 
    executor.commit(OCR_processing,srcmat);
    }

      while(executor.tasksSize()!=0)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

 double timeres = (static_cast<double>( cv::getTickCount()) - start) / cv::getTickFrequency();
    std::cout << "total time:" << timeres << " s" << std::endl;
    std::cout<<"speed:"<< num/timeres<<" img/s"<<std::endl;
    std::cout<<"run end"<<endl;
  return 0;
}
