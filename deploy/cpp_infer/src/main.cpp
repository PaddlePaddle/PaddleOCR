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

#include "glog/logging.h"
#include "omp.h"
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

using namespace std;
using namespace cv;
using namespace PaddleOCR;

extern "C" _declspec(dllexport) int OCRInit(const char*strConfig);
extern "C" _declspec(dllexport) void GetDetectParam(double &det_db_thresh, double &det_db_box_thresh, double &det_db_unclip_ratio);
extern "C" _declspec(dllexport) void SetDetectParam(double det_db_thresh, double det_db_box_thresh, double det_db_unclip_ratio);
extern "C" _declspec(dllexport) int OCRDetector(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>&boxes, int &nboxNum);//仅提取字符位置
extern "C" _declspec(dllexport) int OCRRecognize(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>boxes, std::vector<std::vector<char>>&labels);//识别
extern "C" _declspec(dllexport) int OCRSystem(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>&boxes, std::vector<std::vector<char>>&labels, int &nboxNum);//提取+识别
extern "C" _declspec(dllexport) void OCRFree();
class OCRDetectorClass
{
public:
	OCRDetectorClass(){};
	OCRDetectorClass(std::string strConfig);
	int Init(std::string strConfig);
	void SetDetectParam(double det_db_thresh, double det_db_box_thresh, double det_db_unclip_ratio);
	void GetDetectParam(double &det_db_thresh, double &det_db_box_thresh, double &det_db_unclip_ratio);
	int Detector(cv::Mat imgMat, std::vector<std::vector<std::vector<int>>>&boxes);
	int Recognize(cv::Mat imgMat, std::vector<std::vector<std::vector<int>>>boxes, std::vector<std::string>&labels);
	void Free();
private:
	DBDetector *det = nullptr;
	Classifier *cls = nullptr;
	CRNNRecognizer *rec = nullptr;
};
OCRDetectorClass::OCRDetectorClass(std::string strConfig)
{
	OCRConfig config(strConfig);
	//config.PrintConfigInfo();
	det = new DBDetector(config.det_model_dir, config.use_gpu, config.gpu_id, 
		config.gpu_mem,config.cpu_math_library_num_threads, 
		config.use_mkldnn,config.max_side_len, config.det_db_thresh,
		config.det_db_box_thresh, config.det_db_unclip_ratio, 
		config.visualize, config.use_tensorrt, config.use_fp16);
	if (config.use_angle_cls == true) {
		cls = new Classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
			config.gpu_mem, config.cpu_math_library_num_threads,
			config.use_mkldnn, config.cls_thresh,
			config.use_tensorrt, config.use_fp16);
	}
	rec = new CRNNRecognizer(config.rec_model_dir, config.use_gpu, config.gpu_id,
		config.gpu_mem, config.cpu_math_library_num_threads,
		config.use_mkldnn, config.char_list_file,
		config.use_tensorrt, config.use_fp16);
}
int OCRDetectorClass::Init(std::string strConfig)
{
	try
	{
		OCRConfig config(strConfig);
		//config.PrintConfigInfo();
		if (det==nullptr)
		{
			det = new DBDetector(config.det_model_dir, config.use_gpu, config.gpu_id,
				config.gpu_mem, config.cpu_math_library_num_threads,
				config.use_mkldnn, config.max_side_len, config.det_db_thresh,
				config.det_db_box_thresh, config.det_db_unclip_ratio,
				config.visualize, config.use_tensorrt, config.use_fp16);
		}
		
		if (config.use_angle_cls == true && cls==nullptr) {
			cls = new Classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
				config.gpu_mem, config.cpu_math_library_num_threads,
				config.use_mkldnn, config.cls_thresh,
				config.use_tensorrt, config.use_fp16);
		}
		if (rec==nullptr)
		{
			rec = new CRNNRecognizer(config.rec_model_dir, config.use_gpu, config.gpu_id,
				config.gpu_mem, config.cpu_math_library_num_threads,
				config.use_mkldnn, config.char_list_file,
				config.use_tensorrt, config.use_fp16);
		}
	}
	catch (const std::exception&)
	{
		return 1;
	}
	return 0;
}
void OCRDetectorClass::SetDetectParam(double det_db_thresh, double det_db_box_thresh, double det_db_unclip_ratio)
{
	/*det->det_db_thresh_ = det_db_thresh;
	det->det_db_box_thresh_ = det_db_box_thresh;
	det->det_db_unclip_ratio_ = det_db_unclip_ratio;*/
}
void OCRDetectorClass::GetDetectParam(double &det_db_thresh, double &det_db_box_thresh, double &det_db_unclip_ratio)
{
	//det_db_thresh = det->det_db_thresh_;
	//det_db_box_thresh = det->det_db_box_thresh_;
	//det_db_unclip_ratio = det->det_db_unclip_ratio_;
}
int OCRDetectorClass::Detector(cv::Mat imgMat, std::vector<std::vector<std::vector<int>>>&boxes)
{
	try
	{
		boxes.clear();
		det->Run(imgMat, boxes);
	}
	catch (const std::exception&)
	{
		return 1;
	}
	return 0;
}
int OCRDetectorClass::Recognize(cv::Mat imgMat, std::vector<std::vector<std::vector<int>>>boxes, std::vector<std::string> &labels)
{
	try
	{
		rec->Run(boxes, imgMat, cls);
		labels = rec->vecStrAllResult;
	}
	catch (const std::exception&)
	{
		return 1;
	}
	return 0;
}
void OCRDetectorClass::Free()
{
	if (det)
	{
		delete det;
		det = nullptr;
	}
	if (cls)
	{
		delete cls;
		cls = nullptr;
	}
	if (rec)
	{
		delete rec;
		rec = nullptr;
	}
}
OCRDetectorClass* myOcrDetector = new OCRDetectorClass;
extern "C" _declspec(dllexport) int OCRInit(const char* strConfig)
{
	std::string str = strConfig;
	myOcrDetector->Init(str);
	return 0;
}
extern "C" _declspec(dllexport) void GetDetectParam(double &det_db_thresh, double &det_db_box_thresh, double &det_db_unclip_ratio)
{
	myOcrDetector->GetDetectParam(det_db_thresh, det_db_box_thresh, det_db_unclip_ratio);
}
extern "C" _declspec(dllexport) void SetDetectParam(double det_db_thresh, double det_db_box_thresh, double det_db_unclip_ratio)
{
	myOcrDetector->SetDetectParam(det_db_thresh, det_db_box_thresh, det_db_unclip_ratio);
}
extern "C" _declspec(dllexport) int OCRDetector(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>&boxes, int &nboxNum)
{
	//cv::Mat imgMat;
	//imgMat = Mat::zeros(nHeight, nWidth, CV_8UC1);
	//memcpy(imgMat.data, pData, nWidth*nHeight);
	try
	{
		std::vector<std::vector<std::vector<int>>>boxesTemp;
		int lineByte;
		lineByte = (nWidth + 3) / 4 * 4;
		cv::Mat imgMat(nHeight, nWidth, CV_8UC1, pData, lineByte);
		cv::Mat imgColor;
		cvtColor(imgMat, imgColor, COLOR_GRAY2BGR);
		nboxNum = 0;
		myOcrDetector->Detector(imgColor, boxesTemp);
		if (boxes.size() < boxesTemp.size())
		{
			nboxNum = boxes.size();
		}
		else
		{
			nboxNum = boxesTemp.size();
		}
		for (int i = 0; i < nboxNum; i++)
		{
			boxes.at(i) = boxesTemp.at(i);
		}
	}
	catch (const std::exception&)
	{
		return 1;
	}
	return 0;
}
extern "C" _declspec(dllexport) int OCRRecognize(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>boxes, std::vector<std::vector<char>>&labels)
{
	try
	{
		std::vector<std::string> labelsTemp;
		int lineByte;
		lineByte = (nWidth + 3) / 4 * 4;
		cv::Mat imgMat(nHeight, nWidth, CV_8UC1, pData, lineByte);
		cv::Mat imgColor;
		//cv::imwrite("e:\\1015.bmp", imgMat);
		cvtColor(imgMat, imgColor, COLOR_GRAY2BGR);
		myOcrDetector->Recognize(imgColor, boxes, labelsTemp);
		for (int i = 0; i < boxes.size(); i++)
		{
			copy(labelsTemp.at(i).begin(), labelsTemp.at(i).end(), labels.at(i).begin());
		}
	}
	catch (const std::exception&)
	{
		return 1;
	}
	return 0;
}
extern "C" _declspec(dllexport) int OCRSystem(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>&boxes, 
					std::vector<std::vector<char>>&labels, int &nboxNum)
{
	try
	{
		std::vector<std::string> labelsTemp;
		std::vector<std::vector<std::vector<int>>>boxesTemp;
		int lineByte;
		nboxNum = 0;
		lineByte = (nWidth + 3) / 4 * 4;
		cv::Mat imgMat(nHeight, nWidth, CV_8UC1, pData, lineByte);
		cv::Mat imgColor;
		cvtColor(imgMat, imgColor, COLOR_GRAY2BGR);
		myOcrDetector->Detector(imgColor, boxesTemp);
		if (boxesTemp.size() > 0)
		{
			myOcrDetector->Recognize(imgColor, boxesTemp, labelsTemp);
			if (labels.size() < labelsTemp.size() || boxes.size() < boxesTemp.size())
			{
				nboxNum = boxes.size();
			}
			else
			{
				nboxNum = boxesTemp.size();
			}
			for (int i = 0; i < nboxNum; i++)
			{
				boxes.at(i) = boxesTemp.at(i);
				copy(labelsTemp.at(i).begin(), labelsTemp.at(i).end(), labels.at(i).begin());
			}
		}
	}
	catch (const std::exception&)
	{
		return 1;
	}
	return 0;
}
_declspec(dllexport) void OCRFree()
{
	myOcrDetector->Free();
	delete myOcrDetector;
	myOcrDetector = nullptr;
}
//int main(int argc, char **argv) {
//  if (argc < 3) {
//    std::cerr << "[ERROR] usage: " << argv[0]
//              << " configure_filepath image_path\n";
//    exit(1);
//  }
//
//  OCRConfig config(argv[1]);
//
//  config.PrintConfigInfo();
//
//  std::string img_path(argv[2]);
//
//  cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
//
//  DBDetector det(config.det_model_dir, config.use_gpu, config.gpu_id,
//                 config.gpu_mem, config.cpu_math_library_num_threads,
//                 config.use_mkldnn, config.max_side_len, config.det_db_thresh,
//                 config.det_db_box_thresh, config.det_db_unclip_ratio,
//                 config.visualize, config.use_tensorrt, config.use_fp16);
//
//  Classifier *cls = nullptr;
//  if (config.use_angle_cls == true) {
//    cls = new Classifier(config.cls_model_dir, config.use_gpu, config.gpu_id,
//                         config.gpu_mem, config.cpu_math_library_num_threads,
//                         config.use_mkldnn, config.cls_thresh,
//                         config.use_tensorrt, config.use_fp16);
//  }
//
//  CRNNRecognizer rec(config.rec_model_dir, config.use_gpu, config.gpu_id,
//                     config.gpu_mem, config.cpu_math_library_num_threads,
//                     config.use_mkldnn, config.char_list_file,
//                     config.use_tensorrt, config.use_fp16);
//
//  auto start = std::chrono::system_clock::now();
//  std::vector<std::vector<std::vector<int>>> boxes;
//  det.Run(srcimg, boxes);
//
//  rec.Run(boxes, srcimg, cls);
//  auto end = std::chrono::system_clock::now();
//  auto duration =
//      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//  std::cout << "Cost  "
//            << double(duration.count()) *
//                   std::chrono::microseconds::period::num /
//                   std::chrono::microseconds::period::den
//            << "s" << std::endl;
//
//  return 0;
//}
