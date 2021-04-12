// test.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "Windows.h"
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
HINSTANCE OcrDetectorDll;

int main()
{
	const char* CS_DLLName;
	std::string strConfigPath, strImgPath;
	std::vector<std::vector<std::vector<int>>>vecBoxResult(10);
	//std::vector<char>vecCode(20);
	std::vector<char>vecCode(50);
	std::vector<std::vector<char>> vecResult(10, vecCode);
	strConfigPath = "D:\\4_code\\GitHub_Open\\PaddleOCR\\deploy\\cpp_infer\\tools\\config.txt";
	const char* strConfig = strConfigPath.c_str();
	strImgPath = "F:\\image\\OCR\\err\\1.bmp";
	//cv::Mat srcimg = cv::imread(strImgPath, cv::IMREAD_COLOR);
	cv::Mat srcimg = cv::imread(strImgPath, CV_8UC1);
   // cv::imwrite("e:\\opencv.bmp", srcimg);
	CS_DLLName = "ocr_system.dll";
	OcrDetectorDll = ::LoadLibrary(CS_DLLName);
	//初始化
	typedef int(*OCRDetectorInit)(const char* strConfig);
	OCRDetectorInit OCRDetectorInit_;
//	OCRDetectorInit_ = (OCRDetectorInit)GetProcAddress(OcrDetectorDll, MAKEINTRESOURCE(3));
	OCRDetectorInit_ = (OCRDetectorInit)GetProcAddress(OcrDetectorDll, "OCRInit");
	int i = OCRDetectorInit_(strConfig);
	//执行
	//typedef int(*OCRDetectorRun)(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>&boxes, std::vector<std::string>&labels, int &nboxNum);
	typedef int(*OCRDetectorRun)(unsigned char* pData, int nWidth, int nHeight, std::vector<std::vector<std::vector<int>>>&boxes, std::vector<std::vector<char>>&labels, int &nboxNum);
	OCRDetectorRun OCRDetectorRun_;
	OCRDetectorRun_ = (OCRDetectorRun)GetProcAddress(OcrDetectorDll, "OCRSystem");
	int k = 0;
	int nBoxnum;
	while (k<1)
	{
		auto start = std::chrono::system_clock::now();
		BYTE *pByte;
		pByte = new BYTE[srcimg.cols*srcimg.rows];
		memcpy(pByte, srcimg.data, srcimg.cols*srcimg.rows * sizeof(unsigned char));
		int j = OCRDetectorRun_(pByte,srcimg.cols, srcimg.rows, vecBoxResult, vecResult, nBoxnum);
		auto end = std::chrono::system_clock::now();
		auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "predict result:" << vecResult.size()<<"\t";
		/*	for (int i=0; i<vecResult.size(); i++)
			{
				std::cout << vecResult.at(i);
			}*/
		
		std::cout << "predict time:" << double(duration.count()) *std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << "s" << std::endl;
		k++;
	}
	//释放
	typedef int(*OCRDetectorFree)();
	OCRDetectorFree OCRDetectorFree_;
	OCRDetectorFree_ = (OCRDetectorFree)GetProcAddress(OcrDetectorDll, "OCRFree");
	OCRDetectorFree_();
	std::cout << "释放成功" << std::endl;
	system("pause");
    return 0;
}

