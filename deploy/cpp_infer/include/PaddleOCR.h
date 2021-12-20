#pragma once

#include "Parameter.h"
#include "OCRResult.h"
extern "C" {
	__declspec(dllexport) int Detect(char* modelPath_det_infer, char* modelPath_cls_infer, char* modelPath_rec_infer, char* keys, char* imagefile, OCRParameter  parameter, LpOCRResult*  pOCRResult);

	__declspec(dllexport) int FreeDetectMem(LpOCRResult pOCRResult);
};

