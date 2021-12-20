#include "include/PaddleOCR.h"
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

#include <glog/logging.h>
#include <include/ocr_det.h>
#include <include/ocr_cls.h>
#include <include/ocr_rec.h>
#include <include/utility.h>
#include <sys/stat.h>

#include <gflags/gflags.h>


DEFINE_bool(use_gpu, false, "Infering with GPU or CPU.");
DEFINE_int32(gpu_id, 0, "Device id of GPU to execute.");
DEFINE_int32(gpu_mem, 4000, "GPU id when infering with GPU.");
DEFINE_int32(cpu_threads, 10, "Num of threads with CPU.");
DEFINE_bool(enable_mkldnn, false, "Whether use mkldnn with CPU.");
DEFINE_bool(use_tensorrt, false, "Whether use tensorrt.");
DEFINE_string(precision, "fp32", "Precision be one of fp32/fp16/int8");
DEFINE_bool(benchmark, true, "Whether use benchmark.");
DEFINE_string(save_log_path, "./log_output/", "Save benchmark log path.");
// detection related
DEFINE_string(image_dir, "", "Dir of input image.");
DEFINE_string(det_model_dir, "", "Path of det inference model.");
DEFINE_int32(max_side_len, 2048, "max_side_len of input image.");
DEFINE_double(det_db_thresh, 0.3, "Threshold of det_db_thresh.");
DEFINE_double(det_db_box_thresh, 0.5, "Threshold of det_db_box_thresh.");
DEFINE_double(det_db_unclip_ratio, 1.6, "Threshold of det_db_unclip_ratio.");
DEFINE_bool(use_polygon_score, false, "Whether use polygon score.");
DEFINE_bool(visualize, true, "Whether show the detection results.");
// classification related
DEFINE_bool(use_angle_cls, false, "Whether use use_angle_cls.");
DEFINE_string(cls_model_dir, "", "Path of cls inference model.");
DEFINE_double(cls_thresh, 0.9, "Threshold of cls_thresh.");
// recognition related
DEFINE_string(rec_model_dir, "", "Path of rec inference model.");
DEFINE_int32(rec_batch_num, 1, "rec_batch_num.");
DEFINE_string(char_list_file, "../../ppocr/utils/ppocr_keys_v1.txt", "Path of dictionary.");

using namespace std;
using namespace cv;
using namespace PaddleOCR;
OCRParameter _parameter;

int main_system(modeldata md, LpOCRResult* OCRResult) {

    /*std::cout << " DBDetector det start" << endl;*/
    DBDetector det(md.det_infer, FLAGS_use_gpu, FLAGS_gpu_id,
        FLAGS_gpu_mem, _parameter.numThread,
        FLAGS_enable_mkldnn, _parameter.MaxSideLen, _parameter.BoxThresh,
        _parameter.BoxScoreThresh, _parameter.UnClipRatio,
        FLAGS_use_polygon_score, FLAGS_visualize,
        FLAGS_use_tensorrt, FLAGS_precision);

  /*  std::cout << " DBDetector det end" << endl;*/
    CRNNRecognizer rec(md.rec_infer, FLAGS_use_gpu, FLAGS_gpu_id,
        FLAGS_gpu_mem, FLAGS_cpu_threads,
        FLAGS_enable_mkldnn, md.keys,
        FLAGS_use_tensorrt, FLAGS_precision);

    auto start = std::chrono::system_clock::now();

    cv::Mat srcimg = cv::imread(md.imagefile, cv::IMREAD_COLOR);
    if (!srcimg.data) {
       /* std::cerr << "[ERROR] image read failed! image path: " << md.imagefile << endl;*/
        exit(1);
    }

    std::vector<std::vector<std::vector<int>>> boxes;//这里是每个文本区域的四个点坐标集合
    std::vector<Textblock> textblocks;//这里是每个文本区域文本

    std::vector<double> det_times;
    std::vector<double> rec_times;
    det.Run(srcimg, boxes, &det_times);
    cv::Mat crop_img;
    std::wstring text;
    for (int j = 0; j < boxes.size(); j++) {
        crop_img = Utility::GetRotateCropImage(srcimg, boxes[j]);
        std::wstring textblock = rec.Run(crop_img, &rec_times);
        textblocks.push_back(Textblock(textblock, boxes[j]));
    }
    if (textblocks.empty())
        return 0;

    LpOCRResult pOCRResult = new _OCRResult();
    *OCRResult = pOCRResult;
    pOCRResult->textCount = textblocks.size();
    pOCRResult->pOCRText = new _OCRText[pOCRResult->textCount];
    int idx = 0;
    for (vector<Textblock>::iterator it = textblocks.begin(); it != textblocks.end(); it++) {
        //文本长度
        pOCRResult->pOCRText[idx].textLen = it->textblock.length()*2 + 1;
        //文本
        pOCRResult->pOCRText[idx].ptext = new char[pOCRResult->pOCRText[idx].textLen + 1];
        memset(pOCRResult->pOCRText[idx].ptext, 0, pOCRResult->pOCRText[idx].textLen + 1);
        memcpy(pOCRResult->pOCRText[idx].ptext,  it->textblock.c_str(), pOCRResult->pOCRText[idx].textLen);
        
        //P1
        pOCRResult->pOCRText[idx].points[0].x = it->box[0][0];
        pOCRResult->pOCRText[idx].points[0].y = it->box[0][1];
        //P2
        pOCRResult->pOCRText[idx].points[1].x = it->box[1][0];
        pOCRResult->pOCRText[idx].points[1].y = it->box[1][1];
        //P3
        pOCRResult->pOCRText[idx].points[2].x = it->box[2][0];
        pOCRResult->pOCRText[idx].points[2].y = it->box[2][1];
        //P4
        pOCRResult->pOCRText[idx].points[3].x = it->box[3][0];
        pOCRResult->pOCRText[idx].points[3].y = it->box[3][1];
        idx++;
    }
    return textblocks.size();
}

int Detect(char* modelPath_det_infer, 
    char* modelPath_cls_infer,
    char* modelPath_rec_infer, 
    char* keys,
    char* imagefile, 
    OCRParameter  parameter,
    LpOCRResult* pOCRResult)
{
    modeldata md;
    md.cls_infer = modelPath_cls_infer;
    md.det_infer = modelPath_det_infer;
    md.rec_infer = modelPath_rec_infer;
    md.keys = keys;
    md.imagefile = imagefile;
    _parameter = parameter;
    /*std::cout << "main_system"<<endl;*/

    return  main_system(md, pOCRResult);
}

int FreeDetectMem(LpOCRResult pOCRResult) {
    if (pOCRResult == nullptr)
        return 0;

    if (pOCRResult->textCount == 0)
        return 0;

    //文本快数量
    for (int i = 0; i < pOCRResult->textCount; i++)
    {
        delete[] pOCRResult->pOCRText[i].ptext;
        pOCRResult->pOCRText[i].ptext = NULL;
    }
    delete[] pOCRResult->pOCRText;
    pOCRResult->pOCRText = NULL;

    //整个OCR文本数组内存
    delete pOCRResult;
    pOCRResult = nullptr;

    return 0;
}


////数据结构
//OCRResult
//-Text   识别的文本（所有TextBlocks内的文本拼接）
//-TextBlocks
//--Text   该分割区域下的识别文本
//--BoxPoints   该分割区域的四个点坐标，围成一个范围
//---Point   点   
//----X     X坐标
//----Y     Y坐标

