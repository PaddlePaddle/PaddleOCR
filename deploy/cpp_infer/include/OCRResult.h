#pragma once
#include <vector>
using namespace std;

#pragma pack(push,1)
struct Textblock {
	std::wstring textblock;
	std::vector<std::vector<int>> box;
	Textblock(wstring textblock, std::vector<std::vector<int>> box) {
		this->textblock = textblock;
		this->box = box;
	}
};

//textblock文本四个角的点
struct _OCRTextPoint {
	int x;
	int y;
	_OCRTextPoint() :x(0), y(0) {
	}
};

struct _OCRText {
	//textblock文本
	int textLen;
	char* ptext;
	//一个textblock四个点
	_OCRTextPoint points[4];
	_OCRText() {
		textLen = 0;
		ptext = nullptr;
	}
};

typedef struct _OCRResult {
	//textblock文本个数
	int textCount;
	_OCRText* pOCRText;
}OCRResult, * LpOCRResult;

#pragma pack(pop) 