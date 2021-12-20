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

//textblock�ı��ĸ��ǵĵ�
struct _OCRTextPoint {
	int x;
	int y;
	_OCRTextPoint() :x(0), y(0) {
	}
};

struct _OCRText {
	//textblock�ı�
	int textLen;
	char* ptext;
	//һ��textblock�ĸ���
	_OCRTextPoint points[4];
	_OCRText() {
		textLen = 0;
		ptext = nullptr;
	}
};

typedef struct _OCRResult {
	//textblock�ı�����
	int textCount;
	_OCRText* pOCRText;
}OCRResult, * LpOCRResult;

#pragma pack(pop) 