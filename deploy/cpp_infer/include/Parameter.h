// Copyright (c) 2021 raoyutian Authors. All Rights Reserved.
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

#pragma once
#pragma pack(push,1)
struct OCRParameter
{
	int numThread;
	int    Padding;
	int    MaxSideLen;
	float  BoxScoreThresh;
	float   BoxThresh;
	float   UnClipRatio;
	bool    DoAngle;
	bool   MostAngle;
	OCRParameter()
	{
		numThread = 2;
		Padding = 50;
		MaxSideLen = 2048;
		BoxScoreThresh = 0.618f;
		BoxThresh = 0.3f;
		UnClipRatio = 2.0f;
		DoAngle = true;
		MostAngle = true;
	}
};

struct modeldata
{
	char* det_infer;
	char* cls_infer;
	char* rec_infer; 
	char* keys;
	char* imagefile;
};
#pragma pack(pop) 
