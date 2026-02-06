// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mkldnn_blocklist.h"

namespace Mkldnn {

const std::unordered_set<std::string> MKLDNN_BLOCKLIST = {
    "LaTeX_OCR_rec",
    "PP-FormulaNet-L",
    "PP-FormulaNet-S",
    "UniMERNet",
    "UVDoc",
    "Cascade-MaskRCNN-ResNet50-FPN",
    "Cascade-MaskRCNN-ResNet50-vd-SSLDv2-FPN",
    "Mask-RT-DETR-M",
    "Mask-RT-DETR-S",
    "MaskRCNN-ResNeXt101-vd-FPN",
    "MaskRCNN-ResNet101-FPN",
    "MaskRCNN-ResNet101-vd-FPN",
    "MaskRCNN-ResNet50-FPN",
    "MaskRCNN-ResNet50-vd-FPN",
    "MaskRCNN-ResNet50",
    "SOLOv2",
    "PP-TinyPose_128x96",
    "PP-TinyPose_256x192",
    "Cascade-FasterRCNN-ResNet50-FPN",
    "Cascade-FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "Co-DINO-Swin-L",
    "Co-Deformable-DETR-Swin-T",
    "FasterRCNN-ResNeXt101-vd-FPN",
    "FasterRCNN-ResNet101-FPN",
    "FasterRCNN-ResNet101",
    "FasterRCNN-ResNet34-FPN",
    "FasterRCNN-ResNet50-FPN",
    "FasterRCNN-ResNet50-vd-FPN",
    "FasterRCNN-ResNet50-vd-SSLDv2-FPN",
    "FasterRCNN-ResNet50",
    "FasterRCNN-Swin-Tiny-FPN",
    "MaskFormer_small",
    "MaskFormer_tiny",
    "SLANeXt_wired",
    "SLANeXt_wireless",
    "SLANet",
    "SLANet_plus",
    "YOWO",
    "SAM-H_box",
    "SAM-H_point",
    "PP-FormulaNet_plus-L",
    "PP-FormulaNet_plus-M",
    "PP-FormulaNet_plus-S"};
}
