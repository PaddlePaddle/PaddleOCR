import paddle
import paddle.nn as nn

from .backbone import build_backbone
from .predictor import build_predictor
from .srm_rom_feat import build_srm_rom_feat

class PageNet(nn.Layer):
    def __init__(self, backbone, srm_rom_feat, predictor):
        super(PageNet, self).__init__()

        self.backbone = backbone
        self.srm_rom_feat = srm_rom_feat
        self.predictor = predictor

    def forward(self, input):
        feat = self.backbone(input)
        box_feat, dis_feat, cls_feat, rom_feat = self.srm_rom_feat(feat)
        output = self.predictor(box_feat, dis_feat, cls_feat, rom_feat)
        return output

def build_model_paddle(cfg):
    backbone = build_backbone(cfg)
    srm_rom_feat = build_srm_rom_feat(cfg)
    predictor = build_predictor(cfg)
    
    pagenet = PageNet(
        backbone=backbone,
        srm_rom_feat=srm_rom_feat,
        predictor=predictor
    )

    if cfg['MODEL']['WEIGHTS'] != '':
        pagenet.set_dict(paddle.load(cfg['MODEL']['WEIGHTS']))  

    return pagenet