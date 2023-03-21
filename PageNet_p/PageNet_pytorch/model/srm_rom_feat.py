import torch
import torch.nn as nn
from .block import build_CBLs, CBL

class SRMROMFeat(nn.Module):
    def __init__(self, in_channel, box_channels, dis_channels, cls_channels, rom_channels):
        super(SRMROMFeat, self).__init__()
        self.box_convs = build_CBLs(in_channel, box_channels, [3]*len(box_channels), [1]*len(box_channels), [1]*len(box_channels))
        self.dis_convs = build_CBLs(in_channel, dis_channels, [3]*len(dis_channels), [1]*len(dis_channels), [1]*len(dis_channels))
        self.cls_convs = build_CBLs(in_channel, cls_channels, [3]*len(cls_channels), [1]*len(cls_channels), [1]*len(cls_channels))

        self.box2dis_conv = CBL(box_channels[-1], dis_channels[-1], 1, 1, 0)
        self.cls2dis_conv = CBL(cls_channels[-1], dis_channels[-1], 1, 1, 0)

        rom_in_channel = dis_channels[-1] + cls_channels[-1]
        self.rom_convs = build_CBLs(rom_in_channel, rom_channels, [3]*len(rom_channels), [1]*len(rom_channels), [1]*len(rom_channels))

    def forward(self, input):
        box_feat = self.box_convs(input)
        dis_feat = self.dis_convs(input)
        cls_feat = self.cls_convs(input)

        box2dis_feat = self.box2dis_conv(box_feat)
        cls2dis_feat = self.cls2dis_conv(cls_feat)
        dis_feat = dis_feat + box2dis_feat + cls2dis_feat

        rom_feat = torch.cat((cls_feat, dis_feat), 1)
        rom_feat = self.rom_convs(rom_feat)

        return box_feat, dis_feat, cls_feat, rom_feat


def build_srm_rom_feat(cfg):
    srm_rom_feat = SRMROMFeat(
        in_channel=cfg['MODEL']['BACKBONE']['CHANNELS'][-1],  
        box_channels=cfg['MODEL']['FEAT']['BOX_CHANNELS'],  
        dis_channels=cfg['MODEL']['FEAT']['DIS_CHANNELS'],  
        cls_channels=cfg['MODEL']['FEAT']['CLS_CHANNELS'],  
        rom_channels=cfg['MODEL']['FEAT']['ROM_CHANNELS']   
    )
    return srm_rom_feat