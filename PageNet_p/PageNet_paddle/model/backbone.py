import paddle.nn as nn

from .block import conv1x1, BasicBlock

class Backbone(nn.Layer):
    def __init__(self, block, layers, in_channel, channels):
        super(Backbone, self).__init__()

        self.inplanes = in_channel
        self.n_blocks = len(layers)
        for i in range(self.n_blocks):
            layer = self._make_layer(block, channels[i], layers[i], stride=2)
            self.__setattr__(f'layer{i+1}', layer)

    def forward(self, input):
        output = input
        for i in range(self.n_blocks):
            output = self.__getattr__(f'layer{i+1}')(output)
        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes*block.expansion, stride),
                    nn.BatchNorm2D(planes*block.expansion, momentum=0.1, use_global_stats=False),
                )  
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

def build_backbone(cfg):
    if cfg['MODEL']['BACKBONE']['BLOCK'] == 'basicblock':
        block = BasicBlock
    else:
        raise ValueError
    
    backbone = Backbone(
        block=block,
        layers=cfg['MODEL']['BACKBONE']['LAYERS'],
        in_channel=cfg['MODEL']['BACKBONE']['IN_CHANNEL'],
        channels=cfg['MODEL']['BACKBONE']['CHANNELS']
    )
    return backbone