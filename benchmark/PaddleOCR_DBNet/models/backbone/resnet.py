import math
import paddle
from paddle import nn

BatchNorm2d = nn.BatchNorm2D

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "deformable_resnet18",
    "deformable_resnet50",
    "resnet152",
]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def constant_init(module, constant, bias=0):
    module.weight = paddle.create_parameter(
        shape=module.weight.shape,
        dtype="float32",
        default_initializer=paddle.nn.initializer.Constant(constant),
    )
    if hasattr(module, "bias"):
        module.bias = paddle.create_parameter(
            shape=module.bias.shape,
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(bias),
        )


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias_attr=False
    )


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(BasicBlock, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU()
        self.with_modulated_dcn = False
        if not self.with_dcn:
            self.conv2 = nn.Conv2D(
                planes, planes, kernel_size=3, padding=1, bias_attr=False
            )
        else:
            from paddle.version.ops import DeformConv2D

            deformable_groups = dcn.get("deformable_groups", 1)
            offset_channels = 18
            self.conv2_offset = nn.Conv2D(
                planes, deformable_groups * offset_channels, kernel_size=3, padding=1
            )
            self.conv2 = DeformConv2D(
                planes, planes, kernel_size=3, padding=1, bias_attr=False
            )
        self.bn2 = BatchNorm2d(planes, momentum=0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        super(Bottleneck, self).__init__()
        self.with_dcn = dcn is not None
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = BatchNorm2d(planes, momentum=0.1)
        self.with_modulated_dcn = False
        if not self.with_dcn:
            self.conv2 = nn.Conv2D(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False
            )
        else:
            deformable_groups = dcn.get("deformable_groups", 1)
            from paddle.vision.ops import DeformConv2D

            offset_channels = 18
            self.conv2_offset = nn.Conv2D(
                planes,
                deformable_groups * offset_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
            )
            self.conv2 = DeformConv2D(
                planes, planes, kernel_size=3, padding=1, stride=stride, bias_attr=False
            )
        self.bn2 = BatchNorm2d(planes, momentum=0.1)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = BatchNorm2d(planes * 4, momentum=0.1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dcn = dcn
        self.with_dcn = dcn is not None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # out = self.conv2(out)
        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Layer):
    def __init__(self, block, layers, in_channels=3, dcn=None):
        self.dcn = dcn
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.out_channels = []
        self.conv1 = nn.Conv2D(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias_attr=False
        )
        self.bn1 = BatchNorm2d(64, momentum=0.1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn)

        if self.dcn is not None:
            for m in self.modules():
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    if hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False,
                ),
                BatchNorm2d(planes * block.expansion, momentum=0.1),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dcn=dcn))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))
        self.out_channels.append(planes * block.expansion)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x2, x3, x4, x5


def load_torch_params(paddle_model, torch_patams):
    paddle_params = paddle_model.state_dict()

    fc_names = ["classifier"]
    for key, torch_value in torch_patams.items():
        if "num_batches_tracked" in key:
            continue
        key = (
            key.replace("running_var", "_variance")
            .replace("running_mean", "_mean")
            .replace("module.", "")
        )
        torch_value = torch_value.detach().cpu().numpy()
        if key in paddle_params:
            flag = [i in key for i in fc_names]
            if any(flag) and "weight" in key:  # ignore bias
                new_shape = [1, 0] + list(range(2, torch_value.ndim))
                print(
                    f"name: {key}, ori shape: {torch_value.shape}, new shape: {torch_value.transpose(new_shape).shape}"
                )
                torch_value = torch_value.transpose(new_shape)
            paddle_params[key] = torch_value
        else:
            print(f"{key} not in paddle")
    paddle_model.set_state_dict(paddle_params)


def load_models(model, model_name):
    import torch.utils.model_zoo as model_zoo

    torch_patams = model_zoo.load_url(model_urls[model_name])
    load_torch_params(model, torch_patams)


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        assert (
            kwargs.get("in_channels", 3) == 3
        ), "in_channels must be 3 whem pretrained is True"
        print("load from imagenet")
        load_models(model, "resnet18")
    return model


def deformable_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], dcn=dict(deformable_groups=1), **kwargs)
    if pretrained:
        assert (
            kwargs.get("in_channels", 3) == 3
        ), "in_channels must be 3 whem pretrained is True"
        print("load from imagenet")
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict=False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert (
            kwargs.get("in_channels", 3) == 3
        ), "in_channels must be 3 whem pretrained is True"
        model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]), strict=False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        assert (
            kwargs.get("in_channels", 3) == 3
        ), "in_channels must be 3 whem pretrained is True"
        load_models(model, "resnet50")
    return model


def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], dcn=dict(deformable_groups=1), **kwargs)
    if pretrained:
        assert (
            kwargs.get("in_channels", 3) == 3
        ), "in_channels must be 3 whem pretrained is True"
        model.load_state_dict(model_zoo.load_url(model_urls["resnet50"]), strict=False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        assert (
            kwargs.get("in_channels", 3) == 3
        ), "in_channels must be 3 whem pretrained is True"
        model.load_state_dict(model_zoo.load_url(model_urls["resnet101"]), strict=False)
    return model


def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        assert (
            kwargs.get("in_channels", 3) == 3
        ), "in_channels must be 3 whem pretrained is True"
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]), strict=False)
    return model


if __name__ == "__main__":
    x = paddle.zeros([2, 3, 640, 640])
    net = resnet50(pretrained=True)
    y = net(x)
    for u in y:
        print(u.shape)

    print(net.out_channels)
