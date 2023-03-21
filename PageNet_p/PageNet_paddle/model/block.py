import paddle.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
	return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)

def conv1x1(in_planes, out_planes, stride=1):
	return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)  #Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode=‘zeros’)

def build_CBLs(inplanes, planes, kernel_sizes, strides, paddings):
	layers = []
	for i in range(len(planes)):
		if i == 0:
			inplanes = inplanes
		else:
			inplanes = planes[i-1]
		outplanes = planes[i]
		stride = strides[i]
		padding = paddings[i]
		kernel_size = kernel_sizes[i]
		layers.append(CBL(inplanes, outplanes, kernel_size, stride, padding))
	return nn.Sequential(*layers)

class CBL(nn.Layer):
	def __init__(self, inplanes, outplanes, kernel_size, stride, padding):
		super(CBL, self).__init__()
		self.conv = nn.Conv2D(inplanes, outplanes, kernel_size, stride, padding, bias_attr=False)
		self.bn = nn.BatchNorm2D(outplanes, momentum=0.1, use_global_stats=False)  
		self.leakyrelu = nn.LeakyReLU(0.1)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.leakyrelu(x)
		return x

class BasicBlock(nn.Layer):
	expansion = 1

	def __init__(self, in_planes, out_planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, out_planes, stride)
		self.bn1 = nn.BatchNorm2D(out_planes,momentum=0.1, use_global_stats=False)
		self.relu = nn.LeakyReLU(0.1)  
		self.conv2 = conv3x3(out_planes, out_planes)
		self.bn2 = nn.BatchNorm2D(out_planes,momentum=0.1, use_global_stats=False)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class Bottleneck(nn.Layer):
	expansion = 4

	def __init__(self, in_planes, out_planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = conv1x1(in_planes, out_planes)
		self.bn1 = nn.BatchNorm2D(out_planes, momentum=0.1, use_global_stats=False)
		self.conv2 = conv3x3(out_planes, out_planes, stride)
		self.bn2 = nn.BatchNorm2D(out_planes, momentum=0.1, use_global_stats=False)
		self.conv3 = conv1x1(out_planes, out_planes*self.expansion)
		self.bn3 = nn.BatchNorm2D(out_planes*self.expansion, momentum=0.1, use_global_stats=False)
		self.relu = nn.LeakyReLU(0.1)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out