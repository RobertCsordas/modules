import torch
import torch.nn.functional as F
import math
from layers import Conv2d, Linear, BatchNorm2d

# Adapted from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/resnet.py


def conv3x3(in_planes, out_planes, stride=1):
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.layers = torch.nn.Sequential(
            conv3x3(inplanes, planes, stride),
            BatchNorm2d(planes),
            torch.nn.ReLU(inplace=True),

            conv3x3(planes, planes),
            BatchNorm2d(planes),
        )
        self.downsample = downsample if downsample is not None else lambda x: x

    def forward(self, x):
        return F.relu(self.downsample(x) + self.layers(x), inplace=True)


class ResNet110(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.inplanes = 16
        self.features = torch.nn.Sequential(
            Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, 16, 18),
            self._make_layer(BasicBlock, 32, 18, stride=2),
            self._make_layer(BasicBlock, 64, 18, stride=2)
        )

        self.out_layer = Linear(64 * BasicBlock.expansion, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.weight.shape[-1] * m.weight.shape[-2] * m.weight.shape[0]
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.features(x).mean(dim=(2, 3)))
