from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_eps=1e-5, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


def res_layer(
    block=BasicBlock,
    inplanes=256,
    planes=512,
    outplanes=512,
    blocks=2,
    stride=1,
    bn_eps=1e-5,
    bn_momentum=0.1,
):
    # the default values are for layer4
    assert outplanes == planes * block.expansion

    downsample = None
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            nn.Conv2d(
                inplanes,
                outplanes,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(outplanes, eps=bn_eps, momentum=bn_momentum),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            downsample,
            bn_eps=bn_eps,
            bn_momentum=bn_momentum,
        )
    )

    for _i in range(1, blocks):
        layers.append(block(outplanes, planes, bn_eps=bn_eps, bn_momentum=bn_momentum))

    return nn.Sequential(*layers)
