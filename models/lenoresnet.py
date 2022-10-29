"""
PyTorch implementation of LeNo-ResNet in the following paper:
    "Improved Residual Network Based on Norm-Preservation for Visual Recognition"
    Neural Networks (2022)
    (https://doi.org/10.1016/j.neunet.2022.10.023)
    Author: Bharat Mahaur <bharatmahaur@gmail.com>
"""

import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                     bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, downsampling=True,
                 downsampling_block=False, identity_blocks=False, bn0_exclude=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # self.conv2 can be used to downsample if avgpool is not applied on stride != 1

        if not downsampling_block and not bn0_exclude:
            self.bn0 = norm_layer(inplanes)
        # self.downsample layers downsample the input when stride != 1

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        if not downsampling:
            self.conv2 = conv3x3(planes, planes, stride=1) # only if using avgpool with stride != 1
        else:
            self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        if downsampling_block:
            self.bn3 = norm_layer(planes * self.expansion)

        if identity_blocks:
            self.bn3 = norm_layer(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.downsampling = downsampling
        self.downsampling_block = downsampling_block 
        self.identity_blocks = identity_blocks
        self.bn0_exclude = bn0_exclude

    def forward(self, x):
        identity = x

        if self.downsampling_block:
            out = self.conv1(x) # downsampling block
        elif self.bn0_exclude: 
            out = self.relu(x) # start identity block
            out = self.conv1(out)
        else:
            out = self.bn0(x) # middle identity block
            out = self.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsampling_block:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            if not self.downsampling:
                out = self.avgpool(out)
        
        out += identity

        if self.identity_blocks:
            out = self.bn3(out) # end identity block
            out = self.relu(out)

        return out

class LeNo_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, 
                 norm_layer=None, dropout_prob0=0.0):
        super(LeNo_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, 
                                 padding=1, bias=False)  
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, 
                                 bias=False) # input stem-B
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if dropout_prob0 > 0.0:
            self.dp = nn.Dropout(dropout_prob0, inplace=True)
            print("Using Dropout with the prob to set to 0 of: ", dropout_prob0)
        else:
            self.dp = None

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_pooling=True, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        
        if stride != 1 and self.inplanes != planes * block.expansion: # downsampling projection 
            if not norm_pooling:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                    nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                )
            else:
                downsample = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=stride, padding=1),
                    conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                )
        elif self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif stride != 1:
            downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 
                            norm_layer, downsampling_block=True))
        self.inplanes = planes * block.expansion
        bn0_exclude = True
        for _ in range(1, (blocks-1)):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer,
                                bn0_exclude=bn0_exclude))
            bn0_exclude = False

        layers.append(block(self.inplanes, planes, norm_layer=norm_layer, 
                            identity_blocks=True, bn0_exclude=bn0_exclude))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.dp is not None:
            x = self.dp(x)

        x = self.fc(x)

        return x


def LeNo_ResNet50(**kwargs):
    """
    Constructs a LeNo_ResNet-50 layer model.
    """
    model = LeNo_ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def LeNo_ResNet101(**kwargs):
    """
    Constructs a LeNo_ResNet-101 layer model.
    """
    model = LeNo_ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def LeNo_ResNet152(**kwargs):
    """
    Constructs a LeNo_ResNet-152 layer model.
    """
    model = LeNo_ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

def LeNo_ResNet200(**kwargs):
    """
    Constructs a LeNo_ResNet-200 layer model.
    """
    model = LeNo_ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model

def LeNo_ResNet401(**kwargs):
    """
    Constructs a LeNo_ResNet-401 layer model.
    """
    model = LeNo_ResNet(Bottleneck, [4, 45, 80, 4], **kwargs)
    return model

def LeNo_ResNet500(**kwargs):
    """
    Constructs a LeNo_ResNet-500 layer model.
    """
    model = LeNo_ResNet(Bottleneck, [4, 56, 102, 4], **kwargs)
    return model