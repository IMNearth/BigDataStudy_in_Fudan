import torch
import torch.nn as nn
from collections import OrderedDict

"""
https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
"""

class ResidualBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()

        self.seq = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)),
            ("bn1", nn.BatchNorm2d(out_channel)),
            ("act1", nn.ReLU(inplace=True)),
            ("conv2", nn.Conv2d(out_channel, out_channel*self.expansion, kernel_size=3, stride=1, padding=1, bias=False)),
            ("bn2", nn.BatchNorm2d(out_channel*self.expansion)),
        ]))

        self.downsample = nn.Sequential()
        if stride != 1 or in_channel != out_channel*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        return self.relu(self.seq(x) + self.downsample(x))


class BottleNeck(nn.Module):
    """ Residual block for resnet over 50 layers """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleNeck, self).__init__()
        self.residual_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * self.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * self.expansion))
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.residual_branch(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, 
        layers, 
        in_channel=3, 
        num_labels=10, 
        block=ResidualBlock, 
        zero_init_residual=False, 
        act="relu",
        use_bottle_neck=False
    ):
        super(ResNet, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False), # out_size[32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.in_channels = 64
        self.layer1 = self.make_layer(block, 64,  layers[0], stride=1) # out_size[32,32]
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2) # out_size[16,16]
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2) # out_size[8,8]
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2) # out_size[4,4]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_labels)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.seq1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x




        
        