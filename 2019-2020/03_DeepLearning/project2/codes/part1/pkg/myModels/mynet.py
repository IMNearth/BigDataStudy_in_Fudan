import torch
import torch.nn as nn
from collections import OrderedDict


class ResidualBlock(nn.Module):

    expansion = 1

    act_switcher={
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'elu': nn.ELU,
    }

    def __init__(self, in_channel, out_channel, stride=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.seq = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)),
            ("bn1", nn.BatchNorm2d(out_channel)),
            ("act1", self.act_switcher[act](inplace=True) if act != 'tanh' else self.act_switcher[act]() ),
            ("conv2", nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)),
            ("bn2", nn.BatchNorm2d(out_channel)),
        ]))

        self.downsample = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )
        self.relu = self.act_switcher[act](inplace=True) if act != 'tanh' else self.act_switcher[act]()

    
    def forward(self, x):
        identity = self.downsample(x)
        
        out = self.seq(x)
        out += identity
        out = self.relu(out)

        return out


class NewResNet(nn.Module):
    act_switcher={
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'elu': nn.ELU,
    }
    def __init__(self, layers, in_channel=3, num_labels=10, block=ResidualBlock, zero_init_residual=False, act='relu'):
        super(NewResNet, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False), # out_size[32,32]
            nn.BatchNorm2d(64),
            self.act_switcher[act](),
        )

        self.inplanes=64
        self.layer1 = self.make_layer(block, 64,  layers[0], stride=1, act=act) # out_size[32,32]
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, act=act) # out_size[16,16]
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, act=act) # out_size[8,8]
        # self.layer4 = self.make_layer(block, 512, layers[3], stride=2) # out_size[4,4]

        self.seq2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), # out_size=[4,4]
            nn.Conv2d(256, 512, kernel_size=2, stride=1, bias=False), # out_size=[3,3]
            nn.BatchNorm2d(512),
            self.act_switcher[act](),
            nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False), # out_size=[2,2]
            nn.BatchNorm2d(512),
            self.act_switcher[act](),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, num_labels)
        #self.fc2 = nn.Linear(256, num_labels)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResidualBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def make_layer(self, block, out_channel, num_blocks, stride, act):
        layers = []
        layers.append(block(self.inplanes, out_channel, stride=stride, act=act))
        self.inplanes = out_channel * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, out_channel, act=act))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.seq1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.seq2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        # x = self.fc2(x)

        return x



