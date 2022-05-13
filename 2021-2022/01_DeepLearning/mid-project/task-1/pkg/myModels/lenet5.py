import torch
import torch.nn as nn
from collections import OrderedDict


class LeNet5(nn.Module):
    def __init__(self, in_chalnnel=3, num_labels=10):
        super(LeNet5, self).__init__()
        
        self.seq = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_chalnnel, 6, kernel_size=5)),
            ("act1", nn.ReLU()),
            ("pool1", nn.AvgPool2d(kernel_size=2, stride=2)),
            ("conv2", nn.Conv2d(6, 16, kernel_size=5)),
            ("act2", nn.ReLU()),
            ("pool2", nn.AvgPool2d(kernel_size=2, stride=2)),
            ("conv3", nn.Conv2d(16, 120, kernel_size=5)),
            ("act3", nn.ReLU()),
            ("full_connect", nn.Conv2d(120, 84, kernel_size=1)),
            ("act4", nn.ReLU()),
        ]))

        self.mlp = nn.Linear(84, num_labels) # use mlp instead of RBF
        #self.act5 = nn.Softmax(dim=1)

    def forward(self, images):
        x = self.seq(images)
        x = torch.squeeze(x)
        x = self.mlp(x)
        #logits = self.act5(x)

        return x # [batch_size, num_labels]



