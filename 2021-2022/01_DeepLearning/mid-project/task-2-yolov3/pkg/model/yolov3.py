import torch
import torch.nn as nn
import numpy as np

from pkg.model.backbones.darknet53 import Darknet53
from pkg.model.necks.yolo_fpn import FPN_YOLOV3
from pkg.model.head.yolo_head import Yolo_head
from pkg.model.layers.conv_module import Convolutional

from pkg.config import cfg
from pkg.utils.tools import *

import logging
logger = logging.getLogger("YoloV3.model")


class YoloV3(nn.Module):
    """
    Note : int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, init_weights=True):
        super(YoloV3, self).__init__()

        self.anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.nC = cfg.DATA["NUM"]
        self.out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.nC + 5)

        self.backnone = Darknet53()
        self.fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
                                fileters_out=[self.out_channel, self.out_channel, self.out_channel])

        # small
        self.head_small = Yolo_head(nC=self.nC, anchors=self.anchors[0], stride=self.strides[0])
        # medium
        self.head_medium = Yolo_head(nC=self.nC, anchors=self.anchors[1], stride=self.strides[1])
        # large
        self.head_large = Yolo_head(nC=self.nC, anchors=self.anchors[2], stride=self.strides[2])

        if init_weights:
            self._init_weights()


    def forward(self, x):
        out = []

        x_s, x_m, x_l = self.backnone(x)
        x_s, x_m, x_l = self.fpn(x_l, x_m, x_s)

        out.append(self.head_small(x_s))
        out.append(self.head_medium(x_m))
        out.append(self.head_large(x_l))

        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)


    def _init_weights(self):
        " Note: nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                # logger.info("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)

                # logger.info("initing {}".format(m))


    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"

        logger.info(f"Load darknet weights : {weight_file}")

        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1

                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b

                    # logger.info("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

                # logger.info("loading weight {}".format(conv_layer))
        logger.info(f"Loading weight ... Done!")


if __name__ == '__main__':
    net = Yolov3()
    logger.info(net)

    in_img = torch.randn(12, 3, 416, 416)
    p, p_d = net(in_img)

    for i in range(3):
        logger.info(p[i].shape)
        logger.info(p_d[i].shape)