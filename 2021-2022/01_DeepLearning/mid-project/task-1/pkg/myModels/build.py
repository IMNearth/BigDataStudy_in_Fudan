from .lenet5 import LeNet5
from .resnet import ResNet, BottleNeck, ResidualBlock
from .mynet import NewResNet

block_switcher = {
    "bottle_neck": BottleNeck,
    "residual": ResidualBlock,
}


def lenet5_builder(args, **kwargs):
    return LeNet5(num_labels=args.num_class)


def resnet_builder(args, **kwargs):
    layers = kwargs.get("layer_list", None)
    if layers is None:
        raise NotImplementedError
    assert args.block in block_switcher, "Choose block in [residual|bottle_neck] !"
    return ResNet(layers=layers, block=block_switcher[args.block], act=args.activation, num_labels=args.num_class)

def resnet18_builder(args, **kwargs):
    return ResNet(layers=[2,2,2,2], act=args.activation, num_labels=args.num_class)

def resnet34_builder(args, **kwargs):
    return ResNet(layers=[3,4,6,3], act=args.activation, num_labels=args.num_class)

def resnet50_builder(args, **kwargs):
    return ResNet(layers=[3,4,6,3], block=BottleNeck, act=args.activation, num_labels=args.num_class)

def resnet101_builder(args, **kwargs):
    return ResNet(layers=[3,4,23,3], block=BottleNeck, act=args.activation, num_labels=args.num_class)

def resnet152_builder(args, **kwargs):
    return ResNet(layers=[3,8,36,3], block=BottleNeck, act=args.activation, num_labels=args.num_class)


def newres_builder(args, **kwargs):
    layers = kwargs.get("layer_list", None)
    if layers is None:
        raise NotImplementedError
    return NewResNet(layers=layers, act=args.activation, num_labels=args.num_class)


switcher = {
    "lenet5": lenet5_builder,
    "resnet": resnet_builder,
    "resnet18": resnet18_builder,
    "resnet34": resnet34_builder,
    "resnet50": resnet50_builder,
    "resnet101": resnet101_builder,
    "resnet152": resnet152_builder,
    "newres": newres_builder,
}


def classifier_builder(args, layers=[]):
    model_class = switcher.get(args.classifier, resnet50_builder)
    return model_class(args, layer_list=layers)

