from .lenet5 import LeNet5
from .resnet import ResNet
from .mynet import NewResNet


def lenet5_builder(args, **kwags):
    return LeNet5()

def resnet_builder(args, **kwags):
    layers = kwags.get("layer_list", None)
    if layers is None:
        raise NotImplementedError
    return ResNet(layers=layers, act=args.activation)

def newres_builder(args, **kwags):
    layers = kwags.get("layer_list", None)
    if layers is None:
        raise NotImplementedError
    return NewResNet(layers=layers, act=args.activation)

switcher = {
    "lenet5": lenet5_builder,
    "resnet": resnet_builder,
    "resnet18": resnet_builder,
    "resnet34": resnet_builder,
    "newres": newres_builder,
}

def classifier_builder(args, layers=[]):
    return switcher[args.classifier](args, layer_list=layers)
