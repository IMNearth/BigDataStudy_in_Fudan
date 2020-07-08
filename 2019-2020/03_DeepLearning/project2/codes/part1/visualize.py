import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
from time import time

from myModels import classifier_builder
import engine
from utils import lsuv_init

def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Setting seed done...")

set_random_seed(2020)

parser = argparse.ArgumentParser(description='Project2_16306110435')
parser.add_argument('-c', '--classifier', default="newres", type=str, help="Classifier model to use.")
parser.add_argument('-b', '--batch_size', default=512, type=int, help='Size of each mini batch.')
parser.add_argument('-s', '--save_path', default="./checkpoint", type=str, help="Save trained model to...")
args = parser.parse_args()
print(args)

print("Now loading data...")
train_transform = transforms.Compose([
    transforms.Pad(padding=4),
    transforms.RandomCrop(size=32),
    transforms.transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=train_transform) # 50000
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=test_transform) # 10000
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
print("Finished data loading...")

model = classifier_builder(args, layers=[2,2,2,2])

pre_trainPATH = "./checkpoint/newres/05-09T06-42.pt"
model.load_state_dict(torch.load(pre_trainPATH))

engine.visualization(model, testloader, args, visual_list=["seq2.3", "seq2.6"])
# "seq1.2", "layer1.0.seq", "layer2.0.seq", "layer3.0.seq"
