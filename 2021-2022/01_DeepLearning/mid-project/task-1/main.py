import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
from time import time

from pkg.myModels import classifier_builder
import pkg.engine as engine
from pkg.utils import lsuv_init, get_number_of_parameters


def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print("Setting seed done...")


parser = argparse.ArgumentParser(description='Mid-Part1')
# model
parser.add_argument('--classifier', default="resnet50", type=str, help="Classifier model to use.")
parser.add_argument('--num-class', default=100, type=int, help="Number if classes in dataset.")
parser.add_argument('--activation', default='relu', type=str, help="Activation function to use.")
parser.add_argument('--device', default=0, type=int, help="Which gpu to use.")
parser.add_argument('--use_init', default=False, type=bool, help="Choose to use LSUV init or not.")
# training
parser.add_argument('--batch-size', default=1024, type=int, help='Size of each mini batch.')
parser.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
parser.add_argument('--max-epoch', default=200, type=int, help="Number of loops over the whole trainset.")
parser.add_argument('--weight-decay', default=5e-4, type=float, help="Lambda of L2 regularization.")
parser.add_argument('--momentum', default=0.9, type=float, help='Optimizer momentum')
parser.add_argument('--scheduler', default=True, type=bool, help="Whether to use learning rate decay")
parser.add_argument('--milestones', default=[60, 120, 160], type=list, help='Scheduler milestones')
parser.add_argument('--gamma', default=0.2, type=float, help='Scheduler gamma')
parser.add_argument('--save-path', default="./checkpoint", type=str, help="Save trained model to...")
# data augmentation
parser.add_argument('--data-aug', default="base", type=str, help="Data Augmentation. cutmix|cutout|mixup, base=do nothing")
parser.add_argument('--beta', default=1.0, type=float, help='hyperparameter beta')
parser.add_argument('--cutmix-prob', default=0.5, type=float, help='cutmix probability')
parser.add_argument('--cutout-prob', default=0.5, type=float, help='cutout probability')
parser.add_argument('--mixup-prob', default=0.5, type=float, help='mixup probability')


def main():
    set_random_seed(2022)

    args = parser.parse_args()
    print(args)

    print("Now loading data...")
    train_transform = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0
        ),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0
        ),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='../datasets/cifar', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root='../datasets/cifar', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    print("\t Train samples: {}, Test samples: {}".format(len(trainset), len(testset)))
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print("Finished data loading...")

    model = classifier_builder(args, layers=[2,2,2,2])
    print("Model has [{}] parameters".format(get_number_of_parameters(model)))
    if args.use_init:
        model = lsuv_init(
            model, 
            trainloader, 
            needed_std=1.0, 
            std_tol=0.1, 
            max_attempts=10, 
            do_orthonorm=True, 
            device="cpu"
        )

    optimer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )# weight_decay == lambda in L2 penalty
    
    criterion = nn.CrossEntropyLoss().cuda()

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimer, milestones=args.milestones, gamma=args.gamma)
    else:
        scheduler = None

    start_time = time()
    model = engine.train(model, trainloader, criterion, optimer, testloader, args, scheduler=scheduler)
    end_time = time()
    print("Time elapsed {0:.2f}s".format(end_time - start_time))



if __name__ == "__main__":
    main()

