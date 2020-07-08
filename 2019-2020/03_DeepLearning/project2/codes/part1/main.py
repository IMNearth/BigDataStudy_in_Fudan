import torch
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

set_random_seed(2008)

parser = argparse.ArgumentParser(description='Project2_16306110435')
parser.add_argument('-c', '--classifier', default="lenet5", type=str, help="Classifier model to use.")
parser.add_argument('-b', '--batch_size', default=512, type=int, help='Size of each mini batch.')
parser.add_argument('-l', '--lr', default=1e-3, type=float, help='Learning rate.')
parser.add_argument('-e', '--max_epoch', default=50, type=int, help="Number of loops over the whole trainset.")
parser.add_argument('-o', '--loss', default="cross_entropy", type=str, help="Loss function to use.")
parser.add_argument('-s', '--save_path', default="./checkpoint", type=str, help="Save trained model to...")
parser.add_argument('-r', '--reg', default=0, type=float, help="Lambda of L2 regularization.")
parser.add_argument('-i', '--use_init', default=False, type=bool, help="Choose to use LSUV init or not.")
parser.add_argument('-p', '--optim', default="Adam", type=str, help="Optimization algorithm.")
parser.add_argument('-d', '--scheduler', default=False, type=bool, help="Whether to use learning rate decay")
parser.add_argument('-a', '--activation', default='relu', type=str, help="Activation function to use.")
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
print("Model has [{}] parameters".format(get_number_of_parameters(model)))
if args.use_init:
    model = lsuv_init(model, trainloader, needed_std=1.0, std_tol=0.1, max_attempts=10, do_orthonorm=True, device="cpu")

optimer = eval("torch.optim."+args.optim)(model.parameters(), lr=args.lr, weight_decay=args.reg) # weight_decay == lambda in L2 penalty

if args.scheduler:
    scheduler = torch.optim.lr_scheduler.StepLR(optimer, step_size=25, gamma=0.5)
else:
    scheduler = None

start_time = time()
model = engine.train(model, trainloader, optimer, testloader, args, scheduler=scheduler)
end_time = time()
print("Time elapsed {0:.2f}s".format(end_time - start_time))
