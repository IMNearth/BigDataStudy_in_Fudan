import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
import numpy as np
from time import time

from lib import utils, dataset, trainer, model


parser = argparse.ArgumentParser(description="Project3_16307110435")
parser.add_argument('-cls',     '--classifier',     default="pointnet", type=str)
parser.add_argument('-b',       '--batch_size',     default=16, type=int,                   help='Size of each mini batch.')
parser.add_argument('-l',       '--lr',             default=1e-3, type=float,               help='Learning rate.')
parser.add_argument('-ep',      '--max_epoch',      default=20, type=int,                  help="Number of loops over the whole trainset.")
parser.add_argument('-o',       '--loss',           default="cross_entropy", type=str,      help="Loss function to use.")
parser.add_argument('-save',    '--save_path',      default="./checkpoint", type=str,       help="Save trained model to...")
parser.add_argument('-r',       '--reg',            default=0, type=float,                  help="Lambda of L2 regularization.")
parser.add_argument('-opt',     '--optim',          default="Adam", type=str,               help="Optimization algorithm.")
parser.add_argument('-sd',      '--scheduler',      default=False, action='store_true',     help="Whether to use learning rate decay")
parser.add_argument('-act',     '--activation',     default='relu', type=str,               help="Activation function to use.")
parser.add_argument('-xyz',     '--use_xyz',        default=True, type=bool,                help="Use xyz as features or not.")
parser.add_argument('-len',     '--length',         default=-1, type=int,                   help="Number of partial samples for training and testing.")
parser.add_argument('-da',      '--data_arg',       default=False, action='store_true',     help="Whether to use data argumentation")


if __name__ == "__main__":
    utils.set_random_seed(2020)
    
    args = parser.parse_args()
    print(args)
    args.length = None if args.length == -1 else args.length

    trainset = dataset.ModelNetDataset(mode="train", length=args.length)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    testset = dataset.ModelNetDataset(mode="test", length=args.length)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    print("Finished data loading... Train Sample Size: {}, Test Sample Size: {}".format(len(trainset), len(testset)))
    
    if args.classifier == "pointnet":
        model = model.PointNetClassificationMSG(args)
    else:
        model = model.cls_3d()
    print("Model {} has [{}] parameters".format(args.classifier, utils.get_number_of_parameters(model)))

    optimer = eval("torch.optim."+args.optim)(model.parameters(), lr=args.lr, weight_decay=args.reg) # weight_decay == lambda in L2 penalty

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimer, step_size=50, gamma=0.5)
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    
    start_time = time()
    model = trainer.train(model, trainloader, optimer, criterion, testloader, args, scheduler=scheduler)
    end_time = time()
    print("Time elapsed {0:.2f}min".format((end_time - start_time)/60))
