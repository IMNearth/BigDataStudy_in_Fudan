import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle
import argparse
import numpy as np
import random
from tqdm import tqdm
from colorama import Fore
from IPython import display
from sklearn.metrics import accuracy_score

from VGG_BatchNorm.models import VGG_A, VGG_A_BatchNorm
from VGG_BatchNorm.data.loaders import get_cifar_loader

"""
#################################
#    Functions                  #
#################################
"""
def get_accuracy(model, dataloader, device):
    """ Calculate the accuracy of model classification """
    model.eval()

    pred_labels = np.array([])
    true_labels = np.array([])

    for data_batch in dataloader:
        images, batch_labels = data_batch
        images = images.to(device)

        logits = F.softmax(model(images), dim=1)
        logits = logits.detach().cpu().numpy()
        batch_labels = batch_labels.detach().cpu().numpy()

        logits_labels = np.argmax(logits, axis=1)
        pred_labels = np.concatenate((pred_labels, logits_labels))
        true_labels = np.concatenate((true_labels, batch_labels))
    
    acc = accuracy_score(true_labels, pred_labels)

    return acc


def set_random_seeds(seed_value=0):
    """Set a random seed to ensure reproducible results"""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(model=None, optimizer=None, criterion=None, train_loader=None, val_loader=None, scheduler=None, epochs_n=100, best_model_path=None, device="cpu", args=None):
    """ Complete the entire training process. 

    In order to plot the loss landscape, record the loss value of each step. 
    """
    assert args is not None, "Please specify args!"
    model.to(device)
    
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    pbar = tqdm(range(epochs_n), unit='epoch')

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in pbar:
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            
            prediction = model(x)  # [batch_size, num_classes]
            loss = criterion(prediction, y)
            
            optimizer.zero_grad()
            loss.backward()
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # Add your code
            loss_list.append(loss.detach().cpu().item())
            grad.append(model.classifier[-1].weight.grad.detach().cpu().numpy())
            ## --------------------
            optimizer.step()

        losses_list.append(loss_list)
        grads.append(grad)
        train_accuracy_curve[epoch] = get_accuracy(model, train_loader, device=device)
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader, device=device)
        learning_curve[epoch] = np.sum(loss_list) / float(batches_n)

        if val_accuracy_curve[epoch] > max_val_accuracy:
            max_val_accuracy_epoch = epoch
            max_val_accuracy = val_accuracy_curve[epoch]

        pbar.set_description("Cur_Epoch[{0}] \033[0;32m Train -- loss:[{1:.4f}], acc:[{2:.5f}]\033[0m \033[0;33mTest -- acc:[{3:.5f}] \033[0m"
                            .format(epoch, np.sum(loss_list), train_accuracy_curve[epoch], val_accuracy_curve[epoch]))


    print("Best Accuracy: Epoch [{0}], Train -- loss:[{1:.4f}], acc:[{2:.5f}], Test -- acc:[{3:.5f}]\n".format(
            max_val_accuracy_epoch, learning_curve[max_val_accuracy_epoch],
            train_accuracy_curve[max_val_accuracy_epoch], max_val_accuracy
    ))
    if scheduler:
        scheduler.step()
    
    # Test your model and save figure here (not required)
    # remember to use model.eval()
    ## --------------------
    # Add code as needed
    if args.draw_training_curve:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].grid()
        axes[0].plot(learning_curve, '-', color="r", label="Training Curve")
        axes[0].legend(loc="best")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("average loss")

        axes[1].grid()
        axes[1].plot(train_accuracy_curve, '-', color="r", label="Train Accuracy Curve")
        axes[1].plot(val_accuracy_curve, '-', color="g", label="Test Accuracy Curve")
        axes[1].legend(loc="best")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("accuracy")
        
        fig.savefig("{}/reports/figures/{}_{}.png".format(args.save_path, args.classifier, args.learning_rate))
    
    if args.save_buffer:
        output_buffer = {
            "Grads": grads,
            "Training Loss per iter": losses_list,
            "Avg Training Loss per epoch": learning_curve,
            "Train Accuracy": train_accuracy_curve,
            "Test Accuracy": val_accuracy_curve,
        }
        output_path = "{}/log/{}_{}_{}.pkl".format(args.save_path, args.classifier, args.learning_rate, args.batch_size)
        with open(output_path, "wb") as f:
            pickle.dump(output_buffer, f)
        print("Saving intermediate results into [{}] done!".format(output_path))
        
    
    if args.save_model:
        model.eval()
        model_path = "{}/reports/models/{}_{}.pt".format(args.save_path, args.classifier, args.learning_rate)
        torch.save(model.state_dict(), model_path)
        print("Saving model into [{}] done!".format(model_path))
    ## --------------------

    return losses_list, grads


"""
#################################
#    Plot Functions             #
#################################
"""
def plot_fill_between(x, min_curve, max_curve, avg_curve, save_name="unknown", xlim=None, ylim=None, ylabel="", title=""):
    """ Plot the figure you want
    Args:
        min_curve, max_curve, avg_curve <dict>

    fill the area between the two curves can use plt.fill_between()
    """
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    axes.grid()
    axes.set_xlabel("iters")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.fill_between(x, min_curve['vgg_pure'], max_curve['vgg_pure'], alpha=0.3, color="g", label="vgg_pure")
    axes.fill_between(x, min_curve['vgg_bn'], max_curve['vgg_bn'], alpha=0.3, color="r", label="vgg_bn")

    if xlim is not None:
        axes.set_xlim(xlim)
    if ylim is not None:
        axes.set_ylim(ylim)

    axes.legend(loc="best")
    fig.savefig("./checkpoint/reports/figures/{}.png".format(save_name))


def plot_intermediate_results(learning_curve_list, train_accuracy_curve_list, val_accuracy_curve_list, labels, args):
    """ Plot intermediate result
    
    subplot: [1] Training loss curve (each line represents a different model setting)
             [2] Train accuracy curve 
             [3] Test accuracy curve
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 5))
        
    axes[0].grid()
    for i, learning_curve in enumerate(learning_curve_list):
        axes[0].plot(learning_curve, '-', label=labels[i])
    axes[0].legend(loc="best")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("average loss")
    axes[0].set_title("Training Curve")

    axes[1].grid()
    for i, train_accuracy_curve in enumerate(train_accuracy_curve_list):
        axes[1].plot(train_accuracy_curve, '-', label=labels[i])
    axes[1].legend(loc="best")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Train Accuracy Curve")

    axes[2].grid()
    for i, val_accuracy_curve in enumerate(val_accuracy_curve_list):
        axes[2].plot(val_accuracy_curve, '-', label=labels[i])
    axes[2].legend(loc="best")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("accuracy")
    axes[2].set_title("Test Accuracy Curve")

    fig.savefig("{}/reports/figures/intermediate_cmp.png".format(args.save_path))



"""
#################################
#    Args                       #
#################################
"""
parser = argparse.ArgumentParser(description='Project2_16306110435')
parser.add_argument('-c', '--classifier', default="vgg_pure", type=str, help="Classifier model to use.")
parser.add_argument('-b', '--batch_size', default=128, type=int, help='Size of each mini batch.')
parser.add_argument('-n', '--num_workers', default=4, type=int, help='Workers to load data.')
parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='Learning rate.')
parser.add_argument('-e', '--max_epoch', default=50, type=int, help="Number of loops over the whole trainset.")
parser.add_argument('-o', '--loss', default="cross_entropy", type=str, help="Loss function to use.")
parser.add_argument('-s', '--save_path', default="./checkpoint", type=str, help="Save trained model to...")
parser.add_argument('-r', '--reg', default=0, type=float, help="Lambda of L2 regularization.")
parser.add_argument('-p', '--optim', default="Adam", type=str, help="Optimization algorithm.")
parser.add_argument('-t', '--n_items', default=-1, type=int, help="Whether to use part of dataset.")
parser.add_argument('-v', '--seed_value', default=2020, type=int, help="Random seed value.")
parser.add_argument('-dr', '--draw_training_curve', default=False, type=bool, help="Draw Training Curve.")
parser.add_argument('-sb', '--save_buffer', default=True, type=bool, help="Save Intermediate results.")
parser.add_argument('-sm', '--save_model', default=True, type=bool, help="Save model")

loss_switcher = {
    "cross_entropy": nn.CrossEntropyLoss,
}
optim_switcher = {
    "Adam": torch.optim.Adam,
}
model_switcher = {
    "vgg_pure": VGG_A,
    "vgg_bn": VGG_A_BatchNorm,
}


"""
#################################
#    Initialize                 #
#################################
"""
args = parser.parse_args()
print("------------------------------------------------------------")
print(args)
print("------------------------------------------------------------")
set_random_seeds(seed_value=args.seed_value)
print("Setting seed=[{}] down!".format(args.seed_value), end=" | ")

# Make sure you are using the right device.
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
print("Training on {}".format(device), end=" | ")
print(torch.cuda.get_device_name(0))
print()

"""
#################################
#    Data Loading               #
#################################
"""
def load_data():
    # # add our package dir to path 
    # figures_path = os.path.join(args.save_path, 'reports', 'figures')
    # models_path = os.path.join(args.save_path, 'reports', 'models')

    # Initialize your data loader and 
    # make sure that dataloader works as expected 
    # by observing one sample from it.
    train_loader = get_cifar_loader(root="./data", 
                                    batch_size=args.batch_size, 
                                    train=True, 
                                    shuffle=True, 
                                    num_workers=args.num_workers, 
                                    n_items=args.n_items)
    val_loader = get_cifar_loader(root="./data", 
                                batch_size=args.batch_size, 
                                train=False, 
                                shuffle=False, 
                                num_workers=args.num_workers, 
                                n_items=-1)
    
    return train_loader, val_loader
# for X,y in train_loader:
#     print(X[0])
#     print(y[0])
#     print(X[0].shape)
#     img = np.transpose(X[0], [1,2,0])
#     plt.imshow(img*0.5 + 0.5)
#     plt.savefig('sample.png')
#     print(X[0].max())
#     print(X[0].min())
#     break

"""
#################################
#    Train your model           #
#################################
""" 
def train_model(train_loader, val_loader):
    # feel free to modify
    # loss_save_path = os.path.join(args.save_path, "loss")
    # grad_save_path = os.path.join(args.save_path, "grad")

    model = model_switcher[args.classifier]()
    optimizer = optim_switcher[args.optim](model.parameters(), lr=args.learning_rate)
    criterion = loss_switcher[args.loss]()

    loss, grads = train(model, optimizer, criterion, train_loader, val_loader, 
                        epochs_n=args.max_epoch, device=device, args=args)


"""
#################################
#    Draw Intermediate Results  #
#################################
"""
def draw_intermediate_results():
    learning_curve_list, train_accuracy_curve_list, val_accuracy_curve_list = [],[],[]
    for cname in ['vgg_pure', 'vgg_bn']:    
        loss_list = []
        for lr in ['0.001']:
            with open("./checkpoint/log/batch512/{}_{}.pkl".format(cname, lr), "rb") as f:
                data = pickle.load(f)
            
            learning_curve_list.append(data['Avg Training Loss per epoch'])
            train_accuracy_curve_list.append(data['Train Accuracy'])
            val_accuracy_curve_list.append(data['Test Accuracy'])

    plot_intermediate_results(learning_curve_list, train_accuracy_curve_list, val_accuracy_curve_list, ['vgg_pure', 'vgg_bn'], args)


"""
#################################
#    Draw Lanscape Pictures     #
#################################
"""
def draw_landscape_pictures():
    # Maintain two lists: max_curve and min_curve,
    # select the maximum value of loss in all models
    # on the same step, add it to max_curve, and
    # the minimum value to min_curve
    min_curve = {}
    max_curve = {}
    avg_curve = {}
    ## --------------------
    # Add your code
    for cname in ['vgg_pure', 'vgg_bn']:    
        loss_list = []
        for lr in ['0.001', '0.0001', '5e-05']: #'1e-06', '2e-05', '5e-05', '0.0001', '0.0002', '0.0005', '0.001', '0.002'
            with open("./checkpoint/log/{}_{}_{}.pkl".format(cname, lr, 128), "rb") as f:
                data = pickle.load(f)["Training Loss per iter"]
            data = np.array(data).reshape(1, -1).squeeze()
            loss_list.append(data)
        loss_array = np.array(loss_list)
        min_curve[cname] = np.min(loss_array, axis=0).tolist()
        max_curve[cname] = np.max(loss_array, axis=0).tolist()
        avg_curve[cname] = np.mean(loss_array, axis=0).tolist()
    
    x = np.linspace(0, len(min_curve['vgg_pure'])-1, num=1000).tolist()
    x = list(map(int, x))

    for cname in ['vgg_pure', 'vgg_bn']:
        a = [min_curve[cname][i] for i in x]
        b = [max_curve[cname][i] for i in x]
        c = [avg_curve[cname][i] for i in x]
        min_curve[cname] = a
        max_curve[cname] = b
        avg_curve[cname] = c

    plot_fill_between(x, min_curve, max_curve, avg_curve, 
                      ylim=[0, 2.2], xlim=[0, 8000], ylabel="loss", save_name="loss_landscape")
    ## --------------------


"""
#################################
#    Gradient Predictiveness    #
#################################
"""
def VGG_Grad_Pred(step_size=1):
    """ draw_gradient_pictures """
    min_curve = {}
    max_curve = {}
    avg_curve = {}

    for cname in ['vgg_pure', 'vgg_bn']:
        distance_list = []
        for lr in ['0.001', '0.0001', '1e-05']:
            with open("./checkpoint/log/{}_{}_{}.pkl".format(cname, lr, 128), "rb") as f:
                data = pickle.load(f)["Grads"]
        
            num_epochs = len(data)
            batch_size = len(data[0])
            flatten = []
            for i in range(num_epochs):
                flatten.extend(data[i])
            data = np.array(flatten)

            l2_norm, iters = [], []
            i = 0
            while i < data.shape[0]-step_size:
                grad_b = data[i]
                grad_n = data[i+step_size]
                dist = np.linalg.norm(grad_b-grad_n)
                
                l2_norm.append(dist)
                iters.append(i)
                i = i+step_size
            distance_list.append(l2_norm)
        
        distance_array = np.array(distance_list)
        min_curve[cname] = np.min(distance_array, axis=0)
        max_curve[cname] = np.max(distance_array, axis=0)
        avg_curve[cname] = np.mean(distance_array, axis=0)

    plot_fill_between(iters, min_curve, max_curve, avg_curve, 
                      ylim=[0,6], save_name="gradient", ylabel="l2 distance", title="Gradient Predictiveness")


"""
#################################
#   Effective Beta-Smoothness   #
#################################
"""
def VGG_Beta_Smooth(step_size=10):
    fig, axes = plt.subplots(1, 1, figsize=(8, 5), dpi=100)
    max_curve = {}
    for cname in ['vgg_pure', 'vgg_bn']:
        distance_list = []
        for lr in ['0.001', '0.0001', '1e-05']:
            with open("./checkpoint/log/{}_{}_{}.pkl".format(cname, lr, 128), "rb") as f:
                data = pickle.load(f)["Grads"]
        
            num_epochs = len(data)
            batch_size = len(data[0])
            flatten = []
            for i in range(num_epochs):
                flatten.extend(data[i])
            data = np.array(flatten)

            l2_norm, iters = [], []
            i = 0
            while i < data.shape[0]-step_size:
                grad_b = data[i]
                grad_n = data[i+step_size]
                dist = np.linalg.norm(grad_b - grad_n)
                
                l2_norm.append(dist)
                iters.append(i)
                i = i+step_size
            distance_list.append(l2_norm)
        
        distance_array = np.array(distance_list)
        max_curve[cname] = np.max(distance_array, axis=0)

    axes.grid()
    axes.set_xlabel("iters")
    axes.set_ylabel("l2 distance")
    axes.set_title("Effective Beta-Smoothness")
    axes.plot(iters, max_curve['vgg_pure'], '-', alpha=0.5, color="g", label="vgg_pure")
    axes.plot(iters, max_curve['vgg_bn'], '-', alpha=0.5, color="r", label="vgg_bn")
            
    axes.legend(loc="best")
    fig.savefig("./checkpoint/reports/figures/{}.png".format("beta_smooth"))


if __name__ == "__main__":
    # train_loader, val_loader = load_data()
    # train_model(train_loader, val_loader)
    # draw_intermediate_results()
    # draw_landscape_pictures()
    # VGG_Grad_Pred(step_size=100)
    # VGG_Beta_Smooth(step_size=100)
    pass