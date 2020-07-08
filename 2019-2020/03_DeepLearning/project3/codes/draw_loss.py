import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


def plot_intermediate_results(train_epoch_loss_list, test_accuracy_list, labels, save_path="./checkpoint/report", plus=""):
    """ Plot intermediate result
    
    subplot: [1] Training epoch loss curve (each line represents a different model setting)
             [2] Test accuracy curve
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    axes[0].grid()
    for i, train_accuracy_curve in enumerate(train_epoch_loss_list):
        axes[0].plot(train_accuracy_curve, '-', label=labels[i])
    axes[0].legend(loc="best", fontsize=12)
    axes[0].set_xlabel("Epochs", fontsize=14)
    axes[0].set_ylabel("epoch loss", fontsize=14)
    axes[0].set_title("Learning Curve", fontsize=16)

    axes[1].grid()
    for i, val_accuracy_curve in enumerate(test_accuracy_list):
        axes[1].plot(val_accuracy_curve, '-', label=labels[i])
    axes[1].legend(loc="best", fontsize=12)
    axes[1].set_xlabel("Epochs", fontsize=14)
    axes[1].set_ylabel("accuracy", fontsize=14)
    axes[1].set_title("Test Accuracy Curve", fontsize=16)

    fig.savefig("{}/intermediate_{}.png".format(save_path, plus))


if __name__ == "__main__":
    pointnet_out = pd.read_csv("./out1.csv", header=0)
    baseline_out = pd.read_csv("./out2.csv", header=0)

    pointnet_out["TrainLoss"] = pointnet_out["TrainLoss"] / 615.0
    baseline_out["TrainLoss"] = baseline_out["TrainLoss"] / 615.0

    plot_intermediate_results(
        train_epoch_loss_list = [pointnet_out["TrainLoss"], baseline_out["TrainLoss"]],
        test_accuracy_list = [pointnet_out["TestAcc"], baseline_out["TestAcc"]],
        labels=["pointnet++", "baseline"]
    )
    
