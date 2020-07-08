import torch
import random
import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt


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
    print("Setting seed={} done...".format(seed))


def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()

    return parameters_n


def plot_intermediate_results(train_batch_loss_list, train_epoch_loss_list, test_accuracy_list, labels, args, plus=""):
    """ Plot intermediate result
    
    subplot: [1] Training batch loss curve (each line represents a different model setting)
             [2] Training epoch loss curve 
             [3] Test accuracy curve
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    axes[0].grid()
    for i, learning_curve in enumerate(train_batch_loss_list):
        axes[0].plot(learning_curve, '-', label=labels[i])
    axes[0].legend(loc="best")
    axes[0].set_xlabel("Iters", fontsize=14)
    axes[0].set_ylabel("batch loss", fontsize=14)
    axes[0].set_title("Learning Curve", fontsize=16)

    axes[1].grid()
    for i, train_accuracy_curve in enumerate(train_epoch_loss_list):
        axes[1].plot(train_accuracy_curve, '-', label=labels[i])
    axes[1].legend(loc="best")
    axes[1].set_xlabel("Epochs", fontsize=14)
    axes[1].set_ylabel("epoch loss", fontsize=14)

    axes[2].grid()
    for i, val_accuracy_curve in enumerate(test_accuracy_list):
        axes[2].plot(val_accuracy_curve, '-', label=labels[i])
    axes[2].legend(loc="best")
    axes[2].set_xlabel("Epochs", fontsize=14)
    axes[2].set_ylabel("accuracy", fontsize=14)
    axes[2].set_title("Test Accuracy Curve", fontsize=16)

    fig.savefig("{}/report/intermediate_{}.png".format(args.save_path, plus))


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

# RGB颜色转换为HSL颜色
def rgb2hsl(rgb):
    rgb_normal = [[[rgb[0] / 255, rgb[1] / 255, rgb[2] / 255]]]
    hls = cv.cvtColor(np.array(rgb_normal, dtype=np.float32), cv.COLOR_RGB2HLS)
    return hls[0][0][0], hls[0][0][2], hls[0][0][1]  # hls to hsl
 
 
# HSL颜色转换为RGB颜色
def hsl2rgb(hsl):
    hls = [[[hsl[0], hsl[2], hsl[1]]]]  # hsl to hls
    rgb_normal = cv.cvtColor(np.array(hls, dtype=np.float32), cv.COLOR_HLS2RGB)
    return list(map(float, [rgb_normal[0][0][0], rgb_normal[0][0][1], rgb_normal[0][0][2]]))
 
 
# HSL渐变色
def get_multi_colors_by_hsl(begin_color, end_color, color_count):
    if color_count < 2:
        return []
 
    colors = []
    hsl1 = rgb2hsl(begin_color)
    hsl2 = rgb2hsl(end_color)
    steps = [(hsl2[i] - hsl1[i]) / (color_count - 1) for i in range(3)]
    for color_index in range(color_count):
        hsl = [hsl1[i] + steps[i] * color_index for i in range(3)]
        colors.append(hsl2rgb(hsl))
 
    return colors