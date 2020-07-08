import torch
import torch.nn.functional as F

import os
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class VisualHelper:
    features = {}
    def __init__(self, model, layer_names=[]):
        Names = [layer.split('.') for layer in layer_names]
        
        for layername in Names:
            out = self.hook_layer(model, layername)
            if out == -1:
                print("\033[0;33;40mNot Found Layer {}\033[0m".format(layername))
    
    def hook_fn(self, module, input, output, key=""):
        if key == "":
            self.features[module] = output
        else:
            self.features[key] = output
    
    def hook_layer(self, module, lyname, path=""):
        found = 0
        for name, layer in module._modules.items():
            if name == lyname[0]:
                found = 1
                full_path = ".".join([path, name]) if path != "" else name
                if len(lyname) > 1:
                    self.hook_layer(layer, lyname[1:], path=full_path)
                else:
                    layer.register_forward_hook(
                        lambda m,i,o: self.hook_fn(m, i, o, key=full_path)
                    )
                break
        
        if found == 0:
            return -1
        else:
            return 0

    
def visualization(model, dataloader, args, visual_list=[], num=10, k=1):
    if len(visual_list) == 0:
        print("Please specify which layer to visualize! Format like -- visual_list=['layer1.1.conv1']")
        return
    
    visual_woker = VisualHelper(model, layer_names=visual_list)

    # get the first batch out to visualize
    for data_batch in dataloader:
        images, labels = data_batch
        break
    
    # no more than #num samples
    if images.shape[0] > num:
        images, labels = images[:num], labels[:num]
    
    if torch.cuda.is_available():
        model = model.cuda()
        images = images.cuda()
    
    out = model(images) # forward

    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(images[k].detach().cpu().numpy().transpose(1,2,0))
    fig.savefig("./figures/raw_figure_{}.png".format(k))

    for i, (key, val) in enumerate(visual_woker.features.items()):
        fig = plt.figure(figsize=(50, 50), dpi=100)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.8, hspace=0.1, wspace=0.1)
        channel = val.shape[1]
        for c in range(channel):
            if c == 16: break
            img = val[k][c].detach().cpu().numpy()
            ax = fig.add_subplot(4, 4, c+1, xticks=[], yticks=[])
            ax.imshow(img, cmap="gray")
        
        fig.savefig("./figures/{}_figure{}.png".format(key, i))
