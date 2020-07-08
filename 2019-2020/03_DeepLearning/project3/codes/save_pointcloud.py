import os
import torch
import numpy as np

from lib.dataset import ModelNetDataset
from lib.utils import get_multi_colors_by_hsl


if __name__ == "__main__":
    trainset = ModelNetDataset(mode="train")
    train_out = os.path.join("./data/pointCloud", "train")

    idx = 2021
    data_dict = trainset[idx]

    save_mode = "color"             # select in ["normal", "color"]
    colors = get_multi_colors_by_hsl(begin_color=[255,0,0], end_color=[0,0,255], color_count=8)

    pt = data_dict["points"] # (3, 2048)

    with open(train_out + "%05d.obj" % idx, "w") as f:
        for j in range(pt.shape[0]):
            x, y, z = pt[j, :]
            x ,y, z = list(map(float, [x, y, z]))
            if save_mode == "normal":
                f.write(
                    "v {:.6f} {:.6f} {:.6f} 1.0\n".format(x, y, z)                  # x, y, x, w
                )
            elif save_mode == "color":
                r, g, b = colors[int(x/0.125)]
                f.write(
                    "v {:.6f} {:.6f} {:.6f} {} {} {}\n".format(x, y, z, r, g, b)    # x, y, z, r, g, b
                )
            else:
                raise NotImplementedError

