import os
import h5py
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


class ModelNetDataset(Dataset):

    structure = {
        "train": [
            "data/modelnet40_ply_hdf5_2048/ply_data_train0.h5",
            "data/modelnet40_ply_hdf5_2048/ply_data_train1.h5",
            "data/modelnet40_ply_hdf5_2048/ply_data_train2.h5",
            "data/modelnet40_ply_hdf5_2048/ply_data_train3.h5",
            "data/modelnet40_ply_hdf5_2048/ply_data_train4.h5",
        ],
        "test": [
            "data/modelnet40_ply_hdf5_2048/ply_data_test0.h5",
            "data/modelnet40_ply_hdf5_2048/ply_data_test1.h5",
        ]
    }

    def __init__(self, mode="train", length=None, root="/root/16307110435_zjw/deeplearn/project3"):
        super().__init__()

        self.root = root
        self.data_list = self.structure["train"] if mode=="train" else self.structure["test"]
        # self.cat = {}
        self.pts = []
        self.labels = []

        # We have 5 files for training and 2 files for testing
        for file_name in self.data_list:
            data = h5py.File(os.path.join(self.root, file_name), 'r')
            self.pts.append(data['data'])
            self.labels.append(data['label'])
        # Combine model data from all files
        self.pts = np.vstack(self.pts)
        self.labels = np.vstack(self.labels)

        # Make sure you load the correct data
        if mode == "train":
            assert self.pts.shape == (9840, 2048, 3), "Train file loading error!"
        else:
            assert self.pts.shape == (2468, 2048, 3), "Test file loading error!"
        
        # added for partial training
        if length is not None:
            self.pts = self.pts[:length, :, :]
            self.labels = self.labels[:length]

    def __len__(self):
        return self.pts.shape[0]

    def __getitem__(self, index):

        pts = self.pts[index]  # (2048, 3)
        label = self.labels[index]

        # # Put the channel dimension in front for feeding into the network
        # pts = pts.transpose(1,0)

        return {
            "points": pts,
            "label": label
        }
