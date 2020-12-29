import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PreProcessing import dense_to_one_hot, pickle_2_img_single
from facePatch import get_patch_1d


class MyDataset(Dataset):
    def __init__(self, split="Training", fold=0, transform=None):
        self.transform = transform
        self.split = split
        self.fold = fold
        self.data = "D:/chenchuyang/learning/sparse_coding/patch_mnn/pkl/ckp_2sizeimg.pkl"

        img, label = pickle_2_img_single(self.data)
        train_x = []
        train_y = []
        for i in range(10):
            if i == fold:
                test_x = img[i]
                test_y = label[i]
            else:
                train_x += img[i]
                train_y += label[i]
        if self.split == "Training":
            self.train_data = train_x
            self.train_labels = train_y
        elif self.split == "Testing":
            self.test_data = test_x
            self.test_labels = test_y

    def __getitem__(self, index):
        if self.split == "Training":
            img, target = self.train_data[index].reshape(-1,1536), self.train_labels[index]
        elif self.split == "Testing":
            img, target = self.test_data[index].reshape(-1,1536), self.test_labels[index]

        #for3channelsinput
        # img = img[:, :, np.newaxis]
        # img = np.concatenate((img, img, img), axis=2)


        #img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == "Training":
            return len(self.train_data)
        elif self.split == "Testing":
            return len(self.test_data)


def test():
    trainset = MyDataset(split='Training', fold=1, transform=None)
    testset = MyDataset(split='Testing', fold=1, transform=None)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=5, shuffle=False, num_workers=0)
    print(trainset[0][0].shape)
    print(trainset[0][1])
    # print(type(trainset[0][0]))
    # print(type(trainset[0][1]))
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    #    print(batch_idx, len(inputs), len(targets))
    # plt.imshow(trainset[0][0])
    # plt.show()

if __name__ == "__main__":
    test()
