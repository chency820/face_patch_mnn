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


use_cuda = torch.cuda.is_available()

class MyDataset(Dataset):
    def __init__(self, split="Training", fold=0, transform=None):
        self.transform = transform
        self.split = split
        self.fold = fold
        self.data = "./pkl/ckp_face_and_inner.pkl"

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
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == "Testing":
            img, target = self.test_data[index], self.test_labels[index]

        # for 3 channels input
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
    print('img shape', trainset[0][0][0].shape)
    print('patch shape', trainset[0][0][1].shape)
    print(trainset[0][1])
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs[1].type(torch.FloatTensor)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = Variable(inputs), Variable(targets)

if __name__ == "__main__":
    test()
