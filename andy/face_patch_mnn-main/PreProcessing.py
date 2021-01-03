import os
import pickle
import cv2
import numpy as np
import scipy.fftpack as FFT
from facePatch import get_patch_1d, show_img
import matplotlib.pyplot as plt

eye_indexes = list(range(36, 48))
mouth_indexes = list(range(48, 60))
all_indexes = eye_indexes + mouth_indexes

np.set_printoptions(suppress=True)


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = []
    for i in range(num_classes):
        if i == labels_dense:
            labels_one_hot.append(1)
        else:
            labels_one_hot.append(0)
    return np.array(labels_one_hot)


def pickle_2_img_single(data_file):
    '''load data from pkl'''

    if not os.path.exists(data_file):
        print('file {0} not exists'.format(data_file))
        exit()
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    total_x1, total_y = [], []
    for i in range(len(data)):
        x1 = []
        #x2 = []
        yl = []
        print(len(data[i]['img']))
        for j in range(len(data[i]['labels'])):
            img = data[i]['img'][j]
            img_b = data[i]['img_b'][j]
            show_img(img)
            show_img(img_b)
            print(i, j)
            img_patch = get_patch_1d(img_b, all_indexes, 16, 8)

            label = int(data[i]['labels'][j])

            if label == 7:
                label = 2

            #label = dense_to_one_hot(label, 6)

            x1.append((img, img_b))
            yl.append(label)

        total_x1.append(x1)
        total_y.append(yl)

    return total_x1, total_y


if __name__ == "__main__":
    img, label = pickle_2_img_single("D:/chenchuyang/learning/sparse_coding/patch_mnn/pkl/ckp_2sizeimg.pkl")

