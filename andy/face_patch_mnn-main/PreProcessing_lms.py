import os
import pickle
import cv2
import numpy as np
import scipy.fftpack as FFT

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
        x2 = []
        yl = []
        print(len(data[i]['img']))
        for j in range(len(data[i]['labels'])):
            img = data[i]['img'][j]
            img = FFT.dctn(img)
            img_neu = data[i]['img_neu'][j]
            img_neu = FFT.dctn(img_neu)
            diff = img - img_neu
            lms = data[i]['lms'][j]
            lms = np.array(lms)
            '''
             img = data[i]['img'][j]
             img_neu = data[i]['img_neu'][j]
             diff = img - img_neu
             diff = FFT.dctn(diff)
             '''
            label = int(data[i]['labels'][j])
            if label == 7:
                label = 2

            #label = dense_to_one_hot(label, 6)

            x1.append(lms)
            yl.append(label)

        total_x1.append(x1)
        total_y.append(yl)

    return total_x1, total_y
