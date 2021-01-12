import FaceProcessUtil as fpu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib
import scipy.fftpack as FFT
import os
import pandas as pd


def show_img(img, title='default'):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


imgpath = r"D:\chenchuyang\learning\FNN\fera\cohn-kanade-images\cohn-kanade-images\S010\006\S010_006_00000001.png"
#imgcv_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
flag, imgcv_gray = fpu.calibrateImge(imgpath)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./dlibmodel/shape_predictor_68_face_landmarks.dat")
res = fpu.getLandMarkFeatures_and_ImgPatches(imgcv_gray, withLM=True, withPatches=True)


# define the eye and mouth indexes for local feature
eye_indexes = list(range(36, 48))
mouth_indexes = list(range(48, 60))
all_indexes = eye_indexes + mouth_indexes


# 將dlib偵測到的人臉68個特徵點取出
def _shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


# 偵測單一人臉的臉部特徵
def get_landmarks(im, face_detector, shape_predictor):
    rects = face_detector(im, 1)
    shape = shape_predictor(im, rects[0])
    coords = _shape_to_np(shape, dtype="int")
    lms_x = coords[:, 0]
    lms_y = coords[:, 1]
    return lms_x, lms_y


def get_EyeMouthMiddle(img, X, Y):
    # resize to target
    patches = []
    h = img.shape[0]
    w = img.shape[1]
    img_eye = fpu.__getEyePatch(img, w, h, X, Y)
    img_eye_resize = cv2.resize(img_eye, (64, 24))
    img_mouth = fpu.__getMouthPatch(img, w, h, X, Y)
    img_mouth_resize = cv2.resize(img_mouth, (56, 32))
    img_middle = fpu.__getMiddlePatch(img, w, h, X, Y)
    patches.append(img_eye_resize)
    patches.append(img_middle)
    patches.append(img_mouth_resize)
    return patches


def get_SqrLeye(img, X, Y, size):
    cx = np.mean(X[36:42])
    cy = np.mean(Y[36:42])
    half = size/2
    tlx, tly = int(cx - half), int(cy - half)
    brx, bry = int(cx + half), int(cy + half)
    patch = np.zeros((size, size), dtype="uint8")
    patch[0:size, 0:size] = img[tly:bry, tlx:brx]
    return patch


def get_SqrReye(img, X, Y, size):
    cx = np.mean(X[42:48])
    cy = np.mean(Y[42:48])
    half = size/2
    tlx, tly = int(cx - half), int(cy - half)
    brx, bry = int(cx + half), int(cy + half)
    patch = np.zeros((size, size), dtype="uint8")
    patch[0:size, 0:size] = img[tly:bry, tlx:brx]
    return patch


def get_SqrMouth(img, X, Y, size):
    cx = np.mean(X[48:68])
    cy = np.mean(Y[48:68])
    half = size/2
    tlx, tly = int(cx - half), int(cy - half)
    brx, bry = int(cx + half), int(cy + half)
    patch = np.zeros((size, size), dtype="uint8")
    patch[0:size, 0:size] = img[tly:bry, tlx:brx]
    return patch


# old version of eye, mouth patch
def get_FacePatch(img, X, Y):
    leye_patch = get_SqrLeye(img, X, Y, 80)
    reye_patch = get_SqrReye(img, X, Y, 80)
    mouth_patch = get_SqrMouth(img, X, Y, 104)
    return leye_patch, reye_patch, mouth_patch


# get all image blocks returns 1d tensor
def get_all_lm_blocks(img, X, Y, block_size=8):
    patches = []
    for x, y in zip(X, Y):
        patch = get_landmark_block(img, x, y, block_size)
        patch = FFT.dctn(patch)[:block_size, :block_size]
        patches.append(patch)
        patch_1d = np.array(patches).flatten()
    return patch_1d


# get one block by the landmark position
def get_landmark_block(img, x, y, block_size=8):
    half = block_size / 2
    tlx, tly = int(x - half), int(y - half)
    brx, bry = int(x + half), int(y + half)
    patch = np.zeros((block_size, block_size), dtype="uint8")
    print(block_size, tly, bry, tlx, brx)
    patch[0:block_size, 0:block_size] = img[tly:bry, tlx:brx]
    return patch


def get_landmark_indexes(all_indexes, lmx, lmy):
    x_lms_list = list()
    y_lms_list = list()
    for index_x, index_y in zip(all_indexes, all_indexes):
        x_lms_list.append(lmx[index_x])
        y_lms_list.append(lmy[index_y])
    return x_lms_list, y_lms_list


def get_patch_1d(img, indexes_list):
    lmx, lmy = get_landmarks(img, detector, predictor)
    x_lms_list, y_lms_list = get_landmark_indexes(indexes_list, lmx, lmy)
    patch_1d_tensor = get_all_lm_blocks(img, x_lms_list, y_lms_list)
    return patch_1d_tensor


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return x, y, w, h


# crop by the rectangle and resize
def crop_and_resize(img, type = 0):
    rects = detector(img, type)
    x, y, w, h = rect_to_bb(rects[0])
    tlx, tly = x, y
    brx, bry = x + w, y + h
    patch = np.zeros((h, w), dtype="uint8")
    patch[0:h, 0:w] = img[tly:bry, tlx:brx]
    im_rescale = cv2.resize(patch, (128, 128))
    return im_rescale


def show_landmarks_sqrs(X, Y, img, block_size=8):
    half = block_size//2
    x_indexes, y_indexes = get_landmark_indexes(all_indexes, X, Y)
    plt.scatter(X, Y)
    for (x, y) in zip(x_indexes, y_indexes):
        print(x,y)
        cv2.rectangle(img, (x - half, y - half), (x + half, y + half), (255, 0, 0))
    plt.imshow(img, cmap='gray')
    plt.show()




# for image show, big square
def show_landmarks_sqrs_center(X, Y, img, eye_area=12, mouth_area=12):
    plt.scatter(X, Y)
    center_reye_x = int(np.mean(X[42:48]))
    center_reye_y = int(np.mean(Y[42:48]))
    center_leye_x = int(np.mean(X[36:42]))
    center_leye_y = int(np.mean(Y[36:42]))
    center_mouth_x = int(np.mean(X[48:68]))
    center_mouth_y = int(np.mean(Y[48:68]))
    # plt.scatter(center_leye_x, center_leye_y)
    # print(center_leye_x, center_leye_y)
    cv2.rectangle(img, (center_leye_x - eye_area, center_leye_y - eye_area), (center_leye_x + eye_area, center_leye_y + eye_area), (255, 0, 0))
    cv2.rectangle(img, (center_reye_x - eye_area, center_reye_y - eye_area), (center_reye_x + eye_area, center_reye_y + eye_area), (255, 0, 0))
    cv2.rectangle(img, (center_mouth_x - mouth_area-12, center_mouth_y - mouth_area), (center_mouth_x + mouth_area-12, center_mouth_y + mouth_area), (255, 0, 0))
    cv2.rectangle(img, (center_mouth_x - mouth_area+12, center_mouth_y - mouth_area), (center_mouth_x + mouth_area+12, center_mouth_y + mouth_area), (255, 0, 0))

    # for x, y in zip(xl_select, yl_select):
    #     cv2.rectangle(img, (x-4, y-4), (x+4, y+4), (255,0,0))
    plt.imshow(img, cmap='gray')
    plt.title("landmarks")
    plt.show()


# show the landmarks and the rectangle of the border
def show_landmarks_and_rect(img):
    lmx, lmy = get_landmarks(img, detector, predictor)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rects = detector(img, 0)
    x, y, w, h = rect_to_bb(rects[0])
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor = 'red',linewidth=1)
    plt.imshow(img)
    ax.add_patch(rect)
    plt.scatter(lmx, lmy)
    plt.show()


def show_landmarks_and_img(X, Y, img):
    plt.scatter(X, Y)
    plt.imshow(img, cmap='gray')
    plt.title("landmarks")
    plt.show()


def pre_show():
        print(imgcv_gray.shape)
        show_img(imgcv_gray, 'original')
        print('length: ',len(res))
        show_img(res[0], 'inner')
        show_img(res[4], 'before_eye')
        show_img(res[5], 'before_middle')
        show_img(res[6], 'before_mouth')
        show_img(res[7], 'before gray')
        lms_x, lms_y = get_landmarks(imgcv_gray, detector, predictor)
        eye_patch, middle_patch, mouth_patch = get_EyeMouthMiddle(imgcv_gray, lms_x, lms_y)

        show_img(eye_patch, 'eye')
        print(eye_patch.shape)
        show_img(middle_patch, 'middle')
        show_img(mouth_patch, 'mouth')
        print(mouth_patch.shape)
        show_landmarks_and_img(lms_x, lms_y, imgcv_gray)
        leye_patch = get_SqrLeye(imgcv_gray, lms_x, lms_y, 80)
        reye_patch = get_SqrReye(imgcv_gray, lms_x, lms_y, 80)
        mouth_patch = get_SqrMouth(imgcv_gray, lms_x, lms_y, 104)
        print("aaa", mouth_patch.shape)
        cx = np.mean(np.concatenate([lms_x[17:27],lms_x[36:48]]))
        cy = np.mean(np.concatenate([lms_y[17:27],lms_y[36:48]]))
        show_img(leye_patch, "leye_sqr")
        show_img(reye_patch, "reye_sqr")
        show_img(mouth_patch, "mouth_sqr")
        inner_face = res[0]
        show_img(inner_face, "inner")
        lmx, lmy = get_landmarks(inner_face, detector, predictor)
        print(lmx.shape, lmy.shape)
        show_landmarks_and_img(lmx, lmy, inner_face)
        eye_indexes = list(range(36, 48))
        mouth_indexes = list(range(48, 60))
        all_indexes = eye_indexes + mouth_indexes
        # x, y = lmx[36], lmy[36]
        # x_lms, y_lms = lmx
        # patch = get_landmark_block(inner_face, x, y, 16)
        # show_img(patch, 'patch')
        print(all_indexes)
        # x_list and y_list are the chosen landmark points
        x_list = list()
        y_list = list()
        for index_x, index_y in zip(all_indexes, all_indexes):
            x_list.append(lmx[index_x])
            y_list.append(lmy[index_y])

        print(x_list)
        print(y_list)
        print(lmx[36], lmx[37])
        print(lmy[36], lmy[37])

        patches = get_all_lm_blocks(inner_face, x_list, y_list)
        print(len(patches))
        print('pat', patches[0].shape)
        patch_mat = np.array(patches).flatten()
        print(patch_mat.shape)

        patch_tensor = get_patch_1d(imgcv_gray, all_indexes, 16, 8)
        print(patch_tensor.shape)

        imr = crop_and_resize(imgcv_gray)
        lmx, lmy = get_landmarks(imr, detector, predictor)

        lmx_select, lmy_select = get_landmark_indexes(all_indexes, lmx, lmy)
        show_img(imr)
        show_landmarks_sqrs_center(lmx, lmy, imr)

        im_static = np.abs(FFT.dctn(imr).flatten())
        im_big = np.abs(FFT.dctn(imr).flatten())
        im_big_list = list(im_big)
        print(type(im_static))
        im_static_list = list(im_static)

        img_high1 = np.abs(FFT.dctn(imr)[32:128, :32].flatten())
        img_high2 = np.abs(FFT.dctn(imr)[:32, 32:128].flatten())
        img_high3 = np.abs(FFT.dctn(imr)[32:,32:].flatten())
        l = list(img_high1) + list(img_high2) + list(img_high3)

        x = l
        x = pd.Series(x)

        # histogram on linear scale
        plt.subplot(211)
        hist, bins, _ = plt.hist(x, bins=8)

        # histogram on log scale.
        # Use non-equal bin sizes, such that they look equal on log scale.
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
        plt.subplot(212)
        plt.hist(x, bins=logbins)
        plt.xscale('log')
        plt.show()


        # Normalization

if __name__ == "__main__":
    show_img(imgcv_gray, '001')
    rects = detector(imgcv_gray, 1)
    shape = predictor(imgcv_gray, rects[0])

    crop_img = crop_and_resize(imgcv_gray)

    crop_img2 = fpu.crop_face_only(imgcv_gray, shape=shape)
    crop_img3 = fpu.crop_face_for_patch(imgcv_gray, shape=shape, ratio=0.2)
    show_img(crop_img, '2')
    # the difference of the two is detector 1 and 0
    show_img(crop_img2, '3')
    show_img(res[0], '4')
    show_img(crop_img3, '5')
    lmx, lmy = get_landmarks(crop_img3,detector,predictor)
    lmx_select, lmy_select = get_landmark_indexes(all_indexes, lmx, lmy)
    show_landmarks_and_img(lmx, lmy, crop_img3)
    # show_landmarks_sqrs_center(lmx, lmy, crop_img3)
    # get the patches of the face
    show_landmarks_sqrs(lmx, lmy, crop_img3, block_size=8)












    #show_landmarks_and_rect(imgcv_gray)
    # root_path = r"D:\chenchuyang\learning\FNN\fera\cohn-kanade-images\cohn-kanade-images\S124\007\\"
    # for path in os.listdir(root_path):
    #     print(path)
    #     if path[-3:] == 'png':
    #         imgcv = cv2.imread(root_path + path, cv2.IMREAD_GRAYSCALE)
    #         print(type(imgcv))
    #         show_img(imgcv)
    #         show_landmarks_and_rect(imgcv)


