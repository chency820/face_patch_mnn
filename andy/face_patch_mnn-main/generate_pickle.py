import cv2
import time
import ntpath
import sys, os
import pickle
import FaceProcessUtil as fpu
import numpy as np
import dlib
from facePatch import get_patch_1d, crop_and_resize

# generate pickle for last three frames

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./dlibmodel/shape_predictor_68_face_landmarks.dat")

eye_indexes = list(range(36, 48))
mouth_indexes = list(range(48, 60))
all_indexes = eye_indexes + mouth_indexes

#txt path
txtPath = './txt/CK+_106_single_309.txt'

#replace your data path of CK+
data_root_path = 'D:/chenchuyang/learning/FNN/fera/cohn-kanade-images/cohn-kanade-images/'

imglist = [[],[],[],[],[],[],[],[],[],[]] #ten fold
labellist = [[],[],[],[],[],[],[],[],[],[]] #ten fold

list_txt = open(txtPath,'r')
content = list_txt.readlines()

for i, line in enumerate(content):
    line = line.replace('\n','')
    line = line.split('\t')
    group = int(line[0])
    path = line[1]
    base_path = path[:-12]
    # if neutral expression included:
    # path_neu = data_root_path + base_path + '00000001.png'
    path_num_part = path[-12:-4]
    path_last3 = data_root_path + base_path + '0' * (8 - len(str(int(path_num_part) - 2))) + str(
        int(path_num_part) - 2) + '.png'
    path_last2 = data_root_path + base_path + '0' * (8 - len(str(int(path_num_part) - 1))) + str(
        int(path_num_part) - 1) + '.png'
    path_last1 = data_root_path + path
    print(path_last3, path_last2, path_last1)
    label = line[2]
    imglist[group].append(path_last3)
    imglist[group].append(path_last2)
    imglist[group].append(path_last1)
    labellist[group].append(label)
    labellist[group].append(label)
    labellist[group].append(label)

total = 0

for i ,e in enumerate(labellist):
   # print(len(e))
    total = total+len(e)
print("Total training images:%d"%(total))


count=0

gc = 0

feature_group_of_subject = []

tm1=time.time()

for i in range(len(imglist)):
    ckplus={}
    ckplus_label=[]
    ckplus_img = []
    ckplus_imgb = []
    # ckplus_img_neu = []
    # ckplus_lms = []
    imagelist = imglist[i]
    # imagelist_neu = imglist_neu[i]
    lablist = labellist[i]
    for j,v in enumerate(imagelist):
        count = count+1
        label = int(lablist[j])
        #-----map label-----
        if label == 7:
            label = 2
        #-----map label-----
        label = label-1
        print("\n> Prepare image                                         %f%%"%(count*100/total))

        image_path = v
        print('1', image_path)
        if image_path != "D:/chenchuyang/learning/FNN/fera/cohn-kanade-images/cohn-kanade-images/S129/002/S129_002_0000009.png":
            flag, img = fpu.calibrateImge(image_path)
            if flag:
                    imgr = fpu.getLandMarkFeatures_and_ImgPatches(img, True, False, True)
                    img_cnr = crop_and_resize(img)
                    #img_patch = img = get_patch_1d(img, all_indexes, 16, 8)
                    #lms = get_norm_landmarks(img, detector, predictor)
            else:
                print('Unexpected case while calibrating for:'+str(image_path))
                exit(1)

            if imgr[1]:
                gc = gc + 1
                img = imgr[0]
                imgb = img_cnr


                print("Get Geometry>>>>>>>>>>>>>>")
                ckplus_label.append(label)
                ckplus_img.append(img)
                ckplus_imgb.append(imgb)
                # ckplus_lms.append(lms)
            else:
                print('No feature detected:' + image_path)
                exit(1)

    ckplus['labels']=ckplus_label
    ckplus['img'] = ckplus_img
    ckplus['img_b'] = ckplus_imgb
    #ckplus['lms'] = ckplus_lms
    feature_group_of_subject.append(ckplus)

filenametosave='./pkl/ckp_3_img.pkl'

with open(filenametosave,'wb') as fin:
    pickle.dump(feature_group_of_subject, fin, 4)

tm2=time.time()
dtm = tm2-tm1
print('Total images: %d\tGet: %d'%(count, gc))
print("Total time comsuming: %fs for %d images"%(dtm, count))