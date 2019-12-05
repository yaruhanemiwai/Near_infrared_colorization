#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import h5py
import sys,os
sys.path.append('/home/es1video4/workspace/iwamine/src/')
import function
import copy
import random
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, color, exposure
    
def gethog(folder_data):
    hog_0 = np.zeros([480,640,2,20],dtype = np.float32)
    for img_num in range(0,20):

        img = cv2.imread(folder_data+str(img_num)+".bmp",0)
        twiceImg = cv2.resize(img, None, fx = 2, fy = 2)

        fd_0, hog_image = hog(twiceImg, orientations=90, pixels_per_cell=(2, 2),\
            cells_per_block=(1, 1), visualise=True)
        fd = fd_0.reshape([img.shape[1],img.shape[0],90]).transpose(1,0,2)

        for height_0 in range(img.shape[0]):
            for width_0 in range(img.shape[1]):

                a = fd[height_0,width_0,:]
                hog_0[height_0,width_0,:,img_num] =\
                    np.asarray([a.argmax()*1.0/90,a.max()*1.0]).astype(np.float32)

    return hog_0

def main():
    #Setting
    num = 223840*2
    folder_data = '/home/es1video4/workspace/iwamine/sekigaisen/NightVision/train/ori/'
    folder_label = '/home/es1video4/workspace/iwamine/sekigaisen/Color/train/'
    txt_path = "data_hog/"
    size_input = 30
    size_label = 20
    batch = 20
    list_data = os.listdir(folder_data) 
    
    hog = gethog(folder_data)
            
    for inum in range(0,batch):
        f = './data_hog/train_'+str(inum)+'.h5'
        outfh = h5py.File(f,"w")
        txt = open(txt_path+"path_train.txt",mode = "a")
        data = np.zeros([size_input,size_input,3,num/batch],dtype = np.float32)
        label = np.zeros([size_label,size_label,3,num/batch],dtype = np.float32)
        hog_0 = np.zeros([size_input,size_input,2,num/batch],dtype = np.float32)
                                
        for i in range(inum*(num/batch),(inum+1)*(num/batch)):

            data_rand = random.choice(list_data)
            fd = hog[:,:,:,int(data_rand[:-4])]
            image = cv2.imread(folder_data+str(data_rand))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
            label_0 = cv2.imread(folder_label+str(data_rand))
            label_0 = cv2.cvtColor(label_0, cv2.COLOR_BGR2YCR_CB)

            if random.randint(1,2) == 1:

                image = function.horizontal_flip(image)
                label_0 = function.horizontal_flip(label_0)
                fd = function.horizontal_flip(fd)          

            image = function.random_crop(image,crop_size=(30,30))
            label_0 = function.random_crop(label_0,crop_size=(30,30))
            fd = function.random_crop(fd,crop_size=(30,30))
            
            image = (image*1.0/255).astype(np.float32)
            label_0 = (label_0*1.0/255).astype(np.float32)
            
            data[:,:,:,i%(num/batch)] = image
            hog_0[:,:,:,i%(num/batch)] = fd
            label[:,:,:,i%(num/batch)] = label_0

        outfh.create_dataset('data',data = data.transpose(3,2,0,1))
        outfh.create_dataset('label',data = label.transpose(3,2,0,1))
        outfh.create_dataset('hog',data = hog_0.transpose(3,2,0,1))
        outfh.flush()
        outfh.close()  
        txt.write(f+"\n")
        txt.close()

if __name__ == '__main__':
    main()     
