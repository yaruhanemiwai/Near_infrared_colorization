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

def main():
    #Setting
    num = 223840
    folder_data = 'NightVision/'
    folder_seg = 'Seg/'
    folder_label = 'Color/'
    txt_path = "data/"
    size_input = 30
    size_label = 20
    batch = 2
    list_data = os.listdir(folder_data)

    for inum in range(0,batch):

        f = './data/train_'+str(inum)+'.h5'
        outfh = h5py.File(f,"w")
        txt = open(txt_path+"path_train.txt",mode = "a")
        #data = np.zeros([size_input,size_input,3,num/batch],dtype = np.float32)
        data = np.zeros([size_input,size_input,4,num/batch],dtype = np.float32)
        label = np.zeros([size_label,size_label,3,num/batch],dtype = np.float32)

        for i in range(inum*(num/batch),(inum+1)*(num/batch)):

            data_rand = random.choice(list_data)
            image = cv2.imread(folder_data+str(data_rand))
            image = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
            seg = cv2.imread(folder_seg+str(data_rand),0)
            label_0 = cv2.imread(folder_label+str(data_rand))
            label_0 = cv2.cvtColor(label_0,cv2.COLOR_BGR2YCR_CB)            
            
            """
            if random.randint(1,2) == 1:
                size_rand = random.uniform(0.4, 3)
                image = cv2.resize(image,None,fx = size_rand,fy = size_rand)
                label_0 = cv2.resize(label_0,None,fx = size_rand,fy = size_rand)
            
            if random.randint(1,2) == 2: 
                image = function.horizontal_flip(image)
                label_0 = function.horizontal_flip(label_0)            
            
            if random.randint(1,2) == 2:
                image = function.rotation_usual(image)
                label_0 = function.rotation_usual(label_0)
            """

            image = (image*1.0/255).astype(np.float32)
            seg = (seg*1.0/255).astype(np.float32)
            label_0 = (label_0*1.0/255).astype(np.float32)
           
            image = function.random_crop(image,(30,30))
            label_0 = function.random_crop(label_0,(30,30))
            
            data[:,:,:3,i%(num/batch)] = image
            data[:,:,3,i%(num/batch)] = seg
            label[:,:,:,i%(num/batch)] = label_0

        outfh.create_dataset('data',data = data.transpose(3,2,0,1))
        outfh.create_dataset('label',data = label.transpose(3,2,0,1))
        outfh.flush()
        outfh.close()  
        txt.write(f+"\n")
        txt.close()     

if __name__ == '__main__':
    main()
