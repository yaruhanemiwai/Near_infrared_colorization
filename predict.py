#!/usr/bin/env python
# -*- coding: utf-8 -*-
#dataaugmentation
import numpy as np
import sys,os
sys.path.insert(0,os.path.join('/home/es1video4/caffe','python'))
sys.path.append('/home/es1video4/workspace/iwamine/src/')
import caffe
import function
import cv2
import copy

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
    input_dir = 'NightVision/val/ori/'
    gt_dir    = 'Color_val/'

    caffe.set_mode_gpu()

    for i in range(0,101):

        net = caffe.Net('MRCNN_deploy.prototxt',"./result_fine/seg_eucilidean_0/model_iter_"+str(i*50000)+".caffemodel",caffe.TEST)

        
        for fn in os.listdir(input_dir):
            #print(str(fn))
            file_name = fn[:-4]
            # Inputs
            im = cv2.imread(input_dir + fn)
            im = cv2.cvtColor(im,cv2.COLOR_BGR2YCR_CB)

            im = (im*1.0/255).astype(np.float32)
            im = im[:,:,:,np.newaxis].transpose(3,2,0,1)
            #print(str(im.shape))

            net.blobs['data'].reshape(*(1, 3, im.shape[2], im.shape[3]))
            #net.forward() # dry run
            net.blobs['data'].data[...] = im
            out = net.forward()

            # Predict results
            mat = out['conv3_0']

            # Ground truth
            gt = cv2.imread(gt_dir + fn)
            gt = cv2.resize(gt,(160,120))

            mat = np.clip(mat,0.0,1.0)
            mat = np.transpose(np.squeeze(mat),(1,2,0))
            mat = cv2.cvtColor(mat,cv2.COLOR_BGR2YCR_CB)   
            
            mat = (mat*255).astype('uint8')

            cv2.imwrite("./result_watch/seg_eucilidean_"+str((i-60)*50000)+"_"+file_name+".bmp",mat)
            
            print(str(function.get_PSNR(mat,gt)))
