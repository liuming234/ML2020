import torch
import numpy as np
import cv2 as cv
import random
import os
from os.path import join as pjoin
from pathlib import Path

perspective=True
perspective_x=0.2
perspective_y=0.2
scaling=True
scaling_amplitude=0.1
rotation=True
max_angle=60
translation=True
translation_overflow=0.
allow_artifacts=False

def homography(img):
    seed = random.randint(1,20)
    IMAGE_SHAPE = img.shape
    # persipective
    if perspective:
        location = np.random.rand(6)
        left_top_x = perspective_x*location[0]
        left_top_y = perspective_y*location[1]
        right_top_x = 0.9+perspective_x*location[2]
        right_top_y = perspective_y*location[3]
        left_bottom_x  = perspective_x*location[4]
        left_bottom_y  = 0.9 + perspective_y*location[1]
        right_bottom_x = 0.9 + perspective_x*location[5]
        right_bottom_y = 0.9 + perspective_y*location[4]
        pts = np.array([(IMAGE_SHAPE[1]*left_top_x,IMAGE_SHAPE[0]*left_top_y,1),
                    (IMAGE_SHAPE[1]*right_top_x, IMAGE_SHAPE[0]*right_top_y,1),
                    (IMAGE_SHAPE[1]*left_bottom_x,IMAGE_SHAPE[0]*left_bottom_y,1),
                    (IMAGE_SHAPE[1]*right_bottom_x,IMAGE_SHAPE[0]*right_bottom_y,1)],dtype = 'float32')
    # scaling and rotation
    center = (IMAGE_SHAPE[1]/2, IMAGE_SHAPE[0]/2)
    scale = 1
    rot = 0
    if scaling:
        scale = random.normalvariate(1, scaling_amplitude/2)
    if rotation:
        rot = random.uniform(-max_angle,max_angle)
    RS_mat = cv.getRotationMatrix2D(center, rot, scale)
    H = np.matmul(pts, RS_mat.T).astype('float32')
    return H

def pro_data(data_dir, dsc_dir):
    count = 0
    for item in os.listdir(data_dir):
        sorce_dir = pjoin(data_dir, item)
        desc_dir = pjoin(dsc_dir, item)
        if not Path(desc_dir).exists():
            os.mkdir(desc_dir)
        for i in os.listdir(sorce_dir):
            image_dir = pjoin(sorce_dir, i)
            warp_dir = pjoin(desc_dir, i)
            img = cv.imread(image_dir)

            IMAGE_SHAPE = img.shape
            dsc = homography(img)
            src = np.array([(0,0),
                            (IMAGE_SHAPE[1]-1, 0),
                            (0, IMAGE_SHAPE[0]-1),
                            (IMAGE_SHAPE[1]-1, IMAGE_SHAPE[0]-1)],
                           dtype = 'float32')
            mat = cv.getPerspectiveTransform(src, dsc)
            out_img = cv.warpPerspective(img, mat,(IMAGE_SHAPE[1],IMAGE_SHAPE[0]))

            cv.imwrite(warp_dir, out_img)
            print(warp_dir)
            count = count+1
    print(count)
    return count

src_dir = "/openbayes/input/input0"
dsc_dir = "/openbayes/home/pro_data"
pro_data(src_dir, dsc_dir)
