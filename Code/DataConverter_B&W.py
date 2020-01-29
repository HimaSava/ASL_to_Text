# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 09:23:09 2019

@author: Himanshu
"""

import cv2
import numpy as np
import math
import csv
import os

font = cv2.FONT_HERSHEY_SIMPLEX
address = 'D:/Offline_Projects/asl_dataset/'

#'D:/Offline_Projects/asl_dataset/a/A1.jpeg'

for j in range (65,90):#(65,90)
    data_list = []
    pics = os.listdir(address + chr(j+32))
    print(chr(j+31) + " Done")
    os.chdir(address)
    try:
        os.mkdir(chr(j) + '_B&W')
    except:
        os.chdir(address)
    for i in pics:
        cur_add = address + chr(j+32) + '/' + i
        #cur_add = 'D:/Offline_Projects/asl_dataset/a/A4.jpeg' 
        print(cur_add)
        roi = cv2.imread(cur_add)
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0,20,20])
        upper_red = np.array([200,255,255])
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        mask = cv2.dilate(mask,kernel,iterations =4)
        mask = cv2.GaussianBlur(mask, (3,3), 100)
        
        cv2.imwrite(address + chr(j) + '_B&W' + '/' + i,mask)
        