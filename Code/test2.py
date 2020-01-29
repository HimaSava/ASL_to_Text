# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 08:21:17 2019

@author: Himanshu
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  =cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,20,70])
    upper_red = np.array([20,255,255])
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.dilate(mask,kernel,iterations =4)
    mask = cv2.GaussianBlur(mask, (3,3), 100)
    res = cv2.bitwise_and(frame,frame, mask= mask)    
	    
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    k = cv2.waitKey(10) & 0xFF
    
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()