# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 13:35:52 2019

This code is for testing different hand detection techniques.

Input: Images will be aquired from live camera feed.

Desired Output: A single hand contour which will only consist of the hand and there will be no noise in the background

Latest Output: - Still the output is rough with noise in the background

@author: Himanshu S (MegaPanda)
"""


import cv2
import numpy as np
import math



cap = cv2.VideoCapture(0)
cap.set(0,480)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    roi = frame[70:300,70:300]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv  =cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,30,70])#[0,20,70]
    upper_red = np.array([255,255,255])#[20,255,255]
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    contours, hei = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    contour_area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    
    hull = cv2.convexHull(cnt)
    cv2.drawContours(roi,[hull],-1,(0,255,0),3)
    
    hull = cv2.convexHull(cnt,returnPoints = False)
    
    defects = cv2.convexityDefects(cnt,hull)
    count_defects = 0
    cv2.drawContours(mask, contours, -1, (0,255,0), 3)
    
		
    
    #cv2.imshow('',)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(roi,mask)
    
    cv2.imshow('img', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('roi',roi)
    cv2.imshow('result',result)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()