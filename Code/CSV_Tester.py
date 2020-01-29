# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 18:15:54 2019

@author: Himanshu
"""

import csv
import cv2
import numpy as np
import math
import statistics
temp = []
with open('D:/Offline_Projects/SignLanguage_Recognition/DATABASE_CSV/B_Tester.csv','rt') as csvfile:
    read = csv.reader(csvfile) 
    for row in read:
        temp.append(row)
data = temp[0]
data = list(map(float,data))
solidity_no = 0
aspect_ratio_no = 0
extent_no = 0
angle_t_no = 0
count_defect_no = 0
score_list = []
succ = 0
avg = 0
for i in range(1,61):
    score = 0 
    img_data = []
    
    cur_add = 'D:/Offline_Projects/asl_dataset/Test/Test' + str(i) + '.jpeg'
    print(cur_add)
    roi = cv2.imread(cur_add,-1)
    
    hsv  =cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,20,20])
    upper_red = np.array([20,255,255])
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.dilate(mask,kernel,iterations =4)
    mask = cv2.GaussianBlur(mask, (3,3), 100)
    contours, hei = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #contours, hei = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    x,y,w,h = cv2.boundingRect(cnt)
    
    hull = cv2.convexHull(cnt)
    cv2.drawContours(roi,[hull],-1,(0,255,0),3)
    
    hull = cv2.convexHull(cnt,returnPoints = False)
    
    defects = cv2.convexityDefects(cnt,hull)
    count_defects = 0
    cv2.drawContours(mask, contours, -1, (0,255,0), 3)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])    
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 60	      
        cv2.circle(roi,far,4,[0,0,255],-1)                    
        if angle<=60:
            count_defects+=1
            cv2.circle(roi,far,4,[0,255,0],-1)                    
#        print('Angle:', angle)
    
    moments = cv2.moments(cnt)   
    perimeter = cv2.arcLength(cnt,True)
    area = cv2.contourArea(cnt)
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv2.circle(roi,center,radius,(255,0,0),2)
    area_of_circle=math.pi*radius*radius
    hull_test = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull_test)
    solidity = float(area)/hull_area
    aspect_ratio = float(w)/h
    rect_area = w*h
    extent = float(area)/rect_area
    (x,y),(MA,ma),angle_t = cv2.fitEllipse(cnt)
    if data[1]+data[1]*0.1 > solidity > data[1]-data[1]*0.1:
        score += 2
        solidity_no += 1
    if data[2]+data[2]*0.1 > aspect_ratio > data[2]-data[2]*0.1:
        score += 2
        aspect_ratio_no +=1
    if data[4]+data[4]*0.1 > extent > data[4]-data[4]*0.1:
        score += 2
        extent_no +=1
    if data[5]+30 > angle_t > data[5]-30:
        score+=3
        angle_t_no +=1
    if data[6]+0.75 > count_defects > data[6]-0.75:
        score += 3
        count_defect_no +=1
    score_list.append(score)
    if score >= 5:
        print('It is B and the score is:', score)
        succ += 1
    else:
        print('failed with score',score)
print('The avg score is:',statistics.mean(score_list))
print('The mode score is:', statistics.mode(score_list))
print('The no of success are', succ)
print('The success rate is:',succ/len(score_list)*100)
print('solidity_no',solidity_no)
print('aspect_ratio_no',aspect_ratio_no)
print('extent_no',extent_no)
print('angle_t_no',angle_t_no)
print('count_defect_no',count_defect_no)