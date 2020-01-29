# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:30:46 2019

@author: Himanshu

This program reads the photos from the files. Makes the contour calculations and deposits the data in a csv format

"""
import cv2
import numpy as np
import math
import csv

font = cv2.FONT_HERSHEY_SIMPLEX
address = 'D:/Offline_Projects/asl_dataset/'

'D:/Offline_Projects/asl_dataset/a/A1.jpeg'

for j in range (65,90):
    data_list = []
    if j == 74 or j == 84:
        continue
    for i in range(1,71):
        img_data = []
        
        cur_add = address + chr(j+32) + '/' + chr(j) + str(i) + '.jpeg'
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
        img_data.append(moments)
        img_data.append(perimeter)
        img_data.append(area_of_circle - area)
        img_data.append(radius)
        img_data.append(hull_area)
        img_data.append(solidity)
        img_data.append(aspect_ratio)
        img_data.append(rect_area)
        img_data.append(extent)
        img_data.append(angle_t)
        img_data.append(count_defects)
        data_list.append(img_data)
        cv2.imshow('frame',roi)
#        cv2.imshow('mask',mask)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
    with open('D:/Offline_Projects/' + chr(j) + '.csv','w') as file:
        writer = csv.writer(file,delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #writer.writerow(['moments','perimeter','area_diff','radius','hull_area','solidity','aspect_ratio','rect_area','extent','angle_t','count_defect'])
        writer.writerows(data_list)
    print(chr(j) + ' Done')
j = 84
data_list = []
for i in range(1,66):
    img_data = []
        
    cur_add = address + chr(j+32) + '/' + chr(j) + str(i) + '.jpeg'
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
    img_data.append(moments)
    img_data.append(perimeter)
    img_data.append(area_of_circle - area)
    img_data.append(radius)
    img_data.append(hull_area)
    img_data.append(solidity)
    img_data.append(aspect_ratio)
    img_data.append(rect_area)
    img_data.append(extent)
    img_data.append(angle_t)
    img_data.append(count_defects)
    data_list.append(img_data)
with open('D:/Offline_Projects/T.csv','w') as file:
    writer = csv.writer(file,delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #writer.writerow(['moments','perimeter','area_diff','radius','hull_area','solidity','aspect_ratio','rect_area','extent','angle_t','count_defect'])
    writer.writerows(data_list)