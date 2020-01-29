# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:05:38 2019

@author: Himanshu

Prototype Number: 1

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
#    blurred = cv2.GaussianBlur(gray, (35,35), 0)
#    _, thresh = cv2.threshold(blurred, 0,155, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    hsv  =cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,30,70])#[0,20,70]
    upper_red = np.array([255,255,255])#[20,255,255]
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    #mask = cv2.dilate(mask,kernel,iterations =4)
    #mask = cv2.GaussianBlur(mask, (3,3), 100)
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
        if angle<=90:
            count_defects+=1
    
    moment = cv2.moments(cnt)   
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
		
    if area_of_circle - area < 5000:
        cv2.putText(frame, "A", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)

    elif count_defects ==1:
        if angle_t < 10:
            cv2.putText(frame, "V", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
        elif 40 < angle_t < 66:
            cv2.putText(frame, "C", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
        elif 20 < angle_t < 35:
            cv2.putText(frame, "L", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Y", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
            
    elif count_defects == 2:
        if angle_t > 100:
            cv2.putText(frame, "F", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, "W", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
    elif count_defects == 4:
        cv2.putText(frame, "Hello There ! Callibrate by letter A", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
    else:
        if area > 12000:
            cv2.putText(frame, "B", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
        else:
            if solidity < 0.85:
                if aspect_ratio < 1:
                    if angle_t < 20:
                        cv2.putText(frame, "D", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
                    elif 169 < angle_t < 180:
                        cv2.putText(frame, "I", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
                    elif angle_t < 168:
                        cv2.putText(frame, "J", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
                elif aspect_ratio > 1.01:
                    cv2.putText(frame, "Y", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
            else:
                if 30 < angle_t < 100:
                    cv2.putText(frame, "H", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
                elif angle_t > 120:
                    cv2.putText(frame, "I", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "U", (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
    cv2.imshow('img', frame)
    cv2.imshow('blurred', mask)
    
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()