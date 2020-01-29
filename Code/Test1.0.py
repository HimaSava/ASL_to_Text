#import cv2
#import numpy as np

#cap = cv2.VideoCapture(0)
#
#while True:
#    ret, frame = cap.read()
#    kernel = np.ones((3,3),np.uint8)
#    
#    roi=frame[100:300, 100:300]
#    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
#    
#    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#    
#    # define range of skin color in HSV
#    lower_skin = np.array([0,20,70], dtype=np.uint8)
#    upper_skin = np.array([20,255,255], dtype=np.uint8)
#        
#     #extract skin colur imagw  
#    mask = cv2.inRange(hsv, lower_skin, upper_skin)
#        
#   
#        
#    #extrapolate the hand to fill dark spots within
#    mask = cv2.dilate(mask,kernel,iterations = 4)
#        
#    #blur the image
#    mask = cv2.GaussianBlur(mask,(5,5),100) 
#    
#    
#    contours, hie = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#    
#    cnt = max(contours, key = lambda x: cv2.contourArea(x))
#    
#    cv2.drawContours(roi, cnt, -1, (255,0,0), 2)
#    cv2.imshow('frame',frame)
#    cv2.imshow('mask',mask)    
#
#    k = cv2.waitKey(5) & 0xFF
#    if k == 27:
#        break
#
#cv2.destroyAllWindows()
#cap.release()

import cv2
import numpy as np
cap = cv2.VideoCapture(0)
lower_thresh1 = 150
upper_thresh1 = 255     
while(1):
        
  #an error comes if it does not find anything in window as it cannot find contour of max area
#          #therefore this try error statement
#          
#    ret, frame = cap.read()
#    #frame=cv2.flip(frame,1)
#    kernel = np.ones((3,3),np.uint8)
#        
#        #define region of interest
#    roi=frame[100:300, 100:300]
#        
#        
#    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)    
#    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#        
#        
#         
#    # define range of skin color in HSV
#    lower_skin = np.array([0,20,70], dtype=np.uint8)
#    upper_skin = np.array([20,255,255], dtype=np.uint8)
#        
#     #extract skin colur imagw  
#    mask = cv2.inRange(hsv, lower_skin, upper_skin)
#        
#   
#        
#    #extrapolate the hand to fill dark spots within
#    mask = cv2.dilate(mask,kernel,iterations = 4)
#        
#    #blr the image
#    mask = cv2.GaussianBlur(mask,(5,5),100) 
#    
#        
#    
#    #fid contours
#    contours,hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
#   #fin contour of max area(hand)
#    cnt = max(contours, key = lambda x: cv2.contourArea(x))
#   
#    #aprox the contour a little
#    epsilon = 0.0005*cv2.arcLength(cnt,True)
#    approx= cv2.approxPolyDP(cnt,epsilon,True)
#      
#    
#    #mae convex hull around hand
#    hull = cv2.convexHull(cnt)
#    
#     #dfine area of hull and area of hand
#    areahull = cv2.contourArea(hull)
#    areacnt = cv2.contourArea(cnt)
#     
#    #fid the percentage of area not covered by hand in convex hull
#    arearatio=((areahull-areacnt)/areacnt)*100
#
#     #fnd the defects in convex hull with respect to hand
#    hull = cv2.convexHull(approx, returnPoints=False)
#    defects = cv2.convexityDefects(approx, hull)
#    
#    # l= no. of defects
#    l=0
#    
#    #coe for finding no. of defects due to fingers
#    for i in range(defects.shape[0]):
#        s,e,f,d = defects[i,0]
#        start = tuple(approx[s][0])
#        end = tuple(approx[e][0])
#        far = tuple(approx[f][0])
#        pt= (100,180)
#        
#        
#        # find length of all sides of triangle
#        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
#        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
#        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
#        s = (a+b+c)/2
#        ar = math.sqrt(s*(s-a)*(s-b)*(s-c))
#        
#        #distance between point and convex hull
#        d=(2*ar)/a
#        
#        # apply cosine rule here
#        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
#        
#    font = cv2.FONT_HERSHEY_SIMPLEX
#    # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
#    if angle <= 90 and d>30:
#        l += 1
#        cv2.circle(roi, far, 3, [255,0,0], -1)
#        
#        #draw lines around hand
#        cv2.line(roi,start, end, [0,255,0], 2)
#        
#            
#        l+=1
#        
#        #print corresponding gestures which are in their ranges
#        
#        if l==1:
#            if areacnt<2000:
#                cv2.putText(frame,'Put hand in the box',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#            else:
#                if arearatio<12:
#                    cv2.putText(frame,'0',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#                elif arearatio<17.5:
#                    cv2.putText(frame,'Best of luck',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#                        
#                else:
#                    cv2.putText(frame,'1',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#                
#    elif l==2:
#        cv2.putText(frame,'2',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#        
#    elif l==3:
#     
#        if arearatio<27:
#            cv2.putText(frame,'3',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#        else:
#            cv2.putText(frame,'ok',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#                    
#    elif l==4:
#        cv2.putText(frame,'4',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#        
#    elif l==5:
#        cv2.putText(frame,'5',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#        
#    elif l==6:
#        cv2.putText(frame,'reposition',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#            
#    else :
#        cv2.putText(frame,'reposition',(10,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
#            
#    #show the windows
#    cv2.imshow('mask',mask)
#    cv2.imshow('frame',frame)
#
#    
#    k = cv2.waitKey(5) & 0xFF
#    if k == 27:
#        break

	    ret, img = cap.read()

	    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	    #edges = cv2.Canny(crop_img,100,200)



	    ############################### Corners Detection 


	    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	    
	    lower_red = np.array([0,150,50])
	    

	    upper_red = np.array([195,255,255])
	    
	    

	    
	    mask = cv2.inRange(hsv, lower_red, upper_red)
	    res = cv2.bitwise_and(img,img, mask= mask)




	    value = (35, 35)
	    blurred = cv2.GaussianBlur(grey, value, 0)
	    _, thresh1 = cv2.threshold(blurred, lower_thresh1, upper_thresh1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	    #_, thresh_dofh = cv2.threshold(img_dofh, lower_thresh1, upper_thresh1, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	    #defrt,thresh_for_I = cv2.threshold(letter_I_match_shape,125,255,0)



	    contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#	    xyz,contours_dofh, hierarchy_dofh = cv2.findContours(thresh_dofh,2,1)
	    #cnt_dofh = contours_dofh[1]

	    cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
	    cv2.imshow('mask', mask)
	    cv2.drawContours(img,[cnt],0,(0,255,255),0)
	    cv2.imshow('Image', img)  

      	   
cv2.destroyAllWindows()
cap.release()    
