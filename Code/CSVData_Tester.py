# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 17:05:07 2019

@author: Himanshu
"""

import csv


data = []
with open('D:/Offline_Projects/SignLanguage_Recognition/DATABASE_CSV/B.csv','r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row != []:
            data.append(row)    
avg = []
sum = [0,0,0,0,0,0,0]
for img in data:
    sum[0] += float(img[2])
    sum[1] += float(img[5])
    sum[2] += float(img[6])
    sum[3] += float(img[7])
    sum[4] += float(img[8])
    sum[5] += float(img[9])
    sum[6] += float(img[10])
for i in range(len(sum)):
    sum[i] /=len(data)
    avg.append(round(sum[i],2))
with open('D:/Offline_Projects/B_Tester.csv','w') as file:
    writer = csv.writer(file,delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #writer.writerow(['moments','perimeter','area_diff','radius','hull_area','solidity','aspect_ratio','rect_area','extent','angle_t','count_defect'])
    writer.writerow(avg)

print(sum)