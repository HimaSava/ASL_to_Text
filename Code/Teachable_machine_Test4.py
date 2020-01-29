# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:39:48 2019

@author: Himanshu
"""


import tensorflow.keras
from PIL import Image
import numpy as np
import cv2

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('D:/Offline_Projects/asl_dataset/keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
cap = cv2.VideoCapture(0)
cap.set(0,480)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    roi = frame[70:300,70:300]
    hsv  =cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,30,70])#[0,20,70]
    upper_red = np.array([255,255,255])#[20,255,255]
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(roi,mask)
# Replace this with the path to your image
#image = Image.open('D:/Offline_Projects/asl_dataset/b/B1.jpeg')

# Make sure to resize all images to 224, 224 otherwise they won't fit in the array
    image = cv2.resize(result,(224,224))  #roi.resize((224, 224))
    image_array = np.asarray(image)

# Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

# Load the image into the array
    data[0] = normalized_image_array

# run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0] == max(prediction[0]):
        letter = 'B'
    elif prediction[0][1] == max(prediction[0]):
        letter = 'C'
    elif prediction[0][2] == max(prediction[0]):
        letter = 'D'
    elif prediction[0][3] == max(prediction[0]):
        letter = 'E'
    elif prediction[0][4] == max(prediction[0]):
        letter = 'F'
    elif prediction[0][5] == max(prediction[0]):
        letter = 'G'
    elif prediction[0][6] == max(prediction[0]):
        letter = 'H'
    elif prediction[0][7] == max(prediction[0]):
        letter = 'I'
    elif prediction[0][8] == max(prediction[0]):
        letter = 'K'
    elif prediction[0][9] == max(prediction[0]):
        letter = 'L'
    elif prediction[0][10] == max(prediction[0]):
        letter = 'M'
    elif prediction[0][11] == max(prediction[0]):
        letter = 'N'
    elif prediction[0][12] == max(prediction[0]):
        letter = 'O'
    elif prediction[0][13] == max(prediction[0]):
        letter = 'P'
    elif prediction[0][14] == max(prediction[0]):
        letter = 'Q'
    elif prediction[0][15] == max(prediction[0]):
        letter = 'R'
    elif prediction[0][16] == max(prediction[0]):
        letter = 'S'
    elif prediction[0][17] == max(prediction[0]):
        letter = 'T'
    elif prediction[0][18] == max(prediction[0]):
        letter = 'U'
    elif prediction[0][19] == max(prediction[0]):
        letter = 'V'
    elif prediction[0][20] == max(prediction[0]):
        letter = 'W'
    elif prediction[0][21] == max(prediction[0]):
        letter = 'X'
    else:
        letter = 'Y'
    cv2.putText(frame, letter, (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
    cv2.imshow('img',frame)
    cv2.imshow('img2',roi)
    cv2.imshow('result',result)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

