
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
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#    blurred = cv2.GaussianBlur(gray, (35,35), 0)
#    _, thresh = cv2.threshold(blurred, 0,155, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    hsv  =cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0,30,70])
    upper_red = np.array([20,255,255])
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.dilate(mask,kernel,iterations =4)
    mask = cv2.GaussianBlur(mask, (3,3), 100)
    
    contours, hei = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    cnt = max(contours, key = lambda x: cv2.contourArea(x))
    contour_area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    
    hull = cv2.convexHull(cnt)
    main_img = mask[hull]
    
    cv2.drawContours(roi,[hull],-1,(0,255,0),3)
    cv2.drawContours(mask,[hull],-1,(0,255,0),300)
    mask  =cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
# Replace this with the path to your image
#image = Image.open('D:/Offline_Projects/asl_dataset/b/B1.jpeg')

# Make sure to resize all images to 224, 224 otherwise they won't fit in the array
    image = cv2.resize(mask,(224,224))  #roi.resize((224, 224))
    image_array = np.asarray(image).reshape((224,224,3))

# Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    

# Load the image into the array
    data[0] = image_array

# run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0] == max(prediction[0]):
        letter = 'B'
    elif prediction[0][1] == max(prediction[0]):
        letter = 'C'
    elif prediction[0][2] == max(prediction[0]):
        letter = 'D'
    else:
        letter = 'Q'
    cv2.putText(frame, letter, (320,55), font, 2 , (50,100,190), 3, cv2.LINE_AA)
    cv2.imshow('img',frame)
    cv2.imshow('img2',mask)
    #cv2.imshow('Main Contor',main_img)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

