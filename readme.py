# import cv2 as cv
# import numpy as np

# img = cv.imread('C:/Users/Ankur/Downloads/harshita.jpg')
# cv.imshow('Catty', img) 

# gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# #simple thresholding

# threshold,thresh=cv.threshold(gray,150,255,cv.THRESH_BINARY)
# cv.imshow('Simple Threshold', thresh)

# Adp =cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,11,3)
# cv.imshow('Adaptive Thresh',Adp)

# haar_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
# faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

# for(x, y, w, h) in faces_rect:
#     cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

# cv.imshow('Detected Faces', img)

# average = cv.blur(img, (500, 500))
# cv.imshow("Average", average)

# blank=np.zeros(img.shape[:2], dtype='uint8')   
# cv.imshow('Blank Image', blank)

# mask=cv.circle(blank,(img.shape[1]//2,img.shape[0]//2),100,255,-1)
# cv.imshow('Mask',mask)

# masked=cv.bitwise_and(img,img,mask=mask)
# cv.imshow('Masked Image',masked)

# cv.waitKey(0)
