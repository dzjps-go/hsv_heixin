import cv2
import numpy as np

img_bgr = cv2.imread('test.jpg')
img_hsv = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2HSV)
print(img_hsv.shape)
