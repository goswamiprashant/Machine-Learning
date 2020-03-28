# how to extarct rgb  component from rgb image
import cv2
import numpy as np
img=cv2.imread("leena.jpg")
B,G,R=cv2.split(img) # not working
zeros=np.zeros(img.shape[:2],dtype="uint8")# dtype:usigned int
cv2.imshow("Red",cv2.merge([zeros,zeros,R]))
cv2.imshow("Green",cv2.merge([zeros,G,zeros]))
cv2.imshow("Black",cv2.merge([B,zeros,zeros]))
cv2.waitKey(0)
cv2.destroyAllWindows()