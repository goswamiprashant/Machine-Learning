# produce binary(black and white) image from rgb
import cv2
img=cv2.imread("leena.jfif",0)
cv2.imshow("Test",img)
cv2.waitKey(0)
binary_image=cv2.threshold(img,125,255,cv2.THRESH_BINARY)
cv2.imshow("Binary_image",binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()