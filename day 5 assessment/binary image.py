import cv2
import matplotlib.pyplot as plt
img = cv2.imread("tree.jpeg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
threshold, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
hist = cv2.calcHist(gray,[0],None,[256],[0,256])

# plt.imshow(cv2.cvtColor(ist,cv2.COLOR_BGR2RGB))
plt.plot(hist)
plt.show()