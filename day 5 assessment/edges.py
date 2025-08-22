import cv2
import matplotlib.pyplot as plt
img = cv2.imread("tree.jpeg")

blur = cv2.blur(img,(15,15))
gaussian = cv2.GaussianBlur(img,(15,15),0)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,300)
threshold, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(cv2.cvtColor(edges,cv2.COLOR_BGR2RGB))
plt.show()
