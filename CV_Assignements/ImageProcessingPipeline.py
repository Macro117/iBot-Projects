import cv2 as cv
import matplotlib.pyplot as plt

path = R"D:\visual studio stuff\iBot_DC\CV_Assignements\sampleImage.jpg"
img = cv.imread(path)

#blur
img_blur = cv.GaussianBlur(img, (7,7), 0)

#canny edge detection
img_gray_blur = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
edges = cv.Canny(img_gray_blur, threshold1=100, threshold2=200)

#Binary Thresholding
_, binary = cv.threshold(img_gray_blur, 127, 255, cv.THRESH_BINARY)

#Displaying
plt.figure(figsize= (12,12))

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title('Original')
plt.axis("off")

img_blur_rgb = cv.cvtColor(img_blur, cv.COLOR_BGR2RGB)
plt.subplot(2,2,2)
plt.imshow(img_blur_rgb)
plt.title('Blurred')
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(edges, cmap="gray")
plt.title('Edges')
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(binary, cmap="gray")
plt.title('Threshold')
plt.axis("off")

plt.show()