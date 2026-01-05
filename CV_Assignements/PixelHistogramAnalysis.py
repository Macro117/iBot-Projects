import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#load grayscale image
img = cv.imread(R"D:\visual studio stuff\Practice\Python\CVPractice\sampleImage.jpg",cv.IMREAD_GRAYSCALE)

#histogram
hist = cv.calcHist([img], [0], None, [256], [0, 256])

#statistics
mean_val = np.mean(img)
median_val = np.median(img)
std_val = np.std(img)

#displaying
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img, cmap="gray")
plt.title("Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.plot(hist)
plt.title("Histogram")
plt.xlabel("Intensity")
plt.ylabel("Pixel Count")

plt.text(170, max(hist)*0.7,f"Mean: {mean_val:.2f}\n"f"Median: {median_val:.2f}\n"f"Std: {std_val:.2f}",bbox=dict(facecolor="white", alpha=0.8))
plt.show()
