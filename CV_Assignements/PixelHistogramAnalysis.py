import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


#load grayscale image
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError("image path does not exist")
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("failed to load image")
    return img

#histogram and statistics
def analyze_image(img):
    #histogram
    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    #statistics
    mean_val = np.mean(img)
    median_val = np.median(img)
    std_val = np.std(img)

    return hist, mean_val, median_val, std_val

#display image and histogram
def display_results(img, hist, mean_val, median_val, std_val):
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

    plt.text(
        80, max(hist)*0.8,
        f"Mean: {mean_val:.2f}\n"
        f"Median: {median_val:.2f}\n"
        f"Std: {std_val:.2f}"
    )

    plt.tight_layout()
    plt.show()

#main
path = R"D:\visual studio stuff\iBot_DC\CV_Assignements\sampleImage.jpg"

img = load_image(path)

hist, mean_val, median_val, std_val = analyze_image(img)

display_results(img, hist, mean_val, median_val, std_val)
