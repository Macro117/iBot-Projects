import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


#load image
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError("image path does not exist")
    img = cv.imread(path)
    if img is None:
        raise ValueError("failed to load image")
    return img

#image processing
def process_image(img):
    #blur
    img_blur = cv.GaussianBlur(img, (7, 7), 0)

    #grayscale
    img_gray_blur = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)

    #canny edge detection
    edges = cv.Canny(img_gray_blur, 100, 200)

    #binary thresholding
    _, binary = cv.threshold(img_gray_blur, 127, 255, cv.THRESH_BINARY)

    return img_blur, edges, binary

#display results
def display_results(img, img_blur, edges, binary):
    plt.figure(figsize=(12, 12))

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_blur_rgb = cv.cvtColor(img_blur, cv.COLOR_BGR2RGB)

    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(img_blur_rgb)
    plt.title("Blurred")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(edges, cmap="gray")
    plt.title("Edges")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(binary, cmap="gray")
    plt.title("Threshold")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

#main
path = R"D:\visual studio stuff\iBot_DC\CV_Assignements\sampleImage.jpg"

img = load_image(path)

img_blur, edges, binary = process_image(img)

display_results(img, img_blur, edges, binary)
