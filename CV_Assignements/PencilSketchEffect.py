import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

#load image
def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Image path invalid")
    img = cv.imread(path)
    if img is None:
        raise ValueError("Failed")
    return img

#pencil sketch effect
def pencil_sketch(img):
    #convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #invert grayscale
    gray_inv = 255 - gray

    #gaussian blur
    gray_inv_blur = cv.GaussianBlur(gray_inv, (21, 21), 0)

    #invert blurred image
    inv_blur = 255 - gray_inv_blur

    #divide and scale
    sketch = cv.divide(gray, inv_blur, scale=256)

    #clip values
    sketch = np.clip(sketch, 0, 255).astype(np.uint8)

    return gray, sketch

#display
def display_side_by_side(original_gray, sketch):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_gray, cmap="gray")
    plt.title("Original (Grayscale)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(sketch, cmap="gray")
    plt.title("Pencil Sketch")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

#save img
def save_sketch(sketch, output_path):
    cv.imwrite(output_path, sketch)

#main
input_path = R"D:\visual studio stuff\Practice\Python\CVPractice\sampleImage.jpg"
output_path = R"D:\visual studio stuff\Practice\Python\CVPractice\pencil_sketch.jpg"

img = load_image(input_path)

gray, sketch = pencil_sketch(img)

display_side_by_side(gray, sketch)

save_sketch(sketch, output_path)
