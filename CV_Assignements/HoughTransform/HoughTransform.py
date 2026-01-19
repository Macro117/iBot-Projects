import cv2 as cv
import matplotlib.pyplot as plt
import os
import numpy as np

def preprocess(img_path):
    
    if not os.path.exists(img_path):
        print("File not found")
        return None
    
    img = cv.imread(img_path)
    if img is None:
        print("Could not read image")
        return None
    
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, (9,9), 0)
    img_gray = cv.equalizeHist(img_gray)
    
    return img, img_gray


def find_circles(
    img_gray,
    dp=1.2,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=0,
):
    
    circles = cv.HoughCircles(
        img_gray,
        cv.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    
    if circles is not None:
        circles = np.round(circles[0,:]).astype("int")
    
    return circles


def display_circles(img, circles, save_path=None):
    img_copy = img.copy()
    
    if circles is not None:
        for idx, (x, y, r) in enumerate(circles):
            cv.circle(img_copy, (x, y), r, (0, 255, 0), 2)
            cv.circle(img_copy, (x, y), 2, (0, 0, 255), 3)
            cv.putText(
                img_copy,
                f"ID {idx} r={r}",
                (x - 30, y - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )
        
    img_final = np.hstack((img, img_copy))
    plt.figure(figsize=(12, 6))
    plt.imshow(cv.cvtColor(img_final, cv.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
    
    if save_path:
        ok = cv.imwrite(save_path, img_final)
        print("Saved:", ok, save_path)
            

def calculate_statistics(circles):
    
    if circles is None or len(circles) == 0:
        return{
            "count": 0,
            "min_radius": None,
            "max_radius": None,
            "avg_radius": None,
            "circles": [],
        }
        
    radii = circles[:,2]
    stats = {
        "count": len(circles),
        "min_radius": int(np.min(radii)),
        "max_radius": int(np.max(radii)),
        "avg_radius": float(np.mean(radii)),
        "circles": circles.tolist(),
    } 
    
    return stats


def save_statistics(stats, txt_file_path):
    with open(txt_file_path, "w") as f:
        f.write(f"Total Circles: {stats['count']}\n")
        f.write(f"Min Radius: {stats['min_radius']}\n")
        f.write(f"Max Radius: {stats['max_radius']}\n")
        f.write(f"Average Radius: {stats['avg_radius']}\n\n")
        
        for idx, (x, y, r) in enumerate(stats["circles"]):
            f.write(f"ID {idx}: Center=({x}, {y}), Radius={r}\n")


# main

img_path = R"D:\visual studio stuff\iBot_DC\CV_Assignements\HoughTransform\test_images\test_image_3.png"
final_img_path = R"D:\visual studio stuff\iBot_DC\CV_Assignements\HoughTransform\transformed_images\transformed_image_3.png"
stats_path = R"D:\visual studio stuff\iBot_DC\CV_Assignements\HoughTransform\stats\stats_3.txt"

os.makedirs(os.path.dirname(final_img_path), exist_ok=True)
os.makedirs(os.path.dirname(stats_path), exist_ok=True)

img, img_gray = preprocess(img_path)
if img is None:
    print("Image could not be preprocessed")
    exit()
    
circles = find_circles(img_gray)
stats = calculate_statistics(circles)

print(stats)

display_circles(img, circles, save_path=final_img_path)
save_statistics(stats, stats_path)
