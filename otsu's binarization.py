import numpy as np
import os
import cv2

input_path = (r"_your_input_path_")
output_path = (r"_your_output_path_")

def Grayscale(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def otsu_threshold(image):
    size = image.shape
    histogram = np.zeros(256)
    for x in range(size[0]):
        for y in range(size[1]):
            histogram[image[x, y]] += 1
    
    histogram = np.divide(histogram, size[0] * size[1])
    maximum = -1000
    for t in range(256):
        q1 = np.sum(histogram[:t])
        q2 = np.sum(histogram[t:])
        u1 = np.dot(np.array([poz for poz in range(t)]), histogram[:t]) / q1
        u2 = np.dot(np.array([poz for poz in range(t, 256)]), histogram[t:]) / q2
        val = q1 * (1 - q1) * np.power(u1 - u2, 2)
        if val > maximum:
            maximum = val
            t_min = t

    return t_min


def apply_otsu(image):
    image = Grayscale(image)
    threshold = otsu_threshold(image)
    size = image.shape
    for x in range(size[0]):
        for y in range(size[1]):
            if(image[x, y] > threshold):
                image[x, y] = 0

    return image

if __name__ == "__main__":
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            image = cv2.imread(os.path.join(root, filename))
            image = apply_otsu(image)
            cv2.imwrite(os.path.join(output_path, filename), image)
