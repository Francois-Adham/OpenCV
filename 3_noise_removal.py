import cv2 as cv
import numpy as np
import random

from numpy.lib.function_base import disp

image = cv.imread("images/books.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# ==========================================
# ============== ADDING NOISE ==============
# ==========================================

s_vs_p = 0.5
amount = 0.04
noise = np.copy(gray)
# Salt mode
num_salt = np.ceil(amount * image.size * s_vs_p)
coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in gray.shape]
noise[tuple(coords)] = 255

# Pepper mode
num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in gray.shape]
noise[tuple(coords)] = 0

# print(gray.shape, noise.shape)
disp = np.hstack([gray, noise])
cv.imshow("", disp)
cv.waitKey(0)

des = cv.fastNlMeansDenoising(noise, None, 30, 21, 35)

cv.imshow("l", des)
cv.waitKey(0)
