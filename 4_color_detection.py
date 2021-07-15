import cv2 as cv
import numpy as np
from numpy.lib.function_base import disp

image = cv.imread("images/shapes.png")

# blue mask
blue_mask = np.array(np.where((image[:, :, 0] > 200), 255, 0), dtype=np.uint8)

blue_masked = np.copy(image)
blue_masked[:, :, 0] = blue_mask & image[:, :, 0]
blue_masked[:, :, 1] = blue_mask & image[:, :, 1]
blue_masked[:, :, 2] = blue_mask & image[:, :, 2]


# red mask
red_mask = np.array(np.where(((image[:, :, 2] > 250) & (image[:, :, 1] < 100)), 255, 0), dtype=np.uint8)

red_masked = np.copy(image)
red_masked[:, :, 0] = red_mask & image[:, :, 0]
red_masked[:, :, 1] = red_mask & image[:, :, 1]
red_masked[:, :, 2] = red_mask & image[:, :, 2]



# green mask
green_mask = np.array(np.where(((image[:, :, 2] < 100) & (image[:, :, 1] > 250)), 255, 0), dtype=np.uint8)
green_masked = np.copy(image)
green_masked[:, :, 0] = green_mask & image[:, :, 0]
green_masked[:, :, 1] = green_mask & image[:, :, 1]
green_masked[:, :, 2] = green_mask & image[:, :, 2]



dis = np.hstack([blue_masked, red_masked, green_masked])

cv.imshow(" ", dis)
cv.waitKey(0)

# for k in range(3):
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#                 print(i, j , k, image[i][j][k])
