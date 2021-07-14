import cv2 as cv
import numpy as np

image = cv.imread("images/books.jpg")

gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

canny = cv.Canny(gray_image, 150, 200)

kernel = np.ones((5,5))
dialation = cv.dilate(canny, kernel)

errosion = cv.erode(dialation, kernel)


display = np.hstack([gray_image, canny])
display2 = np.hstack([dialation, errosion])

cv.imshow("original + edges", display)
cv.waitKey(0)

cv.imshow("dialation + errosion", display2)
cv.waitKey(0)

cv.destroyAllWindows()