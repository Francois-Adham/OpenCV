# import OpenCV
import cv2 as cv

# read image 
image = cv.imread("images/books.jpg")

print(image.shape)

# show image
cv.imshow("books", image)
cv.waitKey(0)


# gray representation
gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("gray books", gray_image)
cv.waitKey(0)

# HSV representation
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
cv.imshow("hsv books", hsv_image)
cv.waitKey(0)

# destroy all opened windows
cv.destroyAllWindows()