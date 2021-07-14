# import OpenCV
import cv2 as cv

# read image 
image = cv.imread("images/books.jpg")

print(image.shape)

# show image
cv.imshow("books", image)

# wait to destroy image
cv.waitKey(0)

# destroy all opened windows
cv.destroyAllWindows()