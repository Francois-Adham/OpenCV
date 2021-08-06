import cv2 as cv
import mediapipe as mp
import time

video = cv.VideoCapture(0)
mp_hand =  mp.solutions.hands

hands = mp_hand.Hands()

mp_draw = mp.solutions.drawing_utils


while True:
    success, img = video.read()
    width, height, channels = img.shape

    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if(results.multi_hand_landmarks):
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand, mp_hand.HAND_CONNECTIONS)
            for id, loc in enumerate(hand.landmark):
                x, y = int(loc.x * height), int(loc.y * width)
                if( id % 4 == 0):
                    cv.circle(img, (x, y), 10, [0, 0, 0],-1)
                elif( id % 4 == 1):
                    cv.circle(img, (x, y), 10, [255, 0, 0], -1)
                elif( id % 4 == 2):
                    cv.circle(img, (x, y), 10, [0, 255, 0], -1)
                elif( id % 4 == 3):
                    cv.circle(img, (x, y), 10, [0, 0, 255], -1)
    cv.imshow("Image", img)
    cv.waitKey(1)