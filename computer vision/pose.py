import cv2 as cv
import mediapipe as mp

# class PoseDetector:

video = cv.VideoCapture(0)

mp_pose = mp.solutions.pose
poses = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = video.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = poses.process(imgRGB)
    if(results.pose_landmarks):
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv.imshow("Image", img)
    cv.waitKey(1)