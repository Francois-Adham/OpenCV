import cv2 as cv
import mediapipe as mp

class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth=False, confidence=0.5, tracking_confidence=0.5) -> None:
        
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.confidence = confidence
        self.tracking_confidence = tracking_confidence

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(self.mode, self.complexity, self.smooth, self.confidence, self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if(self.results.pose_landmarks):
            self.mp_draw.draw_landmarks(img, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        return img

if __name__ == "__main__":
    video = cv.VideoCapture(0)
    detector = PoseDetector()

    while True:
        success, img = video.read()
        
        image = detector.find_pose(img)

        cv.imshow("Image", image)
        cv.waitKey(1)