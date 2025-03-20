import cv2
import numpy as np


class MotionDetector:
    def __init__(self, min_motion_area):
        self.substractor = cv2.createBackgroundSubtractorKNN(history=100)
        self.min_motion_area = min_motion_area

    def detect_motion(self, frame):
        fg_mask = self.substractor.apply(frame)

        fg_mask_erode = cv2.erode(fg_mask, np.ones(7, np.uint8))

        motion_area_erode = cv2.findNonZero(fg_mask_erode)

        if motion_area_erode is not None:
            xe, ye, we, he = cv2.boundingRect(motion_area_erode)
            if we * he > self.min_motion_area:
                return True
        return False