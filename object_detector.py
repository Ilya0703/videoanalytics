import os
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path, confidence = 0.5, area_rect=None):
        self.initialized = False
        if model_path is None or not os.path.exists(model_path):
            return
        self.initialized = True
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.area_rect = area_rect

    @staticmethod
    def _rect_contains(rect, pt):
        if rect is None:
            return True
        logic = rect[0] < pt[0] < rect[0] + rect[2] and rect[1] < pt[1] < rect[1] + rect[3]
        return logic

    def detect(self, frame):
        if not self.initialized:
            print("Object detector not initialized")
            return []
        results = self.model(frame, conf=self.confidence, classes=[0])
        res = []
        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if self._rect_contains(self.area_rect, ((x1+x2)/2, y1)):
                    res.append([np.array([x1, y1, x2-x1, y2-y1]), confidence, 0])
        return res
