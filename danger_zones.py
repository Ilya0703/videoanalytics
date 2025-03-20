import cv2
import numpy as np


class DangerZones:
    def __init__(self, polygons, image_shape):
        self.danger_zones = polygons
        self.binary_image = np.zeros((image_shape[1], image_shape[0]), dtype=np.uint8)
        self.image_shape = image_shape
        for polygon in polygons:
            cv2.fillPoly(self.binary_image, [polygon], color=255)

    def check(self, tracker_results):
        violators = []
        for track_id, bbox in tracker_results:
            point_x = max(0, min(int(bbox[0] + bbox[2] / 2), self.image_shape[0] - 1))
            point_y = max(0, min(int(bbox[1] + bbox[3]), self.image_shape[1] - 1))

            if self.binary_image[point_y, point_x] == 255:
                violators.append(track_id)
        return violators

    def get_danger_zones(self):
        return self.binary_image