import cv2
import numpy as np


def get_color(track_id):
    hash_value = hash(track_id) % 256
    r = (hash_value * 37) % 256
    g = (hash_value * 79) % 256
    b = (hash_value * 113) % 256
    return (int(b), int(g), int(r))

class FrameDrawer:
    def __init__(self, danger_zones_mask):
        self.danger_zones_mask = danger_zones_mask

    def draw_results(self, frame, tracker_results):
        red_image = np.zeros_like(frame)
        red_image[:] = (0, 0, 255)
        alpha = 0.5  # 50% прозрачности
        red_mask = cv2.merge([self.danger_zones_mask, self.danger_zones_mask, self.danger_zones_mask])
        blended = cv2.addWeighted(frame, 1, red_image, alpha, 0)
        frame = np.where(red_mask == 255, blended, frame)

        if tracker_results:
            for res in tracker_results:
                track_id, bbox = res
                if bbox is None:
                    continue
                bbox = bbox.astype(int).tolist()
                cv2.putText(frame, str(track_id), (bbox[0] + 5, bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 3, cv2.LINE_AA)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1] + bbox[3]), get_color(track_id), 3)
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
        return frame