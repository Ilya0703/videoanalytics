import cv2
import numpy as np
from fontTools.ttLib.scaleUpem import visit

from frame_drawer import get_color

class MapDrawer:
    def __init__(self, image_shape, danger_zones_mask, scale = 0.5):
        self.image_shape = image_shape
        self.frame = np.zeros((int(image_shape[1]), int(image_shape[0]), 3), dtype=np.uint8)
        self.frame[:, :, :] = 255
        self.list_frame = np.zeros((int(image_shape[1] * scale), int(image_shape[0] * scale), 3), dtype=np.uint8)
        self.scale = scale
        self.danger_zones_mask = danger_zones_mask
        self.frame[self.danger_zones_mask == 255] = (0, 0, 255)
        self.frame = cv2.resize(self.frame, (int(image_shape[0] * scale), int(image_shape[1] * scale)))

    def draw_results(self, trajectories, violators=None):
        frame = self.frame.copy()
        if trajectories is None:
            trajectories = {}

        # Рисуем треки на основном кадре
        for track_id, positions in trajectories.items():
            pts = np.array(positions)
            pts *= self.scale
            pts = pts.reshape((-1, 1, 2))
            pts = pts.astype(np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=get_color(track_id), thickness=2)
            last_position = (int(positions[-1][0] * self.scale), int(positions[-1][1] * self.scale))
            cv2.circle(frame, last_position, 10, color=get_color(track_id), thickness=-1)
            cv2.putText(frame, str(track_id), (last_position[0] + 15, last_position[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

        list_frame = self.list_frame.copy()
        list_height, list_width = list_frame.shape[:2]

        def draw_list(title, items, start_y, column_width, max_height):
            cv2.putText(list_frame, title, (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            x_offset = 10
            y_offset = start_y + 30
            column = 0

            for i, item in enumerate(items):
                if y_offset + 30 > max_height:  # Если текущий столбец заполнен, переходим на следующий
                    column += 1
                    y_offset = start_y + 30
                    x_offset = 10 + column * column_width

                color = get_color(item)
                cv2.circle(list_frame, (x_offset + 20, y_offset - 5), 10, color, -1)
                cv2.putText(list_frame, str(item), (x_offset + 40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                y_offset += 30

        # Отрисовка списка Violators
        if violators is None:
            violators = []
        draw_list("Violators:", violators, 30, 150, list_height)

        # Отрисовка списка All Tracks
        draw_list("All Tracks:", list(trajectories.keys()), 180, 150, list_height)

        # Объединяем кадры
        combined_frame = np.hstack((frame, list_frame))
        cv2.imshow("Map", combined_frame)
        cv2.waitKey(1)
