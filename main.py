import cv2
import numpy as np
import pyrealsense2 as rs

from danger_zones import DangerZones
from frame_drawer import FrameDrawer
from map_drawer import MapDrawer
from motion_detector import MotionDetector
from object_detector import ObjectDetector
from tracker import Tracker

if __name__ == "__main__":
    cam = cv2.VideoCapture("datasets/2.mp4")
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    motionDetector = MotionDetector(200)
    objectDetector = ObjectDetector("models/yolo11n.pt", confidence=0.2)
    tracker = Tracker()
    dangerZones = DangerZones([
    np.array([[593, 582], [690, 701], [804, 572], [702, 477]], np.int32)], (frame_width, frame_height))
    danger_zones_mask = dangerZones.get_danger_zones()
    frameDrawer = FrameDrawer(danger_zones_mask)
    mapDrawer = MapDrawer((frame_width, frame_height), danger_zones_mask)
    # pipeline = rs.pipeline()
    # config = rs.config()
    #
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    #
    # pipeline.start(config)
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        # frames = pipeline.wait_for_frames()
        # frame = frames.get_color_frame()
        # if not frame:
        #     continue
        # frame = np.asanyarray(frame.get_data())

        trackerResults = None
        violators = None
        trajectories = None
        if motionDetector.detect_motion(frame):
            objectDetectorResults = objectDetector.detect(frame)
            trackerResults, trajectories = tracker.track(objectDetectorResults, frame)
            violators = dangerZones.check(trackerResults)
        frameDrawer.draw_results(frame.copy(), trackerResults)
        mapDrawer.draw_results(trajectories, violators)
    #cam.release()
    #cv2.destroyAllWindows()