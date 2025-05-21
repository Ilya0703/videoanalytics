import time
from multiprocessing import Pipe, Process

import cv2
import numpy as np
import pyrealsense2 as rs

import web_server
from config_reader import ConfigReader
from danger_zones import DangerZones
from frame_drawer import FrameDrawer
from map_drawer import MapDrawer
from motion_detector import MotionDetector
from object_detector import ObjectDetector
from tracker import Tracker

def processing_task(camera_id, camera_parameters, configReader, sender):
    cam = cv2.VideoCapture(camera_parameters['rtsp'])
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    motionDetector = MotionDetector(configReader.motion_detector_min_motion_area())
    objectDetector = ObjectDetector(configReader.object_detector_model_path(),
                                    configReader.object_detector_confidence(),
                                    area_rect=camera_parameters["object_detector_area_rect"])
    tracker = Tracker()

    dangerZones = DangerZones(
        [np.array(camera_parameters['danger_zones'], np.int32)],
        (frame_width, frame_height)
    )
    danger_zones_mask = dangerZones.get_danger_zones()

    frameDrawer = FrameDrawer(danger_zones_mask)
    mapDrawer = MapDrawer((frame_width, frame_height), danger_zones_mask)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        trackerResults = None
        violators = None
        trajectories = None

        if motionDetector.detect_motion(frame):
            objectDetectorResults = objectDetector.detect(frame)
            trackerResults, trajectories = tracker.track(objectDetectorResults, frame)
            violators = dangerZones.check(trackerResults)

        processed_frame = frameDrawer.draw_results(frame.copy(), trackerResults)
        map = mapDrawer.draw_results(trajectories, violators)

        sender.send([camera_id, processed_frame, map])
        try:
            sender.send([camera_id, processed_frame, map])
        except (BrokenPipeError, ConnectionResetError):
            break

    cam.release()
    sender.close()

if __name__ == "__main__":
    configReader = ConfigReader("config.yaml")
    sender, receiver = Pipe()
    framesReseiversDict = {}
    processes = []
    for camera_id, params in configReader.cameras_dict().items():
        sender, receiver = Pipe(duplex=True)

        process = Process(
            target=processing_task,
            args=(camera_id, params, configReader, sender)
        )

        framesReseiversDict[camera_id] = receiver
        processes.append(process)
        process.start()
    server = web_server.WebServer(framesReseiversDict, configReader.server_host())
    server.Start()

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
    for id, receiver in framesReseiversDict.items():
        receiver.close()
    server.Close()