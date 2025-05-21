import ast

import yaml
import numpy as np
class ConfigReader:
    def __init__(self, config_path):
        self.config_path = config_path

    def object_detector_model_path(self):
        with open(self.config_path) as f:
            depth_map_parameters = yaml.safe_load(f)["object_detector_parameters"]
        if "model_file" not in depth_map_parameters.keys():
            return ""
        return depth_map_parameters['model_file']

    def object_detector_confidence(self):
        with open(self.config_path) as f:
            depth_map_parameters = yaml.safe_load(f)["object_detector_parameters"]
        if "confidence" not in depth_map_parameters.keys():
            return ""
        return depth_map_parameters['confidence']

    def motion_detector_min_motion_area(self):
        with open(self.config_path) as f:
            depth_map_parameters = yaml.safe_load(f)["motion_detector_parameters"]
        if "min_motion_area" not in depth_map_parameters.keys():
            return ""
        return depth_map_parameters['min_motion_area']

    def server_host(self):
        with open(self.config_path) as f:
            depth_map_parameters = yaml.safe_load(f)["server_parameters"]
        if "host" not in depth_map_parameters.keys():
            return ""
        return depth_map_parameters['host']

    def cameras_dict(self):
        with open(self.config_path) as f:
            depth_map_parameters = yaml.safe_load(f)["cameras"]
        return depth_map_parameters
