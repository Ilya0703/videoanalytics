from deep_sort_realtime.deepsort_tracker import DeepSort


class Tracker:
    def __init__(self):
        self.tracker = DeepSort()
        self.trajectory_max_len = 100
        self.track_memory = 10
        self.trajectories = {}
        self.old_tracks = {}

    def track(self, object_detector_results, frame):
        res = []
        tracks_ids = []
        results = self.tracker.update_tracks(object_detector_results, frame=frame)
        for result in results:
            if result.original_ltwh is None:
                continue
            res.append([result.track_id, result.original_ltwh])
            tracks_ids.append(result.track_id)
        for track_id, bbox in res:
            if track_id in self.trajectories.keys():
                if len(self.trajectories[track_id]) >= self.trajectory_max_len:
                    self.trajectories[track_id].pop(0)
                self.trajectories[track_id].append([bbox[0] + bbox[2]/2, bbox[1] + bbox[3]])
            else:
                self.trajectories[track_id] = [[bbox[0] + bbox[2]/2, bbox[1] + bbox[3]]]
        new_trajectories = self.trajectories.copy()
        for track_id in self.trajectories.keys():
            if track_id not in tracks_ids:
                if track_id not in self.old_tracks:
                    self.old_tracks[track_id] = 1
                else:
                    if self.old_tracks[track_id] > self.track_memory:
                        self.old_tracks.pop(track_id, None)
                        new_trajectories.pop(track_id, None)
                    else:
                        self.old_tracks[track_id] += 1
        self.trajectories = new_trajectories
        return res, self.trajectories
