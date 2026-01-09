from collections import defaultdict

class TrajectoryStore:

    def __init__(self):
        self.trajectories = defaultdict(list)

    def add(self,track):
        self.trajectories[track["track_id"]].append(track)

    def get_all(self):
        return self.trajectories