import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

from detections import Detection


class Track:
    def __init__(self, detection: Detection, track_id: int):
        """
        detection: The first Detection of the track
        track_id: The unique identifier for the track
        """

        self.id = track_id
        self.hits = 1
        self.time_since_update = 0
        self.detection = detection

        # Initialize Kalman filter with 7 state variables and 4 measurement variables
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # State vector [x, y, s, r, x', y', s']
        # x,y: center position
        # s: scale (area)
        # r: aspect ratio - assumed to be constant across frames
        # x', y', s': respective velocities - unobserved but solved via the filter

        # Initialize state transition matrix (motion model)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],  # x = x + x'
                [0, 1, 0, 0, 0, 1, 0],  # y = y + y'
                [0, 0, 1, 0, 0, 0, 1],  # s = s + s'
                [0, 0, 0, 1, 0, 0, 0],  # r = r
                [0, 0, 0, 0, 1, 0, 0],  # x' = x'
                [0, 0, 0, 0, 0, 1, 0],  # y' = y'
                [0, 0, 0, 0, 0, 0, 1],  # s' = s'
            ]
        )

        # Initialize measurement matrix
        # we can only directly measure position, scale and aspect ratio
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R *= 1.0  # measurement noise
        self.kf.P *= 1000  # initial uncertainty
        self.kf.Q *= 10  # process noise

        # Initialize state from first detection
        bbox = detection.bbox
        self.kf.x[:4] = self._bbox_to_z(bbox)

    def predict(self) -> np.ndarray:
        """Advance the state vector and return the predicted bounding box."""
        self.kf.predict()
        self.time_since_update += 1
        return self._x_to_bbox(self.kf.x)

    def update(self, detection) -> None:
        """Update the state vector with observed bbox."""
        self.detection = detection
        self.hits += 1
        self.time_since_update = 0
        self.kf.update(self._bbox_to_z(detection.bbox))

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """
        Convert [x1,y1,x2,y2] normalized box to [x,y,s,r] state.
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0  # center x
        y = bbox[1] + h / 2.0  # center y
        s = w * h  # scale (area)
        r = w / h  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))

    @staticmethod
    def _x_to_bbox(x: np.ndarray) -> np.ndarray:
        """
        Convert [x,y,s,r] state to [x1,y1,x2,y2] normalized box.
        """
        center_x = x[0]
        center_y = x[1]
        area = x[2]
        ratio = x[3]

        w = np.sqrt(area * ratio)
        h = area / w

        x1 = center_x - w / 2.0
        y1 = center_y - h / 2.0
        x2 = center_x + w / 2.0
        y2 = center_y + h / 2.0

        return np.array([x1, y1, x2, y2])

    @staticmethod
    def _bbox_area(bbox: np.ndarray) -> float:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    @staticmethod
    def _iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Calculate IoU between two boxes in xyxyn (normalized) format.
        Both boxes should be in format [x1, y1, x2, y2] with values between 0 and 1.
        """
        # Find intersection box
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # No overlap
        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        bbox1_area = Track._bbox_area(bbox1)
        bbox2_area = Track._bbox_area(bbox2)
        union = bbox1_area + bbox2_area - intersection

        return intersection / union if union > 0 else 0

    def get_state(self) -> np.ndarray:
        """
        Returns the current state estimate as a bounding box [x1,y1,x2,y2].
        """
        return self._x_to_bbox(self.kf.x)


class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initialize SORT tracker

        Args:
            max_age: Maximum number of frames to keep alive a track without associated detections
            min_hits: Minimum number of associated detections before track is initialized
            iou_threshold: Minimum IOU for match
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self.frame_count = 0
        self.track_id_count = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        """
        Update tracks with new detections

        Args:
            detections: list of Detection objects

        Returns:
            list of active tracks
        """
        self.frame_count += 1

        # Get predictions from existing tracks
        for track in self.tracks:
            track.predict()

        # Match detections to tracks
        matched_indices, unmatched_detections, unmatched_tracks = (
            self._match_detections_to_tracks(detections)
        )

        # Update matched tracks
        for detection_idx, track_idx in matched_indices:
            self.tracks[track_idx].update(detections[detection_idx])

        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self.track_id_count += 1
            new_track = Track(detections[detection_idx], self.track_id_count)
            self.tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [
            track for track in self.tracks if track.time_since_update < self.max_age
        ]

        return self.tracks

    def _match_detections_to_tracks(self, detections):
        """
        Match detections to existing tracks using IoU and Hungarian algorithm

        Returns:
            matched_indices: list of tuples (detection_idx, track_idx) where each tuple matches a detection to a track
            unmatched_detections: list of detection indices that weren't matched
            unmatched_tracks: list of track indices that weren't matched
        """
        if not self.tracks:
            return [], list(range(len(detections))), []

        # Calculate IoU between each detection and predicted track location
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d, detection in enumerate(detections):
            for t, track in enumerate(self.tracks):
                predicted_bbox: np.ndarray = track.predict()
                iou_matrix[d, t] = track._iou(detection.bbox, predicted_bbox)

        # Hungarian algorithm works with costs, so we use negative IoU
        matched_detection_indices, matched_track_indices = linear_sum_assignment(
            -iou_matrix
        )

        # Filter matches with low IoU
        matches = []
        matched_det_set = set()
        matched_track_set = set()

        for d, t in zip(matched_detection_indices, matched_track_indices):
            if iou_matrix[d, t] >= self.iou_threshold:
                matches.append([d, t])
                matched_det_set.add(d)
                matched_track_set.add(t)

        unmatched_detections = [
            d for d in range(len(detections)) if d not in matched_det_set
        ]
        unmatched_tracks = [
            t for t in range(len(self.tracks)) if t not in matched_track_set
        ]

        # verify that every detection and track are accounted for
        assert len(matched_det_set) + len(unmatched_detections) == len(detections)
        assert len(matched_track_set) + len(unmatched_tracks) == len(self.tracks)

        return matches, unmatched_detections, unmatched_tracks
