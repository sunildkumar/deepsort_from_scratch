import matplotlib.pyplot as plt
import numpy as np

from detections import Detection
from sort import Sort, Track


def test_track_linear_motion():
    # Create a sequence of detections simulating linear motion
    # Starting at (0.2, 0.2) and moving to (0.8, 0.8) in 5 steps
    steps = 10
    start_pos = np.array([0.2, 0.2])
    end_pos = np.array([0.8, 0.8])
    velocity = (end_pos - start_pos) / steps

    # Fixed box size for simplicity
    width, height = 0.1, 0.1

    # Initialize first detection and track
    initial_bbox = np.array(
        [
            start_pos[0] - width / 2,  # x1
            start_pos[1] - height / 2,  # y1
            start_pos[0] + width / 2,  # x2
            start_pos[1] + height / 2,  # y2
        ]
    )

    first_detection = Detection(
        bbox=initial_bbox, class_id=0, class_name="test", confidence=1.0
    )

    track = Track(first_detection, track_id=1)

    # Storage for predictions and actual positions
    predictions = []
    actual_positions = []

    # Simulate motion and track
    for i in range(1, steps):
        # Current actual position
        current_pos = start_pos + i * velocity

        # Create detection bbox
        current_bbox = np.array(
            [
                current_pos[0] - width / 2,
                current_pos[1] - height / 2,
                current_pos[0] + width / 2,
                current_pos[1] + height / 2,
            ]
        )

        # Get prediction before update
        predicted_bbox = track.predict()
        predictions.append(predicted_bbox)
        actual_positions.append(current_bbox)

        # Update track with new detection
        detection = Detection(
            bbox=current_bbox, class_id=0, class_name="test", confidence=1.0
        )
        track.update(detection)

        # Verify basic track properties
        assert track.hits == i + 1
        assert track.time_since_update == 0

        # Calculate center points
        predicted_center = (predicted_bbox[:2] + predicted_bbox[2:]) / 2
        actual_center = (current_bbox[:2] + current_bbox[2:]) / 2

        # Only check prediction error after the first few steps
        # to allow the filter to stabilize
        error = np.linalg.norm(predicted_center - actual_center)
        if i >= 3:  # Start checking after 3 updates
            assert error < 0.2, f"Prediction error too large: {error}"

    # After the tracking loop, add visualization
    # Convert predictions and actual positions to center points
    pred_centers = np.array(
        [((p[0] + p[2]) / 2, (p[1] + p[3]) / 2) for p in predictions]
    )
    actual_centers = np.array(
        [((a[0] + a[2]) / 2, (a[1] + a[3]) / 2) for a in actual_positions]
    )

    # Include the initial position
    initial_center = (initial_bbox[:2] + initial_bbox[2:]) / 2
    all_actual_centers = np.vstack([initial_center, actual_centers])

    plt.figure(figsize=(10, 10))
    plt.plot(
        all_actual_centers[:, 0],
        all_actual_centers[:, 1],
        "b.-",
        label="Actual",
        markersize=10,
    )
    plt.plot(
        pred_centers[:, 0], pred_centers[:, 1], "r.--", label="Predicted", markersize=10
    )

    # Plot boxes at each position
    for i, (actual, pred) in enumerate(zip(actual_positions, predictions)):
        # Actual box in blue
        plt.plot(
            [actual[0], actual[2], actual[2], actual[0], actual[0]],
            [actual[1], actual[1], actual[3], actual[3], actual[1]],
            "b-",
            alpha=0.3,
        )
        # Predicted box in red
        plt.plot(
            [pred[0], pred[2], pred[2], pred[0], pred[0]],
            [pred[1], pred[1], pred[3], pred[3], pred[1]],
            "r--",
            alpha=0.3,
        )

    plt.grid(True)
    plt.legend()
    plt.title("Object Tracking: Actual vs Predicted Positions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis("equal")
    from imgcat import imgcat

    imgcat(plt.gcf())


def test_track_matching():
    # Initialize tracker
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

    # Frame 1: Create initial track
    initial_bbox = np.array([0.4, 0.4, 0.6, 0.6])  # box in center
    initial_detection = Detection(
        bbox=initial_bbox, class_id=0, class_name="test", confidence=1.0
    )
    tracks = tracker.update([initial_detection])
    assert len(tracks) == 1, "Should create one track"

    # Frame 2: Two detections - one overlapping, one not
    overlapping_bbox = np.array([0.45, 0.45, 0.65, 0.65])  # shifted slightly
    non_overlapping_bbox = np.array([0.1, 0.1, 0.2, 0.2])  # completely separate

    detections = [
        Detection(bbox=overlapping_bbox, class_id=0, class_name="test", confidence=1.0),
        Detection(
            bbox=non_overlapping_bbox, class_id=0, class_name="test", confidence=1.0
        ),
    ]

    matched_idx, unmatched_detections, unmatched_tracks = (
        tracker._match_detections_to_tracks(detections)
    )

    assert len(matched_idx) == 1, "Should match one detection"
    assert matched_idx[0][0] == 0, "Should match the overlapping detection"
    assert len(unmatched_detections) == 1, "Should have one unmatched detection"
    assert (
        unmatched_detections[0] == 1
    ), "The non-overlapping detection should be unmatched"


def test_complex_track_matching():
    # Initialize tracker
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

    # Frame 1: Create three initial tracks
    track1_bbox = np.array([0.2, 0.2, 0.4, 0.4])  # top left
    track2_bbox = np.array([0.6, 0.6, 0.8, 0.8])  # bottom right
    track3_bbox = np.array([0.7, 0.1, 0.9, 0.3])  # top right - will have no matches

    initial_detections = [
        Detection(track1_bbox, class_id=0, class_name="test", confidence=1.0),
        Detection(track2_bbox, class_id=0, class_name="test", confidence=1.0),
        Detection(track3_bbox, class_id=0, class_name="test", confidence=1.0),
    ]

    tracks = tracker.update(initial_detections)
    assert len(tracks) == 3, "Should create three tracks"

    # Frame 2: Five detections with different scenarios
    det1 = np.array([0.25, 0.25, 0.45, 0.45])  # overlaps with track1
    det2 = np.array([0.65, 0.65, 0.85, 0.85])  # overlaps with track2
    det3 = np.array([0.1, 0.1, 0.2, 0.2])  # no overlap
    det4 = np.array([0.4, 0.4, 0.6, 0.6])  # between both tracks
    det5 = np.array([0.15, 0.15, 0.35, 0.35])  # also overlaps track1

    detections = [
        Detection(det1, class_id=0, class_name="test", confidence=1.0),
        Detection(det2, class_id=0, class_name="test", confidence=1.0),
        Detection(det3, class_id=0, class_name="test", confidence=1.0),
        Detection(det4, class_id=0, class_name="test", confidence=1.0),
        Detection(det5, class_id=0, class_name="test", confidence=1.0),
    ]

    matched_idx, unmatched_detections, unmatched_tracks = (
        tracker._match_detections_to_tracks(detections)
    )

    # Verify basic matching properties
    assert len(matched_idx) == 2, "Should have exactly two matches"
    assert len(unmatched_detections) == 3, "Should have three unmatched detections"
    assert len(unmatched_tracks) == 1, "Should have one unmatched track (track3)"

    # Verify specific matches
    # Convert matches to a more easily testable format
    matches_dict = {d: t for d, t in matched_idx}

    # det1 or det5 should match with track1 (index 0)
    assert (
        0 in matches_dict.values() or 4 in matches_dict.values()
    ), "Track1 should match with either det1 or det5"
    # det2 should match with track2 (index 1)
    assert 1 in matches_dict.values(), "Track2 should match with det2"
    # track3 (index 2) should be unmatched
    assert 2 in unmatched_tracks, "Track3 should be unmatched"

    # det3 and det4 should be unmatched
    assert 2 in unmatched_detections, "det3 should be unmatched"
    assert 3 in unmatched_detections, "det4 should be unmatched"


def test_linear_motion_with_noise():
    # Create true trajectory - moving from bottom left to top right
    steps = 10
    start_pos = np.array([0.05, 0.05])
    end_pos = np.array([0.95, 0.95])
    velocity = (end_pos - start_pos) / steps
    true_centers = np.array([start_pos + i * velocity for i in range(steps)])

    width, height = 0.05, 0.05

    # Generate true boxes and noisy detections
    true_boxes = []
    noisy_boxes = []
    noise_level = 0.02  # Adjust this to control detection noise

    for center_x, center_y in true_centers:
        # Create true box
        true_box = np.array(
            [
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2,
            ]
        )
        true_boxes.append(true_box)

        # Add noise while preserving box validity
        noise = np.random.normal(0, noise_level, 4)
        noisy_box = true_box.copy()
        # Apply noise to left/top coordinates
        noisy_box[0] += noise[0]
        noisy_box[1] += noise[1]
        # Ensure right/bottom coordinates stay greater than left/top
        noisy_box[2] = max(noisy_box[0] + 0.02, noisy_box[2] + noise[2])
        noisy_box[3] = max(noisy_box[1] + 0.02, noisy_box[3] + noise[3])

        noisy_boxes.append(noisy_box)

    # Initialize tracker
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)
    tracked_boxes = []

    # Track using noisy boxes
    for noisy_box in noisy_boxes:
        detection = Detection(noisy_box, class_id=0, class_name="test", confidence=1.0)
        tracks = tracker.update([detection])
        if tracks:
            tracked_boxes.append(tracks[0].get_state())

    # Visualize the results
    plt.figure(figsize=(10, 10))

    # Plot true trajectory
    true_boxes = np.array(true_boxes)
    true_centers = (true_boxes[:, :2] + true_boxes[:, 2:]) / 2
    plt.plot(
        true_centers[:, 0], true_centers[:, 1], "b.-", label="True Path", markersize=10
    )

    # Plot tracked trajectory
    tracked_boxes = np.array(tracked_boxes)
    tracked_centers = (tracked_boxes[:, :2] + tracked_boxes[:, 2:]) / 2
    plt.plot(
        tracked_centers[:, 0],
        tracked_centers[:, 1],
        "g.-",
        label="Tracked Path",
        markersize=8,
    )

    # Plot noisy detections
    noisy_boxes = np.array(noisy_boxes)
    noisy_centers = (noisy_boxes[:, :2] + noisy_boxes[:, 2:]) / 2
    plt.plot(
        noisy_centers[:, 0],
        noisy_centers[:, 1],
        "r.",
        label="Noisy Detections",
        markersize=8,
    )

    # Plot boxes
    for true_box, tracked_box, noisy_box in zip(true_boxes, tracked_boxes, noisy_boxes):
        # True box in blue
        plt.plot(
            [true_box[0], true_box[2], true_box[2], true_box[0], true_box[0]],
            [true_box[1], true_box[1], true_box[3], true_box[3], true_box[1]],
            "b-",
            alpha=0.3,
        )
        # Tracked box in green
        plt.plot(
            [
                tracked_box[0],
                tracked_box[2],
                tracked_box[2],
                tracked_box[0],
                tracked_box[0],
            ],
            [
                tracked_box[1],
                tracked_box[1],
                tracked_box[3],
                tracked_box[3],
                tracked_box[1],
            ],
            "g--",
            alpha=0.3,
        )
        # Noisy box in red
        plt.plot(
            [noisy_box[0], noisy_box[2], noisy_box[2], noisy_box[0], noisy_box[0]],
            [noisy_box[1], noisy_box[1], noisy_box[3], noisy_box[3], noisy_box[1]],
            "r--",
            alpha=0.3,
        )

    plt.grid(True)
    plt.legend()
    plt.title("Object Tracking: True Path vs Tracked Path")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis("equal")
    from imgcat import imgcat

    imgcat(plt.gcf())


def test_parabolic_motion_with_noise():
    # Create parabolic trajectory from bottom left to bottom right
    steps = 20
    x = np.linspace(0.05, 0.95, steps)  # x from 5% to 95%

    # Parabola parameters: y = a*(x-h)^2 + k
    # where (h,k) is the vertex of the parabola
    h = 0.5  # x-coordinate of vertex
    k = 0.8  # y-coordinate of vertex (peak height)
    a = -3.0  # controls width of parabola

    y = a * (x - h) ** 2 + k
    true_centers = np.column_stack((x, y))

    width, height = 0.05, 0.05

    # Generate true boxes and noisy detections
    true_boxes = []
    noisy_boxes = []
    noise_level = 0.02

    for center_x, center_y in true_centers:
        # Create true box
        true_box = np.array(
            [
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2,
            ]
        )
        true_boxes.append(true_box)

        # Add noise while preserving box validity
        noise = np.random.normal(0, noise_level, 4)
        noisy_box = true_box.copy()
        # Apply noise to left/top coordinates
        noisy_box[0] += noise[0]
        noisy_box[1] += noise[1]
        # Ensure right/bottom coordinates stay greater than left/top
        noisy_box[2] = max(noisy_box[0] + 0.02, noisy_box[2] + noise[2])
        noisy_box[3] = max(noisy_box[1] + 0.02, noisy_box[3] + noise[3])

        noisy_boxes.append(noisy_box)

    # Initialize tracker
    tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)
    tracked_boxes = []

    # Track using noisy boxes
    for noisy_box in noisy_boxes:
        detection = Detection(noisy_box, class_id=0, class_name="test", confidence=1.0)
        tracks = tracker.update([detection])
        if tracks:
            tracked_boxes.append(tracks[0].get_state())

    # Visualize the results
    plt.figure(figsize=(10, 10))

    # Plot true trajectory
    true_boxes = np.array(true_boxes)
    true_centers = (true_boxes[:, :2] + true_boxes[:, 2:]) / 2
    plt.plot(
        true_centers[:, 0], true_centers[:, 1], "b.-", label="True Path", markersize=10
    )

    # Plot tracked trajectory
    tracked_boxes = np.array(tracked_boxes)
    tracked_centers = (tracked_boxes[:, :2] + tracked_boxes[:, 2:]) / 2
    plt.plot(
        tracked_centers[:, 0],
        tracked_centers[:, 1],
        "g.-",
        label="Tracked Path",
        markersize=8,
    )

    # Plot noisy detections
    noisy_boxes = np.array(noisy_boxes)
    noisy_centers = (noisy_boxes[:, :2] + noisy_boxes[:, 2:]) / 2
    plt.plot(
        noisy_centers[:, 0],
        noisy_centers[:, 1],
        "r.",
        label="Noisy Detections",
        markersize=8,
    )

    # Plot boxes
    for true_box, tracked_box, noisy_box in zip(true_boxes, tracked_boxes, noisy_boxes):
        # True box in blue
        plt.plot(
            [true_box[0], true_box[2], true_box[2], true_box[0], true_box[0]],
            [true_box[1], true_box[1], true_box[3], true_box[3], true_box[1]],
            "b-",
            alpha=0.3,
        )
        # Tracked box in green
        plt.plot(
            [
                tracked_box[0],
                tracked_box[2],
                tracked_box[2],
                tracked_box[0],
                tracked_box[0],
            ],
            [
                tracked_box[1],
                tracked_box[1],
                tracked_box[3],
                tracked_box[3],
                tracked_box[1],
            ],
            "g--",
            alpha=0.3,
        )
        # Noisy box in red
        plt.plot(
            [noisy_box[0], noisy_box[2], noisy_box[2], noisy_box[0], noisy_box[0]],
            [noisy_box[1], noisy_box[1], noisy_box[3], noisy_box[3], noisy_box[1]],
            "r--",
            alpha=0.3,
        )

    plt.grid(True)
    plt.legend()
    plt.title("Object Tracking: True Path vs Tracked Path")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.axis("equal")
    from imgcat import imgcat

    imgcat(plt.gcf())
