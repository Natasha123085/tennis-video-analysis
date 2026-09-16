import os
import sys
import cv2
import mediapipe as mp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from analysis.trackers.player_tracker import PlayerTracker
from analysis.trackers.ball_tracker import BallTracker
from analysis.court_line_detector import CourtLineDetector
from analysis.shot.shot_detector import get_likely_hitter
from analysis.shot.shot_classifier import classify_shot_sequence_5class
from analysis.shot.pose_helpers import get_pose_keypoints_for_player

VIDEO_PATH = r"D:\tennis_project\analysis\input_videos\input_video.mp4"
PLAYER_MODEL_PATH = r"D:\tennis_project\analysis\models\yolov8n.pt"
BALL_MODEL_PATH = r"D:\tennis_project\analysis\models\yolo5_last.pt"
COURT_MODEL_PATH = r"D:\tennis_project\analysis\models\keypoints_model.pth"
OUTPUT_VIDEO_PATH = r"D:\tennis_project\analysis\input_videos\realtime_shot_demo.mp4"

mp_pose = mp.solutions.pose


def kp_point(keypoints, idx):
    return (int(keypoints[2 * idx]), int(keypoints[2 * idx + 1]))


def draw_pose_on_original_frame(frame, bbox, pose_landmarks):
    x1, y1, x2, y2 = bbox
    crop_h = y2 - y1
    crop_w = x2 - x1

    for lm in pose_landmarks.landmark:
        px = int(lm.x * crop_w) + x1
        py = int(lm.y * crop_h) + y1
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        lm1 = pose_landmarks.landmark[start_idx]
        lm2 = pose_landmarks.landmark[end_idx]

        x_start = int(lm1.x * crop_w) + x1
        y_start = int(lm1.y * crop_h) + y1
        x_end = int(lm2.x * crop_w) + x1
        y_end = int(lm2.y * crop_h) + y1

        cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        print("No frames loaded.")
        return

    fps = 24
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h)
    )

    # 1. Detect court once from first frame
    court_line_detector = CourtLineDetector(COURT_MODEL_PATH)
    court_keypoints = court_line_detector.predict(frames[0])

    # 2. Detect players on all frames
    player_tracker = PlayerTracker(model_path=PLAYER_MODEL_PATH)
    all_player_detections = player_tracker.detect_frames(
        frames,
        read_from_stub=False,
        stub_path=None
    )
    filtered_player_detections = player_tracker.choose_and_filter_players(
        court_keypoints,
        all_player_detections
    )

    # 3. Build court side boundary lines
    left_line = (
        kp_point(court_keypoints, 0),
        kp_point(court_keypoints, 2)
    )
    right_line = (
        kp_point(court_keypoints, 1),
        kp_point(court_keypoints, 3)
    )

    # 4. Detect and interpolate ball
    ball_tracker = BallTracker(model_path=BALL_MODEL_PATH)
    ball_detections = ball_tracker.detect_frames(
        frames,
        player_detections=filtered_player_detections,
        left_line=left_line,
        right_line=right_line,
        read_from_stub=False,
        stub_path=None
    )
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # 5. Detect hit frames
    hit_frames = set(ball_tracker.get_ball_shot_frames(ball_detections))
    sorted_hit_frames = sorted(hit_frames)
    print("Detected hit frames:", sorted_hit_frames)

    last_label = ""
    last_player = None
    label_countdown = 0

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.3
    ) as pose:

        for i in range(len(frames)):
            frame = frames[i].copy()

            # classify only when current frame is a hit frame
            if i in hit_frames and i - 1 >= 0 and i + 1 < len(frames):
                curr_players = filtered_player_detections[i]
                curr_ball = ball_detections[i]
                hitter_id = get_likely_hitter(curr_players, curr_ball)

                if hitter_id is not None and 1 in curr_ball:
                    has_prev = hitter_id in filtered_player_detections[i - 1]
                    has_curr = hitter_id in filtered_player_detections[i]
                    has_next = hitter_id in filtered_player_detections[i + 1]

                    if has_prev and has_curr and has_next:
                        prev_bbox = filtered_player_detections[i - 1][hitter_id]
                        curr_bbox = filtered_player_detections[i][hitter_id]
                        next_bbox = filtered_player_detections[i + 1][hitter_id]

                        prev_kp, _, _ = get_pose_keypoints_for_player(frames[i - 1], prev_bbox, pose)
                        curr_kp, curr_landmarks, curr_draw_bbox = get_pose_keypoints_for_player(frames[i], curr_bbox, pose)
                        next_kp, _, _ = get_pose_keypoints_for_player(frames[i + 1], next_bbox, pose)

                        if prev_kp and curr_kp and next_kp:
                            hit_order = sorted_hit_frames.index(i) if i in sorted_hit_frames else None

                            shot_label, shot_conf = classify_shot_sequence_5class(
                                prev_kp,
                                curr_kp,
                                next_kp,
                                curr_ball[1],
                                curr_bbox,
                                frame.shape,
                                hitter_id,
                                hit_order
                            )

                            last_label = f"{shot_label} ({shot_conf:.2f})"
                            last_player = hitter_id
                            label_countdown = 12

                            print(f"Frame {i}: Player {hitter_id} -> {shot_label} | confidence={shot_conf:.2f}")

                            if curr_landmarks is not None and curr_draw_bbox is not None:
                                draw_pose_on_original_frame(frame, curr_draw_bbox, curr_landmarks)

            # draw two filtered players
            for player_id, bbox in filtered_player_detections[i].items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"Player {player_id}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

            # draw ball
            if 1 in ball_detections[i]:
                bx1, by1, bx2, by2 = map(int, ball_detections[i][1])
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Ball",
                    (bx1, max(30, by1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # keep label visible for a few frames
            if label_countdown > 0:
                cv2.putText(
                    frame,
                    f"Player {last_player}: {last_label}",
                    (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )
                label_countdown -= 1

            writer.write(frame)

    writer.release()
    print("Saved realtime demo video to:", OUTPUT_VIDEO_PATH)


if __name__ == "__main__":
    main()