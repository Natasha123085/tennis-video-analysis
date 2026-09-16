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
from analysis.pose.pose_utils import get_player_pose_data

VIDEO_PATH = r"D:\tennis_project\analysis\input_videos\input_video.mp4"
PLAYER_MODEL_PATH = r"D:\tennis_project\analysis\models\yolov8n.pt"
BALL_MODEL_PATH = r"D:\tennis_project\analysis\models\yolo5_last.pt"
COURT_MODEL_PATH = r"D:\tennis_project\analysis\models\keypoints_model.pth"

mp_pose = mp.solutions.pose


def kp_point(keypoints, idx):
    return (int(keypoints[2 * idx]), int(keypoints[2 * idx + 1]))


def enlarge_bbox(bbox, frame_shape, scale_x=0.15, scale_y=0.15):
    x1, y1, x2, y2 = bbox
    h, w = frame_shape[:2]

    bw = x2 - x1
    bh = y2 - y1

    x1 = max(0, int(x1 - bw * scale_x))
    x2 = min(w, int(x2 + bw * scale_x))
    y1 = max(0, int(y1 - bh * scale_y))
    y2 = min(h, int(y2 + bh * scale_y))

    return x1, y1, x2, y2


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


def get_likely_hitter(player_dict, ball_dict):
    if 1 not in ball_dict or not player_dict:
        return None

    bx1, by1, bx2, by2 = ball_dict[1]
    ball_center = ((bx1 + bx2) / 2, (by1 + by2) / 2)

    closest_player = None
    closest_dist = float("inf")

    for player_id, bbox in player_dict.items():
        px1, py1, px2, py2 = bbox
        player_center = ((px1 + px2) / 2, (py1 + py2) / 2)

        dist = ((player_center[0] - ball_center[0]) ** 2 + (player_center[1] - ball_center[1]) ** 2) ** 0.5

        if dist < closest_dist:
            closest_dist = dist
            closest_player = player_id

    return closest_player


def get_pose_keypoints_for_player(frame, bbox, pose_model):
    x1, y1, x2, y2 = enlarge_bbox(bbox, frame.shape, scale_x=0.15, scale_y=0.15)
    player_crop = frame[y1:y2, x1:x2]

    if player_crop.size == 0:
        return None, None

    rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb_crop)

    if not results.pose_landmarks:
        return None, None

    keypoints = get_player_pose_data(results, (x1, y1, x2, y2))
    return keypoints, results.pose_landmarks


def classify_shot_sequence_5class(prev_kp, curr_kp, next_kp, ball_bbox, player_bbox, frame_shape):
    left_shoulder = curr_kp["left_shoulder"]
    right_shoulder = curr_kp["right_shoulder"]
    left_wrist = curr_kp["left_wrist"]
    right_wrist = curr_kp["right_wrist"]
    left_hip = curr_kp["left_hip"]
    right_hip = curr_kp["right_hip"]

    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2

    bx1, by1, bx2, by2 = ball_bbox
    ball_x = (bx1 + bx2) / 2
    ball_y = (by1 + by2) / 2

    px1, py1, px2, py2 = player_bbox
    foot_y = py2
    frame_h, frame_w = frame_shape[:2]

    left_dist = ((left_wrist[0] - ball_x) ** 2 + (left_wrist[1] - ball_y) ** 2) ** 0.5
    right_dist = ((right_wrist[0] - ball_x) ** 2 + (right_wrist[1] - ball_y) ** 2) ** 0.5
    hitting_side = "right" if right_dist < left_dist else "left"

    prev_rw = prev_kp["right_wrist"]
    next_rw = next_kp["right_wrist"]
    prev_lw = prev_kp["left_wrist"]
    next_lw = next_kp["left_wrist"]

    right_motion_x = next_rw[0] - prev_rw[0]
    left_motion_x = next_lw[0] - prev_lw[0]

    near_net = frame_h * 0.35 < foot_y < frame_h * 0.7
    near_baseline_bottom = foot_y > frame_h * 0.78
    near_baseline_top = foot_y < frame_h * 0.22
    near_baseline = near_baseline_bottom or near_baseline_top

    above_shoulders = ball_y < shoulder_center_y
    between_shoulder_and_hip = shoulder_center_y <= ball_y <= hip_center_y

    # stricter serve rule:
    # only top player near very top baseline OR bottom player near very bottom baseline
    # and ball clearly above shoulders
    if above_shoulders and near_baseline:
        bbox_center_y = (py1 + py2) / 2
        player_height = py2 - py1
        if player_height > 140:   # large enough player crop
            return "serve_candidate"

    # smash: high contact but not baseline serve position
    if above_shoulders and not near_baseline:
        return "smash_candidate"

    # volley: around net zone and medium contact height
    if near_net and between_shoulder_and_hip:
        return "volley_candidate"

    # groundstroke
    if hitting_side == "right":
        if ball_x > shoulder_center_x and right_motion_x >= 0:
            return "forehand_like"
        elif ball_x < shoulder_center_x:
            return "backhand_like"
        else:
            return "forehand_like"
    else:
        if ball_x < shoulder_center_x and left_motion_x <= 0:
            return "forehand_like"
        elif ball_x > shoulder_center_x:
            return "backhand_like"
        else:
            return "backhand_like"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []

    max_frames = 120
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        print("No frames loaded.")
        return

    print(f"Loaded {len(frames)} frames.")

    # 1. court
    court_line_detector = CourtLineDetector(COURT_MODEL_PATH)
    court_keypoints = court_line_detector.predict(frames[0])

    # 2. players
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

    # 3. ball
    left_line = (
        kp_point(court_keypoints, 0),
        kp_point(court_keypoints, 2)
    )
    right_line = (
        kp_point(court_keypoints, 1),
        kp_point(court_keypoints, 3)
    )

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
    hit_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    print("Detected hit frames:", hit_frames)

    output_dir = r"D:\tennis_project\analysis\input_videos\hit_pose_sequence_results"
    os.makedirs(output_dir, exist_ok=True)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.3
    ) as pose:

        for frame_idx in hit_frames:
            if frame_idx - 1 < 0 or frame_idx + 1 >= len(frames):
                continue

            curr_players = filtered_player_detections[frame_idx]
            curr_ball = ball_detections[frame_idx]

            hitter_id = get_likely_hitter(curr_players, curr_ball)
            if hitter_id is None:
                print(f"Frame {frame_idx}: no hitter found")
                continue

            # player bbox at prev / curr / next
            if hitter_id not in filtered_player_detections[frame_idx - 1]:
                continue
            if hitter_id not in filtered_player_detections[frame_idx]:
                continue
            if hitter_id not in filtered_player_detections[frame_idx + 1]:
                continue

            prev_bbox = filtered_player_detections[frame_idx - 1][hitter_id]
            curr_bbox = filtered_player_detections[frame_idx][hitter_id]
            next_bbox = filtered_player_detections[frame_idx + 1][hitter_id]

            prev_kp, _ = get_pose_keypoints_for_player(frames[frame_idx - 1], prev_bbox, pose)
            curr_kp, curr_landmarks = get_pose_keypoints_for_player(frames[frame_idx], curr_bbox, pose)
            next_kp, _ = get_pose_keypoints_for_player(frames[frame_idx + 1], next_bbox, pose)

            if prev_kp is None or curr_kp is None or next_kp is None:
                print(f"Frame {frame_idx}: missing pose in sequence")
                continue

            if 1 not in curr_ball:
                print(f"Frame {frame_idx}: missing ball")
                continue

            shot_type = classify_shot_sequence_5class(
                prev_kp,
                curr_kp,
                next_kp,
                curr_ball[1],
                curr_bbox,
                frames[frame_idx].shape
            )

            print(f"Frame {frame_idx}: Player {hitter_id} -> {shot_type}")
            print(f"Prev keypoints: {prev_kp}")
            print(f"Curr keypoints: {curr_kp}")
            print(f"Next keypoints: {next_kp}")

            frame = frames[frame_idx].copy()

            # draw hitter bbox
            x1, y1, x2, y2 = enlarge_bbox(curr_bbox, frame.shape, scale_x=0.15, scale_y=0.15)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"Hitter Player {hitter_id}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            # draw ball
            bx1, by1, bx2, by2 = map(int, curr_ball[1])
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

            cv2.putText(
                frame,
                shot_type,
                (x1, min(frame.shape[0] - 10, y2 + 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            if curr_landmarks is not None:
                draw_pose_on_original_frame(frame, (x1, y1, x2, y2), curr_landmarks)

            cv2.putText(
                frame,
                f"HIT FRAME {frame_idx}",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3
            )

            save_path = os.path.join(output_dir, f"hit_pose_seq_{frame_idx}.jpg")
            cv2.imwrite(save_path, frame)

    print(f"Saved sequence classification images to: {output_dir}")


if __name__ == "__main__":
    main()