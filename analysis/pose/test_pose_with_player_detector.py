import os
import sys
import cv2
import mediapipe as mp

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from analysis.trackers.player_tracker import PlayerTracker
from analysis.court_line_detector import CourtLineDetector
from analysis.pose.pose_utils import extract_keypoints, get_player_pose_data

VIDEO_PATH = r"D:\tennis_project\analysis\input_videos\input_video.mp4"
PLAYER_MODEL_PATH = r"D:\tennis_project\analysis\models\yolov8n.pt"
COURT_MODEL_PATH = r"D:\tennis_project\analysis\models\keypoints_model.pth"
OUTPUT_PATH = r"D:\tennis_project\analysis\input_videos\two_players_pose_result.jpg"

mp_pose = mp.solutions.pose


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


def draw_court_keypoints(frame, keypoints):
    for i in range(0, len(keypoints), 2):
        x = int(keypoints[i])
        y = int(keypoints[i + 1])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(i // 2),
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )


def classify_forehand_backhand(keypoints):
    left_shoulder = keypoints["left_shoulder"]
    right_shoulder = keypoints["right_shoulder"]
    right_wrist = keypoints["right_wrist"]

    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2

    if right_wrist[0] > shoulder_center_x:
        return "forehand_like"
    else:
        return "backhand_like"


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    target_frame_idx = 100
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Could not read video frame.")
        return

    # 1. court keypoints
    court_line_detector = CourtLineDetector(COURT_MODEL_PATH)
    court_keypoints = court_line_detector.predict(frame)
    print("Court keypoints detected.")

    # 2. detect all people
    player_tracker = PlayerTracker(model_path=PLAYER_MODEL_PATH)
    all_player_detections = player_tracker.detect_frame(frame)
    print("All detected people:", all_player_detections)

    if not all_player_detections:
        print("No people detected.")
        return

    # 3. filter to only 2 tennis players
    filtered_players_list = player_tracker.choose_and_filter_players(
        court_keypoints,
        [all_player_detections]
    )
    player_detections = filtered_players_list[0]

    print("Filtered tennis players:", player_detections)

    if not player_detections:
        print("No tennis players selected after filtering.")
        return

    # 4. draw court keypoints
    draw_court_keypoints(frame, court_keypoints)

    # 5. store pose data
    player_pose_data = {}

    # 6. pose estimation only on filtered 2 players
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.3
    ) as pose:

        for player_id, bbox in player_detections.items():
            x1, y1, x2, y2 = enlarge_bbox(bbox, frame.shape, scale_x=0.15, scale_y=0.15)

            player_crop = frame[y1:y2, x1:x2]

            if player_crop.size == 0:
                print(f"Empty crop for player {player_id}")
                continue

            rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                f"Player {player_id}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            if results.pose_landmarks:
                print(f"Pose detected for player {player_id}")

                keypoints = get_player_pose_data(results, (x1, y1, x2, y2))
                player_pose_data[player_id] = keypoints

                print(f"Player {player_id} keypoints:", keypoints)

                shot_type = classify_forehand_backhand(keypoints)
                print(f"Player {player_id} shot type: {shot_type}")

                cv2.putText(
                    frame,
                    shot_type,
                    (x1, min(frame.shape[0] - 10, y2 + 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2
                )

                draw_pose_on_original_frame(frame, (x1, y1, x2, y2), results.pose_landmarks)
            else:
                print(f"No pose detected for player {player_id}")

    print("Final player pose data:", player_pose_data)

    cv2.imwrite(OUTPUT_PATH, frame)
    print("Saved result to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()