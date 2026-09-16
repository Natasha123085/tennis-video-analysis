import os
import sys
import cv2

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from analysis.trackers.player_tracker import PlayerTracker
from analysis.trackers.ball_tracker import BallTracker
from analysis.court_line_detector import CourtLineDetector

VIDEO_PATH = r"D:\tennis_project\analysis\input_videos\input_video.mp4"
PLAYER_MODEL_PATH = r"D:\tennis_project\analysis\models\yolov8n.pt"
BALL_MODEL_PATH = r"D:\tennis_project\analysis\models\yolo5_last.pt"
COURT_MODEL_PATH = r"D:\tennis_project\analysis\models\keypoints_model.pth"


def kp_point(keypoints, idx):
    return (int(keypoints[2 * idx]), int(keypoints[2 * idx + 1]))


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []

    max_frames = 120  # first 120 frames for testing
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

    # 1. court keypoints from first frame
    court_line_detector = CourtLineDetector(COURT_MODEL_PATH)
    court_keypoints = court_line_detector.predict(frames[0])
    print("Court keypoints detected.")

    # 2. player detection on all frames
    player_tracker = PlayerTracker(model_path=PLAYER_MODEL_PATH)
    all_player_detections = player_tracker.detect_frames(
        frames,
        read_from_stub=False,
        stub_path=None
    )
    print("All player detections done.")

    # 3. keep only the 2 tennis players
    filtered_player_detections = player_tracker.choose_and_filter_players(
        court_keypoints,
        all_player_detections
    )
    print("Filtered to 2 tennis players.")

    # 4. build court side lines for ball filtering
    left_line = (
        kp_point(court_keypoints, 0),
        kp_point(court_keypoints, 2)
    )
    right_line = (
        kp_point(court_keypoints, 1),
        kp_point(court_keypoints, 3)
    )

    # 5. ball detection on all frames
    ball_tracker = BallTracker(model_path=BALL_MODEL_PATH)
    ball_detections = ball_tracker.detect_frames(
        frames,
        player_detections=filtered_player_detections,
        left_line=left_line,
        right_line=right_line,
        read_from_stub=False,
        stub_path=None
    )
    print("Ball detections done.")

    # 6. interpolate missing ball positions
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    print("Ball interpolation done.")

    # 7. detect likely shot / hit frames
    hit_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    print("Detected hit frames:")
    print(hit_frames)

    # 8. show which player is closest to ball at each hit frame
    print("\nLikely hitter for each hit frame:")
    for frame_idx in hit_frames:
        if frame_idx >= len(filtered_player_detections) or frame_idx >= len(ball_detections):
            continue

        player_dict = filtered_player_detections[frame_idx]
        ball_dict = ball_detections[frame_idx]

        if 1 not in ball_dict or not player_dict:
            print(f"Frame {frame_idx}: no ball or no players")
            continue

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

        print(f"Frame {frame_idx}: likely hit by Player {closest_player}")

    # 9. optional: save preview frames with hit labels
    output_dir = r"D:\tennis_project\analysis\input_videos\hit_previews"
    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in hit_frames[:10]:  # save first 10 hit frames only
        if frame_idx >= len(frames):
            continue

        frame = frames[frame_idx].copy()

        # draw players
        if frame_idx < len(filtered_player_detections):
            for player_id, bbox in filtered_player_detections[frame_idx].items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"Player {player_id}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        # draw ball
        if frame_idx < len(ball_detections) and 1 in ball_detections[frame_idx]:
            bx1, by1, bx2, by2 = map(int, ball_detections[frame_idx][1])
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                "Ball",
                (bx1, max(30, by1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.putText(
            frame,
            f"HIT FRAME {frame_idx}",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )

        save_path = os.path.join(output_dir, f"hit_frame_{frame_idx}.jpg")
        cv2.imwrite(save_path, frame)

    print(f"\nSaved preview hit frames to: {output_dir}")


if __name__ == "__main__":
    main()