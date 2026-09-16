import cv2
import mediapipe as mp

from analysis.shot.shot_detector import get_likely_hitter
from analysis.shot.shot_classifier import classify_shot_sequence_5class
from analysis.shot.pose_helpers import get_pose_keypoints_for_player

mp_pose = mp.solutions.pose


def draw_pose_on_original_frame(frame, bbox, pose_landmarks):
    x1, y1, x2, y2 = bbox
    crop_h = y2 - y1
    crop_w = x2 - x1

    for lm in pose_landmarks.landmark:
        px = int(lm.x * crop_w) + x1
        py = int(lm.y * crop_h) + y1
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)
    labels = {
        11: "LS", 12: "RS", 13: "LE", 14: "RE", 15: "LW", 16: "RW"
    }

    for idx, text in labels.items():
        lm = pose_landmarks.landmark[idx]
        px = int(lm.x * crop_w) + x1
        py = int(lm.y * crop_h) + y1
        cv2.putText(frame, text, (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        lm1 = pose_landmarks.landmark[start_idx]
        lm2 = pose_landmarks.landmark[end_idx]

        x_start = int(lm1.x * crop_w) + x1
        y_start = int(lm1.y * crop_h) + y1
        x_end = int(lm2.x * crop_w) + x1
        y_end = int(lm2.y * crop_h) + y1

        cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
def draw_shot_overlay(frame, player_id, shot_label, shot_conf):
    text1 = f"Player {player_id}"
    label_map = {
        "serve_candidate": "Serve",
        "forehand_like": "Forehand",
        "backhand_like": "Backhand",
        "volley_candidate": "Volley",
        "smash_candidate": "Smash"
    }
    text2 = label_map.get(shot_label, shot_label.replace("_", " ").title())
    text3 = f"Confidence: {shot_conf:.2f}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale_title = 0.9
    scale_body = 0.7
    thickness_title = 2
    thickness_body = 2

    # Box position
    x, y = 30, frame.shape[0] - 140
    box_w, box_h = 390, 120

    # Semi-transparent dark box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (20, 20, 20), -1)
    alpha = 0.65
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Border
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2)

    # Text
    cv2.putText(frame, text1, (x + 15, y + 30), font, scale_title, (0, 255, 255), thickness_title)
    cv2.putText(frame, text2, (x + 15, y + 65), font, scale_title, (255, 255, 255), thickness_title)
    cv2.putText(frame, text3, (x + 15, y + 95), font, scale_body, (180, 255, 180), thickness_body)

def detect_and_classify_shots(
    frames,
    filtered_player_detections,
    ball_detections,
    hit_frames,
    draw_on_frames=True,
    min_confidence=0.30
):
    """
    Returns:
        shot_events: list of dicts
        annotated_frames: list of frames with shot labels drawn
    """

    annotated_frames = [frame.copy() for frame in frames]
    shot_events = []

    sorted_hit_frames = sorted(hit_frames)
    print("ball_shot_frames:", sorted_hit_frames)
    last_label = ""
    last_player = None
    last_conf = 0.0
    label_countdown = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3
    ) as pose:

        for i in range(len(frames)):
            frame = annotated_frames[i]

            if i in hit_frames and i - 1 >= 0 and i + 1 < len(frames):
                print(f"[HIT FRAME] i={i}")
                curr_players = filtered_player_detections[i]
                curr_ball = ball_detections[i]
                hitter_id = get_likely_hitter(curr_players, curr_ball)
                if hitter_id is None:
                    print(f"[SHOT TYPE MISS] frame={i}, reason=no hitter")

                if 1 not in curr_ball:
                    print(f"[SHOT TYPE MISS] frame={i}, reason=no ball")
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
                        print(f"Checking hit frame {i}, hitter={hitter_id}")
                        print(
                            f"prev_kp exists: {prev_kp is not None}, curr_kp exists: {curr_kp is not None}, next_kp exists: {next_kp is not None}")
                        hit_order = sorted_hit_frames.index(i) if i in sorted_hit_frames else None

                        # fallback: if one side frame is missing, reuse current frame
                        if curr_kp is None:
                            print(f"[POSE FALLBACK] frame={i}, current pose missing, use prev/next pose")

                            if prev_kp is not None:
                                curr_kp = prev_kp
                            elif next_kp is not None:
                                curr_kp = next_kp
                            elif prev_kp is not None:
                                curr_kp = prev_kp
                            else:
                                print(f"[SHOT TYPE MISS] frame={i}, reason=no pose")
                                continue

                        if prev_kp is None:
                            prev_kp = curr_kp
                        if next_kp is None:
                            next_kp = curr_kp

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
                        print(f"[SHOT TYPE OK] frame={i}, player={hitter_id}, label={shot_label}, confidence={shot_conf:.2f}")

                        if shot_conf >= min_confidence:
                            shot_events.append({
                                "frame": i,
                                "player_id": hitter_id,
                                "label": shot_label,
                                "confidence": float(shot_conf)
                            })

                            last_label = shot_label
                            last_player = hitter_id
                            last_conf = shot_conf
                            label_countdown = 12

                            print(
                                f"Frame {i}: Player {hitter_id} -> "
                                f"{shot_label} | confidence={shot_conf:.2f}"
                            )

                            if draw_on_frames and curr_landmarks is not None and curr_draw_bbox is not None:
                                    draw_pose_on_original_frame(frame, curr_draw_bbox, curr_landmarks)
                        else:
                            print(
                                f"Frame {i}: low confidence shot ignored "
                                f"({shot_label}, {shot_conf:.2f})"
                            )

            if draw_on_frames and label_countdown > 0:
                draw_shot_overlay(frame, last_player, last_label, last_conf)
                label_countdown -= 1

    return shot_events, annotated_frames