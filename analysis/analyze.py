from .utils import (
    read_video,
    save_video,
    measure_distance,
    draw_player_stats,
    convert_pixel_distance_to_meters
)
from . import constants
from .trackers import PlayerTracker, BallTracker
from .court_line_detector import CourtLineDetector
from .mini_court import MiniCourt
from .core_analysis.shot_analysis import classify_shot
from .core_analysis.smoothing import smooth_positions
from .shot.shot_pipeline import detect_and_classify_shots
import cv2
import pandas as pd
from copy import deepcopy
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def kp_point(keypoints, idx):
    return (int(keypoints[2 * idx]), int(keypoints[2 * idx + 1]))
def calculate_raw_ball_detection_metrics(raw_ball_detections):
    total_frames = len(raw_ball_detections)
    detected_frames = 0

    for frame_data in raw_ball_detections:
        if frame_data and len(frame_data) > 0:
            detected_frames += 1

    missed_frames = total_frames - detected_frames
    detection_rate = (detected_frames / total_frames) * 100 if total_frames > 0 else 0

    return {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "missed_frames": missed_frames,
        "detection_rate": round(detection_rate, 2)
    }
def calculate_player_tracking_metrics(player_detections):
    total_frames = len(player_detections)

    both_players_present = 0
    player_1_present = 0
    player_2_present = 0
    player_1_missing_frames = 0
    player_2_missing_frames = 0
    id_switch_count = 0
    prev_ids = None

    for frame_data in player_detections:
        current_ids = sorted(list(frame_data.keys())) if frame_data else []

        if 1 in current_ids:
            player_1_present += 1
        else:
            player_1_missing_frames += 1

        if 2 in current_ids:
            player_2_present += 1
        else:
            player_2_missing_frames += 1

        if 1 in current_ids and 2 in current_ids:
            both_players_present += 1

        if prev_ids is not None and current_ids != prev_ids:
            id_switch_count += 1

        prev_ids = current_ids

    both_players_rate = (both_players_present / total_frames) * 100 if total_frames > 0 else 0
    player_1_rate = (player_1_present / total_frames) * 100 if total_frames > 0 else 0
    player_2_rate = (player_2_present / total_frames) * 100 if total_frames > 0 else 0

    return {
        "total_frames": total_frames,
        "both_players_present_frames": both_players_present,
        "both_players_present_rate": round(both_players_rate, 2),
        "player_1_present_rate": round(player_1_rate, 2),
        "player_2_present_rate": round(player_2_rate, 2),
        "player_1_missing_frames": player_1_missing_frames,
        "player_2_missing_frames": player_2_missing_frames,
        "id_switch_count": id_switch_count
    }
def calculate_final_ball_trajectory_metrics(ball_detections):
    total_frames = len(ball_detections)
    usable_frames = 0

    for frame_data in ball_detections:
        if frame_data and len(frame_data) > 0:
            usable_frames += 1

    missing_frames = total_frames - usable_frames
    continuity_rate = (usable_frames / total_frames) * 100 if total_frames > 0 else 0

    return {
        "total_frames": total_frames,
        "usable_frames": usable_frames,
        "missing_frames": missing_frames,
        "continuity_rate": round(continuity_rate, 2)
    }


def calculate_player_movement_variance(player_positions):
    import numpy as np

    movements = []

    for i in range(1, len(player_positions)):
        if i in player_positions and (i - 1) in player_positions:
            x1, y1 = player_positions[i - 1]
            x2, y2 = player_positions[i]
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            movements.append(dist)

    if len(movements) == 0:
        return 0

    return np.std(movements)
def analyze_video(input_video_path, output_video_path):
    # Read Video
    video_frames = read_video(input_video_path)

    # Detect Players and Ball
    player_tracker = PlayerTracker(model_path=os.path.join(MODEL_DIR, "yolov8n.pt"))
    ball_tracker = BallTracker(model_path=os.path.join(MODEL_DIR, "yolo5_last.pt"))

    stub_player_path = os.path.join(BASE_DIR, "tracker_stubs", "player_detections.pkl")
    stub_ball_path = os.path.join(BASE_DIR, "tracker_stubs", "ball_detections.pkl")

    # 1. Detect players first
    raw_player_detections = player_tracker.detect_frames(
        video_frames,
        read_from_stub=False,
        stub_path=stub_player_path
    )

    # 2. Detect court keypoints
    court_model_path = os.path.join(BASE_DIR, "models/keypoints_model.pth")
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # 3. Choose and fix the two players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, raw_player_detections)

    # 4. Court side boundary lines
    # left side line: keypoints 0 -> 2
    # right side line: keypoints 1 -> 3
    left_line = (
        kp_point(court_keypoints, 0),
        kp_point(court_keypoints, 2)
    )
    right_line = (
        kp_point(court_keypoints, 1),
        kp_point(court_keypoints, 3)
    )

    # 5. Detect ball within the two court side lines
    raw_ball_detections = ball_tracker.detect_frames(
        video_frames,
        player_detections=player_detections,
        left_line=left_line,
        right_line=right_line,
        read_from_stub=False,
        stub_path=stub_ball_path
    )

    # 6. Interpolate ball positions
    ball_detections = ball_tracker.interpolate_ball_positions(raw_ball_detections)

    # Detect shot frames
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Evaluation metrics
    raw_ball_metrics = calculate_raw_ball_detection_metrics(raw_ball_detections)
    final_ball_metrics = calculate_final_ball_trajectory_metrics(ball_detections)
    player_tracking_metrics = calculate_player_tracking_metrics(player_detections)

    print("\n===== Raw Ball Detection Metrics =====")
    for key, value in raw_ball_metrics.items():
        print(f"{key}: {value}")

    print("\n===== Final Ball Trajectory Metrics =====")
    for key, value in final_ball_metrics.items():
        print(f"{key}: {value}")

    print("\n===== Player Tracking Metrics =====")
    for key, value in player_tracking_metrics.items():
        print(f"{key}: {value}")
    # Detect and classify shot events
    shot_events, shot_annotated_frames = detect_and_classify_shots(
        frames=video_frames,
        filtered_player_detections=player_detections,
        ball_detections=ball_detections,
        hit_frames=set(ball_shot_frames),
        draw_on_frames=True,
        min_confidence=0.50
    )
    # MiniCourt
    mini_court = MiniCourt(video_frames[0])
    #mini_court.compute_homography(court_keypoints)
    center_x = mini_court.get_center_x()
    pixels_per_meter = mini_court.get_width_of_mini_court() / constants.DOUBLE_LINE_WIDTH
    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = (
        mini_court.convert_bounding_boxes_to_mini_court_coordinates(
            player_detections,
            ball_detections,
            court_keypoints
        )
    )

    player_mini_court_detections = smooth_positions(
        player_mini_court_detections,
        alpha=0.7
    )

    player1_positions = {}
    player2_positions = {}

    for frame_idx, frame_data in enumerate(player_mini_court_detections):
        if 1 in frame_data:
            player1_positions[frame_idx] = frame_data[1]
        if 2 in frame_data:
            player2_positions[frame_idx] = frame_data[2]
    player1_variance = calculate_player_movement_variance(player1_positions)
    player2_variance = calculate_player_movement_variance(player2_positions)

    print("\n===== Player Movement Stability =====")
    print(f"Player 1 movement variance: {player1_variance:.4f}")
    print(f"Player 2 movement variance: {player2_variance:.4f}")

    recovery_fail = {1: 0, 2: 0}
    total_positions = {1: 0, 2: 0}

    for frame_data in player_mini_court_detections:
        for player_id, (player_x, player_y) in frame_data.items():

            if abs(player_x - center_x) > mini_court.get_width_of_mini_court() * 0.2:
                recovery_fail[player_id] += 1

            total_positions[player_id] += 1

    recovery_rate_1 = 0
    recovery_rate_2 = 0

    for player_id in [1, 2]:
        if total_positions[player_id] > 0:
            recovery_rate = 1 - (recovery_fail[player_id] / total_positions[player_id])

            if player_id == 1:
                recovery_rate_1 = recovery_rate
            else:
                recovery_rate_2 = recovery_rate

    zones = mini_court.get_court_zones()

    shot_stats = {
        "left": 0,
        "middle": 0,
        "right": 0,
        "short": 0,
        "mid": 0,
        "deep": 0
    }

    direction_stats = {
        "crosscourt": 0,
        "down_the_line": 0,
        "middle": 0
    }

    ball_speeds = []
    player_speeds = []


    player_stats_data = [{
        "frame_num": 0,
        "player_1_number_of_shots": 0,
        "player_1_total_shot_speed": 0,
        "player_1_last_shot_speed": 0,
        "player_1_total_player_speed": 0,
        "player_1_last_player_speed": 0,

        "player_2_number_of_shots": 0,
        "player_2_total_shot_speed": 0,
        "player_2_last_shot_speed": 0,
        "player_2_total_player_speed": 0,
        "player_2_last_player_speed": 0,
    }]
    num_hit_events = len(ball_shot_frames)
    num_shot_intervals = max(0, num_hit_events - 1)
    for ball_shot_ind in range(num_shot_intervals):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]

        # ball landing position
        landing_position = ball_mini_court_detections[end_frame][1]
        horizontal, depth = classify_shot(landing_position, zones)
        shot_stats[horizontal] += 1
        shot_stats[depth] += 1

        # direction classification
        x, y = landing_position
        width = mini_court.get_width_of_mini_court()

        if x < width / 3:
            direction_type = "crosscourt"
        elif x > 2 * width / 3:
            direction_type = "down_the_line"
        else:
            direction_type = "middle"

        direction_stats[direction_type] += 1

        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(
            ball_mini_court_detections[start_frame][1],
            ball_mini_court_detections[end_frame][1]
        )

        # mini court width = real doubles match width
        pixels_per_meter = mini_court.get_width_of_mini_court() / constants.DOUBLE_LINE_WIDTH
        distance_covered_by_ball_meters = distance_covered_by_ball_pixels / pixels_per_meter

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6
        ball_speeds.append(speed_of_ball_shot)

        # player who shot the ball
        player_positions = player_mini_court_detections[start_frame]
        if len(player_positions) < 2:
            continue

        player_shot_ball = min(
            player_positions.keys(),
            key=lambda player_id: measure_distance(
                player_positions[player_id],
                ball_mini_court_detections[start_frame][1]
            )
        )

        # opponent player speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        if opponent_player_id not in player_mini_court_detections[start_frame] or \
           opponent_player_id not in player_mini_court_detections[end_frame]:
            continue

        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id]
        )

        distance_covered_by_opponent_meters = distance_covered_by_opponent_pixels / pixels_per_meter
        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6
        player_speeds.append(speed_of_opponent)

        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += speed_of_ball_shot
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = speed_of_ball_shot

        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += speed_of_opponent
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    total_shots = num_hit_events

    if ball_speeds:
        max_ball_speed = max(ball_speeds)
        min_ball_speed = min(ball_speeds)
        avg_ball_speed = sum(ball_speeds) / len(ball_speeds)
    else:
        max_ball_speed = 0
        min_ball_speed = 0
        avg_ball_speed = 0

    if player_speeds:
        avg_player_speed = sum(player_speeds) / len(player_speeds)
    else:
        avg_player_speed = 0

    left_percentage = middle_percentage = right_percentage = 0
    short_percentage = mid_percentage = deep_percentage = 0

    if num_shot_intervals > 0:
        left_percentage = shot_stats["left"] / num_shot_intervals
        middle_percentage = shot_stats["middle"] / num_shot_intervals
        right_percentage = shot_stats["right"] / num_shot_intervals

        short_percentage = shot_stats["short"] / num_shot_intervals
        mid_percentage = shot_stats["mid"] / num_shot_intervals
        deep_percentage = shot_stats["deep"] / num_shot_intervals

    cross_pct = direction_stats["crosscourt"] / num_shot_intervals if num_shot_intervals > 0 else 0
    dtl_pct = direction_stats["down_the_line"] / num_shot_intervals if num_shot_intervals > 0 else 0
    middle_dir_pct = direction_stats["middle"] / num_shot_intervals if num_shot_intervals > 0 else 0

    feedback = []

    if middle_percentage > 0.4:
        feedback.append(
            f"{middle_percentage*100:.0f}% of shots land in the middle. This reduces tactical pressure."
        )

    if deep_percentage < 0.4:
        feedback.append(
            f"Only {deep_percentage*100:.0f}% of shots are deep. Greater depth would push opponent back."
        )

    if left_percentage > 0.6:
        feedback.append(
            f"{left_percentage*100:.0f}% of shots target the left side. Strong directional pattern detected."
        )

    if not feedback:
        feedback.append("Shot placement is tactically balanced with good depth variation.")

    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({"frame_num": list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on="frame_num", how="left")
    player_stats_data_df = player_stats_data_df.ffill()

    player_stats_data_df["player_1_average_shot_speed"] = (
        player_stats_data_df["player_1_total_shot_speed"] /
        player_stats_data_df["player_1_number_of_shots"].replace(0, 1)
    )

    player_stats_data_df["player_2_average_shot_speed"] = (
        player_stats_data_df["player_2_total_shot_speed"] /
        player_stats_data_df["player_2_number_of_shots"].replace(0, 1)
    )

    player_stats_data_df["player_1_average_player_speed"] = (
        player_stats_data_df["player_1_total_player_speed"] /
        player_stats_data_df["player_1_number_of_shots"].replace(0, 1)
    )

    player_stats_data_df["player_2_average_player_speed"] = (
        player_stats_data_df["player_2_total_player_speed"] /
        player_stats_data_df["player_2_number_of_shots"].replace(0, 1)
    )

    if len(player_stats_data_df) > 0:
        avg_speed_1 = player_stats_data_df["player_1_average_shot_speed"].iloc[-1]
        avg_speed_2 = player_stats_data_df["player_2_average_shot_speed"].iloc[-1]
        last_shot_speed_1 = player_stats_data_df["player_1_last_shot_speed"].iloc[-1]
        last_shot_speed_2 = player_stats_data_df["player_2_last_shot_speed"].iloc[-1]

        last_player_speed_1 = player_stats_data_df["player_1_last_player_speed"].iloc[-1]
        last_player_speed_2 = player_stats_data_df["player_2_last_player_speed"].iloc[-1]

        avg_player_speed_1 = player_stats_data_df["player_1_average_player_speed"].iloc[-1]
        avg_player_speed_2 = player_stats_data_df["player_2_average_player_speed"].iloc[-1]
    else:
        avg_speed_1 = 0
        avg_speed_2 = 0
        last_shot_speed_1 = 0
        last_shot_speed_2 = 0
        last_player_speed_1 = 0
        last_player_speed_2 = 0
        avg_player_speed_1 = 0
        avg_player_speed_2 = 0

    # Draw output
    output_video_frames = shot_annotated_frames
    output_video_frames = player_tracker.draw_bboxes(output_video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    # Draw court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Draw court side boundary lines
    for frame in output_video_frames:
        cv2.line(frame, left_line[0], left_line[1], (255, 0, 0), 2)
        cv2.line(frame, right_line[0], right_line[1], (255, 0, 0), 2)

        cv2.putText(
            frame,
            "LEFT SIDE",
            left_line[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            "RIGHT SIDE",
            right_line[0],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    # Draw Mini Court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(
        output_video_frames,
        ball_mini_court_detections,
        color=(0, 255, 255)
    )

    # Draw Player Stats
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df, mini_court)

    # Draw frame number on top left corner
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    shot_type_counts = {
        "serve_candidate": 0,
        "forehand_like": 0,
        "backhand_like": 0,
        "volley_candidate": 0,
        "smash_candidate": 0
    }

    for event in shot_events:
        label = event["label"]
        if label in shot_type_counts:
            shot_type_counts[label] += 1

    results = {
        "total_shots": total_shots,

        "ball_speed": {
            "average": avg_ball_speed,
            "max": max_ball_speed,
            "min": min_ball_speed,
            "all_speeds": ball_speeds
        },

        "player_speed": {
            "average": avg_player_speed,
            "all_speeds": player_speeds
        },

        "shot_distribution": shot_stats,

        "direction_distribution": direction_stats,

        "average_shot_speed": {
            "player_1": avg_speed_1,
            "player_2": avg_speed_2
        },

        "recovery_rate": {
            "player_1": recovery_rate_1,
            "player_2": recovery_rate_2
        },



        "player_stats": {
            "last_shot_speed": {
                "player_1": last_shot_speed_1,
                "player_2": last_shot_speed_2
            },

            "last_player_speed": {
                "player_1": last_player_speed_1,
                "player_2": last_player_speed_2
            },

            "average_player_speed": {
                "player_1": avg_player_speed_1,
                "player_2": avg_player_speed_2
            }
        },

        "feedback": feedback,
        "shot_events": shot_events,
        "shot_type_counts": shot_type_counts
    }

    temp_output = output_video_path.replace(".mp4", "_temp.mp4")
    save_video(output_video_frames, temp_output)
    os.system(f'ffmpeg -i "{temp_output}" -vcodec libx264 -acodec aac "{output_video_path}" -y')

    os.remove(temp_output)
    return results