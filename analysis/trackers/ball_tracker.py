from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import math


class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        self.last_ball_center = None
        self.missed_frames = 0
        self.max_missed_frames = 5
        self.max_ball_jump = 180

        self.last_selected_center = None
        self.static_count = 0
        self.static_distance_threshold = 6
        self.max_static_frames = 4

    def get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def is_near_player_feet(self, center, player_bboxes, foot_region_ratio=0.22, x_margin=20, y_margin=12):
        cx, cy = center

        for bbox in player_bboxes:
            px1, py1, px2, py2 = bbox

            foot_top = py2 - (py2 - py1) * foot_region_ratio
            foot_left = px1 - x_margin
            foot_right = px2 + x_margin
            foot_bottom = py2 + y_margin

            if foot_left <= cx <= foot_right and foot_top <= cy <= foot_bottom:
                return True

        return False

    def point_line_side(self, point, line_p1, line_p2):
        x, y = point
        x1, y1 = line_p1
        x2, y2 = line_p2
        return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

    def is_inside_side_boundaries(self, point, left_line, right_line):
        if left_line is None or right_line is None:
            return True

        left_p1, left_p2 = left_line
        right_p1, right_p2 = right_line

        inside_reference = (
            (left_p1[0] + left_p2[0] + right_p1[0] + right_p2[0]) / 4.0,
            (left_p1[1] + left_p2[1] + right_p1[1] + right_p2[1]) / 4.0
        )

        left_ref_side = self.point_line_side(inside_reference, left_p1, left_p2)
        right_ref_side = self.point_line_side(inside_reference, right_p1, right_p2)

        point_left_side = self.point_line_side(point, left_p1, left_p2)
        point_right_side = self.point_line_side(point, right_p1, right_p2)

        left_ok = (left_ref_side == 0) or (point_left_side == 0) or (left_ref_side * point_left_side > 0)
        right_ok = (right_ref_side == 0) or (point_right_side == 0) or (right_ref_side * point_right_side > 0)

        return left_ok and right_ok

    def interpolate_ball_positions(self, ball_positions):
        processed_positions = []

        for frame_dict in ball_positions:
            bbox = frame_dict.get(1, None)

            if bbox is None or len(bbox) != 4:
                processed_positions.append([None, None, None, None])
            else:
                processed_positions.append(bbox)

        df_ball_positions = pd.DataFrame(
            processed_positions,
            columns=['x1', 'y1', 'x2', 'y2']
        )

        if df_ball_positions.isna().all().all():
            return [{} for _ in ball_positions]

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.ffill()

        output_positions = []
        for row in df_ball_positions.to_numpy().tolist():
            if any(v is None or pd.isna(v) for v in row):
                output_positions.append({})
            else:
                output_positions.append({1: row})

        return output_positions

    def get_ball_shot_frames(self, ball_positions):
        processed_positions = []

        for frame_dict in ball_positions:
            bbox = frame_dict.get(1, None)

            if bbox is None or len(bbox) != 4:
                processed_positions.append([None, None, None, None])
            else:
                processed_positions.append(bbox)

        df_ball_positions = pd.DataFrame(
            processed_positions,
            columns=['x1', 'y1', 'x2', 'y2']
        )


        if df_ball_positions.isna().all().all():
            return []

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        df_ball_positions = df_ball_positions.ffill()

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(
            window=5, min_periods=1, center=False
        ).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 12

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = (
                    df_ball_positions['delta_y'].iloc[i] > 0 and
                    df_ball_positions['delta_y'].iloc[i + 1] < 0
            )
            positive_position_change = (
                    df_ball_positions['delta_y'].iloc[i] < 0 and
                    df_ball_positions['delta_y'].iloc[i + 1] > 0
            )

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = (
                            df_ball_positions['delta_y'].iloc[i] > 0 and
                            df_ball_positions['delta_y'].iloc[change_frame] < 0
                    )
                    positive_position_change_following_frame = (
                            df_ball_positions['delta_y'].iloc[i] < 0 and
                            df_ball_positions['delta_y'].iloc[change_frame] > 0
                    )

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        filtered_hits = []
        min_gap = 22

        for hit in frame_nums_with_ball_hits:
            if len(filtered_hits) == 0 or hit - filtered_hits[-1] >= min_gap:
                filtered_hits.append(hit)
            else:
                filtered_hits[-1] = hit

        return filtered_hits

    def detect_frames(
        self,
        frames,
        player_detections=None,
        left_line=None,
        right_line=None,
        read_from_stub=False,
        stub_path=None
    ):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        self.last_ball_center = None
        self.missed_frames = 0
        self.last_selected_center = None
        self.static_count = 0

        for i, frame in enumerate(frames):
            current_player_bboxes = []
            if player_detections is not None and i < len(player_detections):
                current_player_bboxes = list(player_detections[i].values())

            ball_dict = self.detect_frame(
                frame,
                player_bboxes=current_player_bboxes,
                left_line=left_line,
                right_line=right_line
            )
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame, player_bboxes=None, left_line=None, right_line=None):
        results = self.model.predict(frame, conf=0.25, verbose=False)[0]
        candidates = []

        for box in results.boxes:
            bbox = box.xyxy.tolist()[0]
            conf = float(box.conf.tolist()[0])

            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            area = w * h
            ratio = w / (h + 1e-6)
            center = self.get_center(bbox)

            if area < 16 or area > 500:
                continue

            if ratio < 0.5 or ratio > 2.0:
                continue

            if not self.is_inside_side_boundaries(center, left_line, right_line):
                continue

            if player_bboxes is not None and len(player_bboxes) > 0:
                if self.is_near_player_feet(center, player_bboxes):
                    continue

            if self.last_ball_center is not None:
                dist = math.hypot(
                    center[0] - self.last_ball_center[0],
                    center[1] - self.last_ball_center[1]
                )
                if dist > self.max_ball_jump:
                    continue

            candidates.append((bbox, conf, center))

        ball_dict = {}

        if not candidates:
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self.last_ball_center = None
            return ball_dict

        self.missed_frames = 0

        if self.last_ball_center is None:
            best = max(candidates, key=lambda x: x[1])
        else:
            def score(candidate):
                bbox, conf, center = candidate
                dist = math.hypot(
                    center[0] - self.last_ball_center[0],
                    center[1] - self.last_ball_center[1]
                )
                return conf - 0.003 * dist

            best = max(candidates, key=score)

        best_bbox, _, best_center = best

        if self.last_selected_center is not None:
            move_dist = math.hypot(
                best_center[0] - self.last_selected_center[0],
                best_center[1] - self.last_selected_center[1]
            )

            if move_dist < self.static_distance_threshold:
                self.static_count += 1
            else:
                self.static_count = 0
        else:
            self.static_count = 0

        self.last_selected_center = best_center

        if self.static_count >= self.max_static_frames:
            return {}

        ball_dict[1] = best_bbox
        self.last_ball_center = best_center

        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame, f"Ball: {track_id}",
                    (int(x1), int(y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames