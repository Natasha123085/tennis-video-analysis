from ultralytics import YOLO 
import cv2
import pickle
import sys
import math
import torch
sys.path.append('../')
from ..utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        xs = court_keypoints[::2]
        ys = court_keypoints[1::2]

        court_left = min(xs)
        court_right = max(xs)
        court_top = min(ys)
        court_bottom = max(ys)

        court_center_x = (court_left + court_right) / 2
        court_mid_y = (court_top + court_bottom) / 2

        margin_x = 100
        margin_top = 60
        margin_bottom = 150

        top_target = (court_center_x, court_top + (court_mid_y - court_top) * 0.75)
        bottom_target = (court_center_x, court_bottom - (court_bottom - court_mid_y) * 0.25)

        prev_top_bbox = None
        prev_bottom_bbox = None

        filtered_player_detections = []

        for player_dict in player_detections:
            top_candidates = []
            bottom_candidates = []

            for track_id, bbox in player_dict.items():
                foot_x, foot_y = self.get_foot_position(bbox)

                if foot_x < court_left - margin_x or foot_x > court_right + margin_x:
                    continue
                if foot_y < court_top - margin_top or foot_y > court_bottom + margin_bottom:
                    continue

                if foot_y < court_mid_y:
                    top_candidates.append((track_id, bbox))
                else:
                    bottom_candidates.append((track_id, bbox))

            filtered_player_dict = {}


            if top_candidates:
                if prev_top_bbox is not None:
                    top_best = min(
                        top_candidates,
                        key=lambda item: self.bbox_distance(item[1], prev_top_bbox)
                    )
                    top_best_dist = self.bbox_distance(top_best[1], prev_top_bbox)
                    if top_best_dist < 150:
                        filtered_player_dict[1] = top_best[1]
                        prev_top_bbox = top_best[1]
                    else:
                        filtered_player_dict[1] = prev_top_bbox
                else:
                    top_best = min(
                        top_candidates,
                        key=lambda item: self.point_distance(
                            self.get_foot_position(item[1]),
                            top_target
                        )
                    )
                    filtered_player_dict[1] = top_best[1]
                    prev_top_bbox = top_best[1]
            elif prev_top_bbox is not None:
                filtered_player_dict[1] = prev_top_bbox

            if bottom_candidates:
                if prev_bottom_bbox is not None:
                    bottom_best = min(
                        bottom_candidates,
                        key=lambda item: self.bbox_distance(item[1], prev_bottom_bbox)
                    )
                    bottom_best_dist = self.bbox_distance(bottom_best[1], prev_bottom_bbox)

                    if bottom_best_dist < 150:
                        filtered_player_dict[2] = bottom_best[1]
                        prev_bottom_bbox = bottom_best[1]
                    else:
                        filtered_player_dict[2] = prev_bottom_bbox
                else:
                    bottom_best = min(
                        bottom_candidates,
                        key=lambda item: self.point_distance(
                            self.get_foot_position(item[1]),
                            bottom_target
                        )
                    )
                    filtered_player_dict[2] = bottom_best[1]
                    prev_bottom_bbox = bottom_best[1]
            elif prev_bottom_bbox is not None:
                filtered_player_dict[2] = prev_bottom_bbox

            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections
    def get_foot_position(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y2)

    def get_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def point_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def bbox_distance(self, bbox1, bbox2):
        c1 = self.get_bbox_center(bbox1)
        c2 = self.get_bbox_center(bbox2)
        return self.point_distance(c1, c2)
    def choose_players(self, court_keypoints, player_dict):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0,len(court_keypoints),2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))
        
        # sort the distances in ascending order
        distances.sort(key = lambda x: x[1])
        # Choose the first 2 tracks
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players


    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame_num, frame in enumerate(frames):
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

            if frame_num % 50 == 0:
                torch.cuda.empty_cache()

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            imgsz=640,
            device="cuda",
            verbose=False
        )[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            if box.id is None:
                continue

            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict

    def draw_bboxes(self,video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        
        return output_video_frames


    