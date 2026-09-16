import cv2
import mediapipe as mp
from analysis.pose.pose_utils import get_player_pose_data

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

def get_pose_keypoints_for_player(frame, bbox, pose_model):
    x1, y1, x2, y2 = enlarge_bbox(bbox, frame.shape, scale_x=0.15, scale_y=0.15)
    player_crop = frame[y1:y2, x1:x2]

    if player_crop.size == 0:
        return None, None, None

    rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb_crop)

    if not results.pose_landmarks:
        return None, None, None

    keypoints = get_player_pose_data(results, (x1, y1, x2, y2))
    return keypoints, results.pose_landmarks, (x1, y1, x2, y2)