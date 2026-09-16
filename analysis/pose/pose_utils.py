def extract_keypoints(pose_landmarks, bbox):
    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1

    lm = pose_landmarks.landmark

    def get_point(index):
        return {
            "x": int(lm[index].x * crop_w) + x1,
            "y": int(lm[index].y * crop_h) + y1,
            "visibility": float(getattr(lm[index], "visibility", 1.0))
        }

    return {
        "left_shoulder": get_point(11),
        "right_shoulder": get_point(12),
        "left_elbow": get_point(13),
        "right_elbow": get_point(14),
        "left_wrist": get_point(15),
        "right_wrist": get_point(16),
        "left_hip": get_point(23),
        "right_hip": get_point(24),
    }


def get_player_pose_data(results, bbox):
    if not results.pose_landmarks:
        return None
    return extract_keypoints(results.pose_landmarks, bbox)