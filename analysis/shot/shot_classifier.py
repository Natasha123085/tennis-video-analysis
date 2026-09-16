def classify_shot_sequence_5class(
    prev_kp,
    curr_kp,
    next_kp,
    ball_bbox,
    player_bbox,
    frame_shape,
    hitter_id,
    hit_index_in_rally=None
):
    def pt(kp, name):
        return kp[name]["x"], kp[name]["y"]

    def vis(kp, name):
        return kp[name]["visibility"]

    left_shoulder = pt(curr_kp, "left_shoulder")
    right_shoulder = pt(curr_kp, "right_shoulder")
    left_wrist = pt(curr_kp, "left_wrist")
    right_wrist = pt(curr_kp, "right_wrist")
    left_elbow = pt(curr_kp, "left_elbow")
    right_elbow = pt(curr_kp, "right_elbow")
    left_hip = pt(curr_kp, "left_hip")
    right_hip = pt(curr_kp, "right_hip")

    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2

    bx1, by1, bx2, by2 = ball_bbox
    ball_x = (bx1 + bx2) / 2
    ball_y = (by1 + by2) / 2

    px1, py1, px2, py2 = player_bbox
    frame_h, frame_w = frame_shape[:2]
    foot_y = py2
    player_height = py2 - py1

    # distances to ball
    left_dist = ((left_wrist[0] - ball_x) ** 2 + (left_wrist[1] - ball_y) ** 2) ** 0.5
    right_dist = ((right_wrist[0] - ball_x) ** 2 + (right_wrist[1] - ball_y) ** 2) ** 0.5

    # arm extension: shoulder -> wrist
    left_arm_len = ((left_wrist[0] - left_shoulder[0]) ** 2 + (left_wrist[1] - left_shoulder[1]) ** 2) ** 0.5
    right_arm_len = ((right_wrist[0] - right_shoulder[0]) ** 2 + (right_wrist[1] - right_shoulder[1]) ** 2) ** 0.5

    # choose hitting side more robustly:
    # prefer visible + closer + more extended arm
    left_score = 0
    right_score = 0

    if vis(curr_kp, "left_wrist") > 0.5:
        left_score += 1
    if vis(curr_kp, "right_wrist") > 0.5:
        right_score += 1

    if left_dist < right_dist:
        left_score += 1
    else:
        right_score += 1

    if left_arm_len > right_arm_len:
        left_score += 1
    else:
        right_score += 1

    hitting_side = "left" if left_score > right_score else "right"

    prev_rw = pt(prev_kp, "right_wrist")
    next_rw = pt(next_kp, "right_wrist")
    prev_lw = pt(prev_kp, "left_wrist")
    next_lw = pt(next_kp, "left_wrist")

    right_motion_x = next_rw[0] - prev_rw[0]
    left_motion_x = next_lw[0] - prev_lw[0]

    wrist_gap = abs(right_dist - left_dist)
    motion_strength = max(abs(right_motion_x), abs(left_motion_x))

    near_net = frame_h * 0.35 < foot_y < frame_h * 0.7
    near_baseline_bottom = foot_y > frame_h * 0.80
    near_baseline_top = foot_y < frame_h * 0.20
    near_baseline = near_baseline_bottom or near_baseline_top

    clearly_above_shoulders = ball_y < (shoulder_center_y - 12)
    between_shoulder_and_hip = shoulder_center_y <= ball_y <= hip_center_y

    is_possible_server_position = (
        (hitter_id == 1 and near_baseline_top) or
        (hitter_id == 2 and near_baseline_bottom)
    )

    if (
        hit_index_in_rally is not None and hit_index_in_rally == 0
        and clearly_above_shoulders
        and is_possible_server_position
        and player_height > 140
    ):
        confidence = 0.75
        if wrist_gap > 15:
            confidence += 0.05
        return "serve_candidate", round(min(confidence, 0.95), 2)

    if clearly_above_shoulders and not near_baseline and not near_net:
        confidence = 0.70
        if motion_strength > 8:
            confidence += 0.05
        return "smash_candidate", round(min(confidence, 0.9), 2)

    if near_net and between_shoulder_and_hip:
        return "volley_candidate", 0.68


    # 4. Groundstroke classification
    # Use ball side relative to torso as the main signal

    if hitter_id == 2:  # bottom player
        if ball_x > shoulder_center_x:
            return "forehand_like", 0.78 if wrist_gap > 10 else 0.68
        else:
            return "backhand_like", 0.78 if wrist_gap > 10 else 0.68

    else:  # top player
        if ball_x < shoulder_center_x:
            return "forehand_like", 0.78 if wrist_gap > 10 else 0.68
        else:
            return "backhand_like", 0.78 if wrist_gap > 10 else 0.68