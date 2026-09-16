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