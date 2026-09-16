def smooth_positions(player_positions, alpha=0.7):
    smoothed = []
    prev_positions = {}

    for frame_data in player_positions:
        smoothed_frame = {}

        for player_id, pos in frame_data.items():

            if player_id not in prev_positions:
                smoothed_frame[player_id] = pos
            else:
                prev = prev_positions[player_id]
                x = alpha * pos[0] + (1 - alpha) * prev[0]
                y = alpha * pos[1] + (1 - alpha) * prev[1]
                smoothed_frame[player_id] = (x, y)

            prev_positions[player_id] = smoothed_frame[player_id]

        smoothed.append(smoothed_frame)

    return smoothed