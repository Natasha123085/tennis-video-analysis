import numpy as np
import cv2

def draw_player_stats(output_video_frames, player_stats, mini_court=None):

    for index, row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        frame = output_video_frames[index]
        scale = frame.shape[1] / 1920

        width = max(320, int(420 * scale))
        height = max(190, int(250 * scale))

        # 默认右下角
        start_x = frame.shape[1] - width - 20
        start_y = frame.shape[0] - height - 20

        # 避开 MiniCourt
        if mini_court is not None:
            end_x = start_x + width
            end_y = start_y + height

            overlap_x = not (start_x > mini_court.end_x or end_x < mini_court.start_x)
            overlap_y = not (start_y > mini_court.end_y or end_y < mini_court.start_y)

            if overlap_x and overlap_y:
                start_y = int(mini_court.start_y - height - 20)

        # 防止出屏幕
        margin = 20
        if start_x < margin:
            start_x = margin
        if start_x + width > frame.shape[1] - margin:
            start_x = frame.shape[1] - width - margin

        if start_y < margin:
            start_y = margin
        if start_y + height > frame.shape[0] - margin:
            start_y = frame.shape[0] - height - margin

        end_x = start_x + width
        end_y = start_y + height

        # 半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 字体参数
        font_header = max(0.45, 0.6 * scale)
        font_label = max(0.35, 0.48 * scale)
        font_value = max(0.35, 0.46 * scale)

        thick_header = max(1, int(2 * scale))
        thick_text = 1

        # 列位置
        label_x = start_x + int(14 * scale)
        p1_x = start_x + int(width * 0.52)
        p2_x = start_x + int(width * 0.78)

        # 行位置
        header_y = start_y + int(28 * scale)
        line_y = start_y + int(42 * scale)

        row1_y = start_y + int(78 * scale)
        row2_y = start_y + int(114 * scale)
        row3_y = start_y + int(150 * scale)
        row4_y = start_y + int(186 * scale)

        # 标题
        cv2.putText(frame, "Player Stats", (label_x, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_header, (255, 255, 255), thick_header)

        # 表头
        cv2.putText(frame, "P1", (p1_x, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (255, 255, 255), thick_text)
        cv2.putText(frame, "P2", (p2_x, header_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (255, 255, 255), thick_text)

        # 分隔线
        cv2.line(frame, (start_x + int(10 * scale), line_y),
                 (end_x - int(10 * scale), line_y), (180, 180, 180), 1)

        # 行标签
        cv2.putText(frame, "Shot Speed", (label_x, row1_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (255, 255, 255), thick_text)
        cv2.putText(frame, "Player Speed", (label_x, row2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (255, 255, 255), thick_text)
        cv2.putText(frame, "Avg Shot", (label_x, row3_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (255, 255, 255), thick_text)
        cv2.putText(frame, "Avg Player", (label_x, row4_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (255, 255, 255), thick_text)

        # 数值列
        cv2.putText(frame, f"{player_1_shot_speed:.1f}", (p1_x, row1_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_value, (255, 255, 255), thick_text)
        cv2.putText(frame, f"{player_2_shot_speed:.1f}", (p2_x, row1_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_value, (255, 255, 255), thick_text)

        cv2.putText(frame, f"{player_1_speed:.1f}", (p1_x, row2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_value, (255, 255, 255), thick_text)
        cv2.putText(frame, f"{player_2_speed:.1f}", (p2_x, row2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_value, (255, 255, 255), thick_text)

        cv2.putText(frame, f"{avg_player_1_shot_speed:.1f}", (p1_x, row3_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_value, (255, 255, 255), thick_text)
        cv2.putText(frame, f"{avg_player_2_shot_speed:.1f}", (p2_x, row3_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_value, (255, 255, 255), thick_text)

        cv2.putText(frame, f"{avg_player_1_speed:.1f}", (p1_x, row4_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_value, (255, 255, 255), thick_text)
        cv2.putText(frame, f"{avg_player_2_speed:.1f}", (p2_x, row4_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_value, (255, 255, 255), thick_text)


        unit_x = end_x - int(42 * scale)
        cv2.putText(frame, "km/h", (unit_x, row1_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (180, 180, 180), 1)
        cv2.putText(frame, "km/h", (unit_x, row2_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (180, 180, 180), 1)
        cv2.putText(frame, "km/h", (unit_x, row3_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (180, 180, 180), 1)
        cv2.putText(frame, "km/h", (unit_x, row4_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_label, (180, 180, 180), 1)

        output_video_frames[index] = frame

    return output_video_frames