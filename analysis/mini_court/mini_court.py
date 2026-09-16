import cv2
import numpy as np
import sys

sys.path.append('../')
from ..import constants
from ..utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self,frame):
        self.frame_height, self.frame_width = frame.shape[:2]
        self.scale = self.frame_width / 1920
        self.drawing_rectangle_width = int(250 * self.scale)
        self.drawing_rectangle_height = int(
            self.drawing_rectangle_width * (constants.REAL_COURT_HEIGHT / constants.DOUBLE_LINE_WIDTH)
        )
        self.buffer = int(50 * self.scale)
        self.padding_court = int(20 * self.scale)
        self.homography_matrix = None

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

        print("MiniCourt NEW VERSION LOADED")


    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.drawing_rectangle_width - 2 * self.padding_court
        self.court_drawing_height = self.drawing_rectangle_height - 2 * self.padding_court

    def set_canvas_background_box_position(self,frame):
        h, w = frame.shape[:2]

        self.end_x = w - int(20 * self.scale)
        self.start_x = self.end_x - self.drawing_rectangle_width

        self.start_y = int(50 * self.scale)
        self.end_y = self.start_y + self.drawing_rectangle_height

    def draw_court(self,frame):
        radius = max(2, int(5 * self.scale))
        thickness = max(1, int(2 * self.scale))
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y), radius, (0,0,255), -1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0,0,0), thickness)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), thickness)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    
    def get_height_of_mini_court(self):
        return self.court_drawing_height
    
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points
    
    def get_baselines(self):
        top_baseline = self.court_start_y
        bottom_baseline = self.court_end_y
        return top_baseline, bottom_baseline
    
    def get_center_x(self):
        return self.court_start_x + self.court_drawing_width/2
    
    def get_singles_lines(self):
        margin_ratio = (constants.DOUBLE_LINE_WIDTH - constants.SINGLE_LINE_WIDTH) / 2 / constants.DOUBLE_LINE_WIDTH
        margin_pixels = self.court_drawing_width * margin_ratio
        left = self.court_start_x + margin_pixels
        right = self.court_end_x - margin_pixels
        return left, right

    def compute_homography(self, original_court_keypoints):
        src_pts = np.array([
            [original_court_keypoints[0], original_court_keypoints[1]],   # upper left
            [original_court_keypoints[2], original_court_keypoints[3]],   # upper right
            [original_court_keypoints[4], original_court_keypoints[5]],   # down left
            [original_court_keypoints[6], original_court_keypoints[7]]    # down right
        ], dtype=np.float32)

        dst_pts = np.array([
            [self.drawing_key_points[0], self.drawing_key_points[1]],   # upper left
            [self.drawing_key_points[2], self.drawing_key_points[3]],   # upper right
            [self.drawing_key_points[4], self.drawing_key_points[5]],   # down left
            [self.drawing_key_points[6], self.drawing_key_points[7]]    # down right
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)


    def map_point_to_mini_court(self, point):
        if self.homography_matrix is None:
            return point
        px = np.array([[point]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(px, self.homography_matrix)
        return tuple(mapped[0][0])

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):

        if self.homography_matrix is None:
            self.compute_homography(original_court_key_points)

        output_player_boxes = []
        output_ball_boxes = []

        for frame_num, player_bbox in enumerate(player_boxes):
            output_player_bboxes_dict = {}
            output_ball_dict = {}

            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)
                mini_court_player_position = self.map_point_to_mini_court(foot_position)
                output_player_bboxes_dict[player_id] = mini_court_player_position

            ball_box = ball_boxes[frame_num].get(1, None)

            if ball_box is None:
                if len(output_ball_boxes) > 0 and 1 in output_ball_boxes[-1]:
                    output_ball_dict[1] = output_ball_boxes[-1][1]
            else:
                ball_position = get_center_of_bbox(ball_box)
                mini_court_ball_position = self.map_point_to_mini_court(ball_position)
                output_ball_dict[1] = mini_court_ball_position

            output_player_boxes.append(output_player_bboxes_dict)
            output_ball_boxes.append(output_ball_dict)

        return output_player_boxes, output_ball_boxes

    def draw_points_on_mini_court(self, output_video_frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(output_video_frames):
            if frame_num >= len(positions):
                continue

            if positions[frame_num] is None:
                continue

            for _, position in positions[frame_num].items():
                x = int(position[0])
                y = int(position[1])
                cv2.circle(frame, (x, y), 5, color, -1)

        return output_video_frames
    
    def get_court_zones(self):
        width = self.get_width_of_mini_court()
        height = self.get_height_of_mini_court()

        start_x = self.court_start_x
        start_y = self.court_start_y

        zones = {
            "left": (start_x, start_x + width / 3),
            "middle": (start_x + width / 3, start_x + 2 * width / 3),
            "right": (start_x + 2 * width / 3, start_x + width),

            "short": (start_y, start_y + height / 3),
            "mid": (start_y + height / 3, start_y + 2 * height / 3),
            "deep": (start_y + 2 * height / 3, start_y + height)
        }
        return zones
