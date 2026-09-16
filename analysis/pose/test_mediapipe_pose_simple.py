import cv2
import mediapipe as mp

IMAGE_PATH = r"D:\tennis_project\analysis\input_videos\pose1.webp"
OUTPUT_PATH = r"D:\tennis_project\analysis\input_videos\pose1_result.webp"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found")
    exit()

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.3
) as pose:

    results = pose.process(rgb)

    if results.pose_landmarks:
        print("Pose detected!")

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
    else:
        print("No pose detected")

cv2.imwrite(OUTPUT_PATH, image)
print("Saved to:", OUTPUT_PATH)