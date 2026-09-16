import cv2
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"[ERROR] Cannot open video: {video_path}")

    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Start reading video: {video_path}")
    print(f"[INFO] Total frames (reported): {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            if total_frames > 0:
                print(f"[INFO] Read {frame_count}/{total_frames} frames")
            else:
                print(f"[INFO] Read {frame_count} frames")

    cap.release()
    print(f"[INFO] Finished reading video. Total frames read: {frame_count}")
    return frames

def save_video(output_video_frames, output_video_path):
    height, width, _ = output_video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))

    for frame in output_video_frames:
        out.write(frame)

    out.release()