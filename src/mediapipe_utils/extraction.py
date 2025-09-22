import sys

import cv2
import mediapipe as mp
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .drawing import draw_landmarks_on_image
from .utils import flatten_mp_landmarks


def extract_video(fp: str, label: str) -> pd.DataFrame:
    """
    Extract pose landmarks from a video file using MediaPipe's PoseLandmarker.
    Args:
        fp (str): File path to the video.
        label (str): Label associated with the video.
    Returns:
        pd.DataFrame: DataFrame containing pose landmarks for each frame.
    """
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path="models/pose_landmarker_lite.task"
        ),
        running_mode=vision.RunningMode.VIDEO,
        min_tracking_confidence=0.5,
        min_pose_detection_confidence=0.5,
    )
    landmarks_seq = []
    with vision.PoseLandmarker.create_from_options(pose_options) as landmarker:
        video_capture = cv2.VideoCapture(fp)
        if not video_capture.isOpened():
            print(f"Error: Could not open video file {fp}", file=sys.stderr)
            return pd.DataFrame()
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_number = 0
        while True:
            ret, frame = video_capture.read()
            if not ret or frame is None or frame.size == 0:
                break
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            frame_timestamp_ms = int((frame_number / fps) * 1000)
            pose_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            landmarks_seq.extend(
                flatten_mp_landmarks(
                    pose_result.pose_landmarks, frame_number, label, fp
                )
            )
            # Optional: Draw landmarks on the frame for visualization
            annotated_frame = draw_landmarks_on_image(frame, pose_result.pose_landmarks)
            cv2.imshow("Annotated Frame", annotated_frame)
            cv2.waitKey(1)
            frame_number += 1

        video_capture.release()
        cv2.destroyAllWindows()
    df_landmarks = pd.DataFrame(
        landmarks_seq,
        columns=["frame", "joint", "x", "y", "z", "visibility", "label", "video"],
    )
    return df_landmarks
