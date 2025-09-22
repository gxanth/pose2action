import glob
import pathlib

import cv2

# Core MediaPipe Tasks API Imports
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

# Visualization Utilities


def extract_video(fp: str, label: str) -> pd.DataFrame:
    """
    Extract pose landmarks from a video file using MediaPipe's PoseLandmarker.

    Args:
        fp (str): File path to the video.
        label (str): Label associated with the video.

    Returns:
        pd.DataFrame: DataFrame containing pose landmarks for each frame.
    """
    BaseOptions = mp.tasks.BaseOptions
    # Configure PoseLandmarker options
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/pose_landmarker_lite.task"),
        running_mode=RunningMode.VIDEO,
        min_tracking_confidence=0.5,
        min_pose_detection_confidence=0.5,
    )

    # Initialize sequence to store flattened landmark data
    landmarks_seq: list[list] = []

    # Process video with PoseLandmarker
    with PoseLandmarker.create_from_options(pose_options) as landmark_detector:
        video_capture = cv2.VideoCapture(fp)
        if not video_capture.isOpened():
            print(f"Error: Could not open video file {fp}")
            return pd.DataFrame()

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_number = 0

        while True:
            ret, frame = video_capture.read()  # bool, ndarray
            if not ret or frame is None:
                break
            # cv2.imshow("frame", frame)
            # cv2.waitKey(1)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            frame_timestamp_ms = int((frame_number / fps) * 1000)

            pose_result = landmark_detector.detect_for_video(
                mp_image, frame_timestamp_ms
            )

            # Flatten landmarks: one row per landmark
            if pose_result and pose_result.pose_landmarks:
                for joint_idx, lm in enumerate(pose_result.pose_landmarks[0][11:]):
                    landmarks_seq.append(
                        [
                            frame_number,  # frame
                            joint_idx,  # joint
                            lm.x,
                            lm.y,
                            lm.z,
                            getattr(lm, "visibility", None),
                            label,
                            fp,
                        ]
                    )

            frame_number += 1
            annotated_image = draw_landmarks_on_image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                pose_result.pose_landmarks,
            )
            cv2.imshow("annotated", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        video_capture.release()
        cv2.destroyAllWindows()

    # Convert landmarks to DataFrame
    df_landmarks = pd.DataFrame(
        landmarks_seq,
        columns=["frame", "joint", "x", "y", "z", "visibility", "label", "video"],
    )
    return df_landmarks


def draw_landmarks_on_image(rgb_image: np.ndarray, pose_landmarks_list: list):
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def main(in_dir="data/raw_videos", out_dir="data/keypoints"):
    output_path: pathlib.Path = pathlib.Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_files: list[str] = glob.glob(f"{in_dir}/*.mp4")
    all_landmarks_dfs: list[pd.DataFrame] = []
    for video_path in video_files:
        label: str = pathlib.Path(video_path).stem.split("_", 1)[1]
        df_landmarks: pd.DataFrame = extract_video(video_path, label)
        df_landmarks.to_parquet(
            output_path / (pathlib.Path(video_path).stem + ".parquet")
        )
        all_landmarks_dfs.append(df_landmarks)
    if all_landmarks_dfs:
        pd.concat(all_landmarks_dfs).to_parquet(output_path / "_all.parquet")
    print("Done. Wrote keypoints to", output_path)


if __name__ == "__main__":
    main()
