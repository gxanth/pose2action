import glob
import pathlib
import sys

import pandas as pd

from .extraction import extract_video


def main(in_dir="data/raw_videos", out_dir="data/keypoints"):
    """
    Orchestrate extraction of pose landmarks for all videos in a directory.
    """
    output_path = pathlib.Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_files = glob.glob(f"{in_dir}/*.mp4")
    all_landmarks_dfs = []

    if not video_files:
        print(f"No video files found in {in_dir}", file=sys.stderr)
        return

    for video_path in video_files:
        try:
            label = pathlib.Path(video_path).stem.split("_", 1)[1]
        except IndexError:
            print(
                f"Skipping video file with invalid naming convention: {video_path}",
                file=sys.stderr,
            )
            continue
        df_landmarks = extract_video(video_path, label)
        if not df_landmarks.empty:
            df_landmarks.to_parquet(
                output_path / (pathlib.Path(video_path).stem + ".parquet")
            )
            all_landmarks_dfs.append(df_landmarks)
    if all_landmarks_dfs:
        pd.concat(all_landmarks_dfs).to_parquet(output_path / "_all.parquet")
    print("Done. Wrote keypoints to", output_path)
