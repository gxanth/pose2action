import glob
import pathlib

import cv2
import mediapipe as mp
from numpy import ndarray
import pandas as pd



def extract_video(fp: str, label: str) -> pd.DataFrame:
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(fp)
    seq = []
    with mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        static_image_mode=False) as pose:
        f = 0
        while True:
            ok:bool , frame:ndarray = cap.read() # ok: bool, frame: np.ndarray
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                for j, lm in enumerate(res.pose_landmarks.landmark):
                    seq.append((f, j, lm.x, lm.y, lm.z, lm.visibility, label, fp))
            f += 1
    cap.release()
    return pd.DataFrame(
        seq, columns=["frame", "joint", "x", "y", "z", "vis", "label", "video"]
    )


def main(in_dir="data/raw_videos", out_dir="data/keypoints"):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = glob.glob(f"{in_dir}/*.mp4")
    all_ = []
    for fp in files:
        label = pathlib.Path(fp).stem.split("_", 1)[1]
        df = extract_video(fp, label)
        df.to_parquet(out / (pathlib.Path(fp).stem + ".parquet"))
        all_.append(df)
    if all_:
        pd.concat(all_).to_parquet(out / "_all.parquet")
    print("Done. Wrote keypoints to", out)


if __name__ == "__main__":
    main()
