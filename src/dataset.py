import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

JOINTS = 33


def _sample_frames(frames, T=32):
    idx = np.linspace(0, len(frames) - 1, T).round().astype(int)
    return [frames[i] for i in idx]


class KeypointSeqDataset(Dataset):
    def __init__(self, keypoint_dir="data/keypoints", T=32, classes=None):
        self.dir = pathlib.Path(keypoint_dir)
        self.files = sorted(
            [p for p in self.dir.glob("*.parquet") if not p.name.startswith("_")]
        )
        self.T = T
        self.classes = classes or sorted({p.stem.split("_", 1)[1] for p in self.files})
        self.cls2id = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        df = pd.read_parquet(p)
        frames = sorted(df.frame.unique())
        if len(frames) == 0:
            raise RuntimeError(f"No frames in {p}")
        frames = _sample_frames(frames, self.T)
        seq = []
        for f in frames:
            fdf = df[df.frame == f].sort_values("joint")
            xy = fdf[["x", "y"]].values.reshape(-1)  # 66
            seq.append(xy)
        x = np.stack(seq, 0).astype("float32")  # (T, 66)
        label = df.label.iloc[0]
        y = self.cls2id[label]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
