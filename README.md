# Pose2Action

Mini-project: classify simple actions from webcam video using pose keypoints.

## Pipeline
1. Record short labeled clips from webcam.
2. Extract pose keypoints with MediaPipe.
3. Train a temporal model (MLP/LSTM/TCN).
4. Evaluate, explain, and generate report.

## Commands
```bash
python scripts/record.py --label sit
python scripts/extract_pose.py
python scripts/train_model.py --config configs/baseline.yaml
```
