# Utility functions for MediaPipe-based pose extraction and drawing


def get_video_fps(video_capture) -> float:
    """Get frames per second from a cv2.VideoCapture object."""
    return video_capture.get(5)  # cv2.CAP_PROP_FPS


def flatten_mp_landmarks(pose_landmarks, frame_number, label, fp):
    """
    Flattens MediaPipe pose_landmarks for a single frame into a list of rows.
    Each row: [frame_number, joint_idx, x, y, z, visibility, label, video]
    """
    rows = []
    if pose_landmarks:
        for joint_idx, lm in enumerate(pose_landmarks[0]):
            rows.append(
                [
                    frame_number,
                    joint_idx,
                    lm.x,
                    lm.y,
                    lm.z,
                    getattr(lm, "visibility", None),
                    label,
                    fp,
                ]
            )
    return rows


# Add more helpers as needed
