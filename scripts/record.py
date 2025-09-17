import argparse
import pathlib
import time

import cv2


def main(label: str, seconds: int = 4, fps: int = 20, out_dir: str = "data/raw_videos"):
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No webcam found.")
    w, h = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fname = out / f"{int(time.time())}_{label}.mp4"
    vw = cv2.VideoWriter(str(fname), fourcc, fps, (w, h))
    start = time.time()
    while time.time() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.putText(
            frame, f"REC {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        vw.write(frame)
        cv2.imshow("record", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    vw.release()
    cv2.destroyAllWindows()
    print("Saved:", fname)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--seconds", type=int, default=4)
    args = ap.parse_args()
    main(args.label, args.seconds)
