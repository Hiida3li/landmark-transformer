"""Record labeled gesture sequences from the webcam and save them as .npz files."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.landmarks import HandLandmarkExtractor, draw_landmarks


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gesture", required=True, help="label, e.g. peace")
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config()
    if args.gesture not in cfg["gestures"]:
        raise ValueError(f"'{args.gesture}' not in config gestures: {cfg['gestures']}")

    seq_len = cfg["seq_len"]
    out_dir = Path(cfg["data_dir"]) / args.gesture
    out_dir.mkdir(parents=True, exist_ok=True)
    session = time.strftime("%Y%m%d_%H%M%S")            # groups this run's samples

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(0)
    saved = 0

    while saved < args.samples:
        # --- countdown: give the user time to pose ---
        for remaining in (3, 2, 1):
            t_end = time.time() + 1.0
            while time.time() < t_end:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"{args.gesture}  sample {saved+1}/{args.samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"recording in {remaining}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
                cv2.imshow("Collect", frame)
                cv2.waitKey(1)

        # --- record seq_len consecutive frames ---
        frames: list[np.ndarray] = []
        dropped = False
        while len(frames) < seq_len:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            lms = extractor.extract(frame)               # (21, 3) or None
            if lms is None:
                dropped = True
                break                                    # hand lost -> discard sample
            frames.append(lms)
            draw_landmarks(frame, lms)
            cv2.putText(frame, f"REC {len(frames)}/{seq_len}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Collect", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release(); cv2.destroyAllWindows(); return

        if dropped:
            print("hand lost mid-sample, retrying")
            continue

        sample = np.stack(frames)                        # (seq_len, 21, 3)
        assert sample.shape == (seq_len, 21, 3), sample.shape
        np.savez(out_dir / f"{session}_{saved:03d}.npz",
                 landmarks=sample, label=args.gesture, session=session)
        saved += 1
        print(f"saved {saved}/{args.samples}  shape={sample.shape}")

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()