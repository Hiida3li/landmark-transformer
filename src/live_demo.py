"""Real-time gesture prediction: webcam -> landmarks -> Transformer -> label on screen."""

from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.landmarks import HandLandmarkExtractor, draw_landmarks
from src.model import LandmarkTransformer
from src.normalize import normalize_sequence, flatten_sequence

CKPT = "models/transformer_best.pt"


def main() -> None:
    cfg = yaml.safe_load(open("configs/config.yaml"))
    gestures, seq_len = cfg["gestures"], cfg["seq_len"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = LandmarkTransformer(n_classes=len(gestures), seq_len=seq_len).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()                                       # dropout off for inference

    buffer: deque = deque(maxlen=seq_len)              # rolling window of last 30 frames
    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        lms = extractor.extract(frame)                 # (21, 3) or None
        if lms is not None:
            buffer.append(lms)
            draw_landmarks(frame, lms)
        else:
            buffer.clear()                             # hand gone -> start a fresh window

        label, conf = "...", 0.0
        if len(buffer) == seq_len:
            seq = np.stack(buffer)                     # (30, 21, 3)
            seq = flatten_sequence(normalize_sequence(seq))          # (30, 63)
            x = torch.from_numpy(seq).unsqueeze(0).to(device)        # (1, 30, 63)
            with torch.no_grad():
                probs = F.softmax(model(x), dim=1)[0]                # (5,)
            idx = int(probs.argmax())
            label, conf = gestures[idx], float(probs[idx])

        color = (0, 255, 0) if conf > 0.8 else (0, 165, 255)
        cv2.putText(frame, f"{label} {conf:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"buffer {len(buffer)}/{seq_len}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Live demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()


if __name__ == "__main__":
    main()