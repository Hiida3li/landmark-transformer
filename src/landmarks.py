"""Wraps MediaPipe HandLandmarker: BGR frame in -> (21, 3) numpy array out."""

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "models/hand_landmarker.task"

# Which landmark indices connect to form the skeleton (for drawing only).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (0, 9), (9, 10), (10, 11), (11, 12),       # middle
    (0, 13), (13, 14), (14, 15), (15, 16),     # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm
]


class HandLandmarkExtractor:
    def __init__(self, model_path: str = MODEL_PATH) -> None:
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def extract(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        """Return landmarks as float32 array of shape (21, 3), or None if no hand."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)      # MediaPipe wants RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self._timestamp_ms += 33                               # ~30 fps; must increase
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.hand_landmarks:                          # empty list = no hand
            return None

        hand = result.hand_landmarks[0]                        # first (only) hand
        return np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> None:
    """Draw the 21 points and skeleton lines onto a BGR frame (in place)."""
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in landmarks]   # normalized -> pixels
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 255, 0), -1)


if __name__ == "__main__":
    # Quick visual test: webcam + skeleton overlay + landmark shape in terminal.
    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        lms = extractor.extract(frame)
        if lms is not None:
            draw_landmarks(frame, lms)
            print("landmarks:", lms.shape, "wrist xyz:", lms[0].round(3))
        cv2.imshow("Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    extractor.close()