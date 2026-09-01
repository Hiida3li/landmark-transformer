"""Webcam capture loop: reads frames, mirrors them, shows live FPS."""

import time

import cv2

WIDTH, HEIGHT = 640, 480


def main() -> None:
    cap = cv2.VideoCapture(0)                       # 0 = default (built-in) camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)        # request a smaller resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Camera could not be opened. Check permissions.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()                     # frame: numpy array (H, W, 3), BGR
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)                  # 1 = horizontal flip (mirror)

        # --- FPS measurement ---
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )

        cv2.imshow("Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):       # press q to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()