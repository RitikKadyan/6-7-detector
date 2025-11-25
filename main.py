import cv2
import mediapipe as mp
import imageio
import time
from collections import deque
import numpy as np
import tkinter as tk

# ------------------------------
# Detect Full Monitor Resolution
# ------------------------------
def get_screen_size():
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h

screen_w, screen_h = get_screen_size()

# ------------------------------
# MediaPipe Hand Setup
# ------------------------------
mp_hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.67,
    min_tracking_confidence=0.67
)

cap = cv2.VideoCapture(0)

# ------------------------------
# Load GIF
# ------------------------------
gif = imageio.mimread("67.gif")
gif_frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR) for frame in gif]

# ------------------------------
# Motion Buffers
# ------------------------------
buffer_left = deque(maxlen=5)
buffer_right = deque(maxlen=5)
trail_left = deque(maxlen=10)
trail_right = deque(maxlen=10)

JUMP_THRESHOLD = 0.06
HALF_THRESHOLD = JUMP_THRESHOLD * 0.5

show_gif = False
gif_start = 0
gif_index = 0

# FPS tracking
prev_time = time.time()
fps = 0
last_fps_report = 0

# Hand tracking logs
prev_hand_count = 0

# ------------------------------
# Fullscreen Window
# ------------------------------
cv2.namedWindow("6-7 Detector", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("6-7 Detector", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setUseOptimized(True)
cv2.setNumThreads(8)

# ------------------------------
# Main Loop
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera.")
        break

    cam_h, cam_w, _ = frame.shape

    # 2× wide canvas: LEFT = camera, RIGHT = GIF
    out_w = cam_w * 2
    out_h = cam_h
    output = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    output[:, :cam_w] = frame

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Only log FPS if changed significantly
    if abs(fps - last_fps_report) > 5:
        print(f"[FPS] {fps:.1f}")
        last_fps_report = fps

    # Run MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(rgb)

    left_y = None
    right_y = None
    left_pt = None
    right_pt = None

    # ------------------------------
    # Hand Detection Logging
    # ------------------------------
    hand_count = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
    if hand_count != prev_hand_count:
        print(f"[HANDS] Detected {hand_count} hand(s)")
        prev_hand_count = hand_count

    # ------------------------------
    # Tracking Overlay
    # ------------------------------
    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):

            wrist = hand_landmarks.landmark[0]
            palm = hand_landmarks.landmark[9]

            wrist_px = (int(wrist.x * cam_w), int(wrist.y * cam_h))
            palm_px  = (int(palm.x * cam_w), int(palm.y * cam_h))

            # Palm-center point
            center_x = int((wrist_px[0] + palm_px[0]) / 2)
            center_y = int((wrist_px[1] + palm_px[1]) / 2)
            center_pt = (center_x, center_y)

            label = handedness.classification[0].label

            # LOG HAND POSITION OCCASIONALLY
            print(f"[{label}] Center pos = {center_pt}")

            # Add to motion trail
            if label == "Left":
                trail_left.append(center_pt)
            else:
                trail_right.append(center_pt)

            # Draw tracking points
            cv2.circle(output, wrist_px, 10, (0,255,0) if label=="Left" else (0,128,255), -1)
            cv2.circle(output, palm_px, 10, (0,255,0) if label=="Left" else (0,128,255), -1)
            cv2.circle(output, center_pt, 14, (255,0,0), -1)

            cv2.line(output, wrist_px, palm_px, (0,255,0) if label=="Left" else (0,128,255), 3)
            cv2.line(output, palm_px, center_pt, (255,0,0), 3)

            # Assign Y
            if label == "Left":
                left_y = center_y / cam_h
                left_pt = center_pt
            else:
                right_y = center_y / cam_h
                right_pt = center_pt

    # ------------------------------
    # Update motion buffers
    # ------------------------------
    if left_y is not None:
        buffer_left.append(left_y)

    if right_y is not None:
        buffer_right.append(right_y)

    # ------------------------------
    # FAST 6-7 Gesture Detection
    # ------------------------------
    if len(buffer_left) == 5 and len(buffer_right) == 5:

        left_jump = buffer_left[0] - buffer_left[-1]
        right_jump = buffer_right[0] - buffer_right[-1]

        # Log if movement is noticeable
        if abs(left_jump) > HALF_THRESHOLD or abs(right_jump) > HALF_THRESHOLD:
            print(f"[MOTION] LeftJump={left_jump:.3f} RightJump={right_jump:.3f}")

        gesture = False

        # Left up + right down
        if left_jump > JUMP_THRESHOLD and right_jump < -JUMP_THRESHOLD:
            print("[GESTURE] Left UP + Right DOWN detected")
            gesture = True

        # Right up + left down
        if right_jump > JUMP_THRESHOLD and left_jump < -JUMP_THRESHOLD:
            print("[GESTURE] Right UP + Left DOWN detected")
            gesture = True

        if gesture:
            if not show_gif:
                print("[GIF] Starting GIF playback")
            show_gif = True
            gif_start = time.time()
            gif_index = 0

    # ------------------------------
    # GIF Playback Logging
    # ------------------------------
    if show_gif:
        gif_frame = cv2.resize(gif_frames[gif_index], (cam_w, cam_h))
        output[:, cam_w:out_w] = gif_frame

        gif_index = (gif_index + 1) % len(gif_frames)

        if time.time() - gif_start > 1.5:
            print("[GIF] Finished GIF playback")
            show_gif = False
    else:
        output[:, cam_w:out_w] = (40, 40, 40)

    # ------------------------------
    # Fullscreen Scale
    # ------------------------------
    fullscreen_output = cv2.resize(output, (screen_w, screen_h))
    cv2.imshow("6-7 Detector", fullscreen_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[EXIT] User requested quit.")
        break

cap.release()
cv2.destroyAllWindows()
mp_hands.close()
print("[SHUTDOWN] Program terminated cleanly.")
