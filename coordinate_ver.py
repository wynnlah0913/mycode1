import os
import csv
import cv2
import dlib
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# === File dialog ===
tk.Tk().withdraw()
video_path = filedialog.askopenfilename(
    title="Select Video File",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)
if not video_path:
    print("No video selected.")
    exit()

if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    print("❌ Missing shape_predictor_68_face_landmarks.dat")
    exit()

# === Init models ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# === Init video ===
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Cannot open video.")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0
coordinates_data = []
paused = False
show_relative = True  # ← 預設為相對座標模式

scroll_offset = 0
line_height = 14

def mouse_scroll(event, x, y, flags, param):
    global scroll_offset
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scroll_offset = max(scroll_offset - line_height, 0)
        else:
            scroll_offset += line_height

cv2.namedWindow("Landmark Annotator")
cv2.setMouseCallback("Landmark Annotator", mouse_scroll)

# === 偵測函式 ===
def detect_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_coords = [(0.0, 0.0, 0.0)] * 68
    for face in faces:
        landmarks = predictor(gray, face)
        face_coords = [(landmarks.part(i).x, landmarks.part(i).y, 0.0) for i in range(68)]

    R_hand = [(0.0, 0.0, 0.0)] * 21
    L_hand = [(0.0, 0.0, 0.0)] * 21
    handedness = []
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_handedness:
        for idx, hand_info in enumerate(results.multi_handedness):
            handedness.append(hand_info.classification[0].label)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_pts = [(lm.x * frame.shape[1], lm.y * frame.shape[0], lm.z) for lm in hand_landmarks.landmark]
            if handedness[idx] == 'Right':
                R_hand = hand_pts
            else:
                L_hand = hand_pts
    return face_coords, L_hand, R_hand, results

# === 相對座標轉換 ===
def compute_relative(face_coords, hand_pts):
    if not face_coords or all(x == 0 and y == 0 for x, y, _ in face_coords):
        return hand_pts
    x_min = min(x for (x,y,_) in face_coords)
    x_max = max(x for (x,y,_) in face_coords)
    y_min = min(y for (x,y,_) in face_coords)
    y_max = max(y for (x,y,_) in face_coords)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w = max(1, (x_max - x_min))
    h = max(1, (y_max - y_min))
    def normalize(points):
        return [((x - cx) / w, (y - cy) / h, z) for (x,y,z) in points]
    return normalize(hand_pts)

# === Main loop ===
last_frame = None
face_coords, L_hand, R_hand = [(0,0,0)]*68, [(0,0,0)]*21, [(0,0,0)]*21

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        if not paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (1000, 750))
            last_frame = frame.copy()
            face_coords, L_hand, R_hand, results = detect_landmarks(frame)
            coordinates_data.append([frame_idx, face_coords, L_hand, R_hand])
        else:
            frame = last_frame.copy() if last_frame is not None else np.zeros((750,1000,3),dtype=np.uint8)

        # === Coordinate mode ===
        if show_relative:
            face_display = compute_relative(face_coords, face_coords)
            L_display = compute_relative(face_coords, L_hand)
            R_display = compute_relative(face_coords, R_hand)
            mode_text = "RELATIVE (to face center)"
        else:
            face_display, L_display, R_display = face_coords, L_hand, R_hand
            mode_text = "ABSOLUTE (screen)"

        # === Draw landmarks (with labels) ===
        for i, (x, y, _) in enumerate(face_coords):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv2.putText(frame, f"F{i+1}", (int(x)+4, int(y)-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        for i, (x, y, _) in enumerate(L_hand):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
                cv2.putText(frame, f"L{i}", (int(x)+4, int(y)-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        for i, (x, y, _) in enumerate(R_hand):
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                cv2.putText(frame, f"R{i}", (int(x)+4, int(y)-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)


        # === Mode label on top-left ===
        cv2.putText(frame, f"Mode: {mode_text}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # === Right panel ===
        info_width = 350
        frame = cv2.copyMakeBorder(frame, 0, 0, 0, info_width, cv2.BORDER_CONSTANT, value=(50, 50, 50))
        col1_x, col2_x = 1020, 1180
        y_offset = 30
        cv2.putText(frame, f"=== Landmark Info ({mode_text}) ===", (col1_x-10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        lines_col1 = [f"F{i+1}: ({x:.2f}, {y:.2f}, {z:.2f})" for i, (x, y, z) in enumerate(face_display)]
        lines_col2 = [f"L{i}: ({x:.2f}, {y:.2f}, {z:.2f})" for i, (x, y, z) in enumerate(L_display)]
        lines_col2 += [f"R{i}: ({x:.2f}, {y:.2f}, {z:.2f})" for i, (x, y, z) in enumerate(R_display)]
        start_idx = scroll_offset // line_height
        visible_lines = (frame.shape[0] - 180) // line_height
        y = y_offset + 20
        for line in lines_col1[start_idx:start_idx+visible_lines]:
            cv2.putText(frame, line, (col1_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1)
            y += line_height
        y = y_offset + 20
        for line in lines_col2[start_idx:start_idx+visible_lines]:
            cv2.putText(frame, line, (col2_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 255), 1)
            y += line_height

        # === Progress bar ===
        bar_x1, bar_x2 = 10, 900
        bar_y1, bar_y2 = frame.shape[0] - 20, frame.shape[0] - 10
        cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (100, 100, 100), -1)
        progress = frame_idx / max(1, total_frames)
        progress_x = int(bar_x1 + progress * (bar_x2 - bar_x1))
        cv2.rectangle(frame, (bar_x1, bar_y1), (progress_x, bar_y2), (255, 200, 0), -1)
        cv2.putText(frame, f"{progress*100:.1f}%", (bar_x2+10, bar_y2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # === Control hints ===
        help_y = frame.shape[0] - 80
        cv2.rectangle(frame, (5, help_y-20), (480, help_y+60), (255, 255, 255), -1)
        cv2.putText(frame, "[SPACE] Play/Pause   [A] Back30   [D] Fwd30", (10, help_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "[Q] Back1   [E] Fwd1   [ESC] Exit", (10, help_y+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, "[R] Relative   [T] Absolute", (10, help_y+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("Landmark Annotator", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('a'):
            frame_idx = max(frame_idx - 30, 0); paused = False
        elif key == ord('d'):
            frame_idx = min(frame_idx + 30, total_frames-1); paused = False
        elif key == ord('q'):
            frame_idx = max(frame_idx - 1, 0); paused = False
        elif key == ord('e'):
            frame_idx = min(frame_idx + 1, total_frames-1); paused = False
        elif key == ord('r'):
            show_relative = True
        elif key == ord('t'):
            show_relative = False

        if not paused:
            frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# === Export CSV ===
print("💾 Saving coordinates.csv and relative_coordinates.csv ...")

coord_header = ["frame"]
for i in range(21):
    coord_header += [f"R{i}_x", f"R{i}_y", f"R{i}_z"]
for i in range(21):
    coord_header += [f"L{i}_x", f"L{i}_y", f"L{i}_z"]
for i in range(68):
    coord_header += [f"F{i+1}_x", f"F{i+1}_y", f"F{i+1}_z"]

absolute_rows = []
relative_rows = []

for frame_id, face_coords, L_hand, R_hand in coordinates_data:
    abs_row = [frame_id]
    rel_row = [frame_id]

    # absolute
    for pt in R_hand + L_hand + face_coords:
        abs_row.extend(pt)
    absolute_rows.append(abs_row)

    # relative
    def normalize(points):
        x_min = min(x for (x,y,_) in face_coords)
        x_max = max(x for (x,y,_) in face_coords)
        y_min = min(y for (x,y,_) in face_coords)
        y_max = max(y for (x,y,_) in face_coords)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = max(1, x_max - x_min)
        h = max(1, y_max - y_min)
        return [((x - cx) / w, (y - cy) / h, z) for (x,y,z) in points]

    R_rel = normalize(R_hand)
    L_rel = normalize(L_hand)
    F_rel = normalize(face_coords)
    for pt in R_rel + L_rel + F_rel:
        rel_row.extend(pt)
    relative_rows.append(rel_row)

with open("coordinates.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f); writer.writerow(coord_header); writer.writerows(absolute_rows)
with open("relative_coordinates.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f); writer.writerow(coord_header); writer.writerows(relative_rows)

print("✅ Done! Files saved: coordinates.csv & relative_coordinates.csv")
