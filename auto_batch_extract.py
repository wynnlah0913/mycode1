import os
import csv
import cv2
import dlib
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# === 選擇影片資料夾 ===
tk.Tk().withdraw()
folder_path = filedialog.askdirectory(title="選擇影片資料夾")
if not folder_path:
    print("❌ 未選擇資料夾")
    exit()

# === 模型檢查 ===
if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
    print("❌ 缺少 shape_predictor_68_face_landmarks.dat")
    exit()

# === 支援的影片格式 ===
video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
video_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in video_extensions]

if not video_files:
    print("❗ 選取的資料夾中沒有影片")
    exit()

# === 模型初始化 ===
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
mp_hands = mp.solutions.hands

# === 旋轉校正 ===
def correct_rotation(frame, rotation_code):
    if rotation_code == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_code == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_code == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def get_rotation_metadata(video_path):
    try:
        import ffmpeg
        probe = ffmpeg.probe(video_path)
        rotate_tag = probe['streams'][0]['tags'].get('rotate', '0')
        return int(rotate_tag)
    except:
        return 0

# === 批次處理每一部影片 ===
for filename in video_files:
    video_path = os.path.join(folder_path, filename)
    base_name = os.path.splitext(filename)[0]

    print(f"🎬 處理中：{filename}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片：{filename}")
        continue

    rotation_code = get_rotation_metadata(video_path)
    coordinates_data = []
    frame_idx = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1440, 810))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 臉部偵測
            faces = detector(gray)
            face_coords = [(0.0, 0.0, 0.0)] * 68
            for face in faces:
                landmarks = predictor(gray, face)
                face_coords = [(landmarks.part(i).x, landmarks.part(i).y, 0.0) for i in range(68)]

            # 手部偵測
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            R_hand = [(0.0, 0.0, 0.0)] * 21
            L_hand = [(0.0, 0.0, 0.0)] * 21
            handedness = []

            if results.multi_handedness:
                for idx, hand_info in enumerate(results.multi_handedness):
                    handedness.append(hand_info.classification[0].label)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_pts = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    hand_pts = [(int(x * frame.shape[1]), int(y * frame.shape[0]), z) for x, y, z in hand_pts]
                    if handedness[idx] == 'Right':
                        R_hand = hand_pts
                    else:
                        L_hand = hand_pts

            # 儲存座標資料
            flat_data = [frame_idx]
            for pt in R_hand + L_hand + face_coords:
                flat_data.extend(pt)
            coordinates_data.append(flat_data)
            frame_idx += 1

    cap.release()

    # === 輸出 CSV 到與程式相同的資料夾 ===
    coord_header = ["frame"]
    for i in range(21):
        coord_header += [f"R{i}_x", f"R{i}_y", f"R{i}_z"]
    for i in range(21):
        coord_header += [f"L{i}_x", f"L{i}_y", f"L{i}_z"]
    for i in range(68):
        coord_header += [f"F{i+1}_x", f"F{i+1}_y", f"F{i+1}_z"]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, base_name + ".csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(coord_header)
        writer.writerows(coordinates_data)

    print(f"✅ 完成：{base_name}.csv（總幀數：{frame_idx}）")

print("\n🎉 所有影片處理完成")
