import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont
from collections import Counter

# ===================== 基本設定 =====================
GESTURES_FILE = "gestures.json"
CODE_HAND = 21   # 每手 21 點
CODE_FACE = 68   # 臉 68 點（取 FaceMesh 前 68 點）
TOTAL_CODE_LEN = CODE_HAND * 2 + CODE_FACE  # 110

# 若無 gestures.json，建立空白
if not os.path.exists(GESTURES_FILE):
    with open(GESTURES_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=4, ensure_ascii=False)

# 讀取 gestures.json
try:
    with open(GESTURES_FILE, "r", encoding="utf-8") as f:
        hand_gestures = json.load(f)
except Exception:
    hand_gestures = {}

# MediaPipe 初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 攝影機
cap = cv2.VideoCapture(0)

# 狀態
sentence = []
last_time_seen = time.time()
instruction_text = "請點選按鈕：新增／辨識／刪除／匯入 JSON"
recognizing = False

# ===================== 工具函式 =====================
def draw_text(frame, text, pos=(30, 30), font_size=28, color=(0, 255, 0)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("msjh.ttc", font_size)
    except:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def extract_hands_landmarks(results):
    """
    將 MediaPipe Hands 的結果依據 handedness 對應為 {'Right': landmarks, 'Left': landmarks}
    """
    hands_map = {}
    if results.multi_hand_landmarks and results.multi_handedness:
        for handed, lm in zip(results.multi_handedness, results.multi_hand_landmarks):
            label = handed.classification[0].label  # 'Right' / 'Left'
            hands_map[label] = lm.landmark
    return hands_map

def bits_from_ys(ys, n):
    """
    由 n 個 y 值產生 n 位二值字串；若幾乎全 0 視為缺席，回傳 '0'*n
    規則：以自身平均為門檻，y < center → '1'，否則 '0'
    """
    if not ys or not any(abs(v) > 1e-6 for v in ys):
        return '0' * n
    c = float(np.mean(ys))
    return ''.join('1' if y < c else '0' for y in ys)

def make_110_code_from_live(frame_bgr, hands_results):
    """
    從即時畫面產生 110 位 code（Right21 + Left21 + Face68）
    缺席部位以 0 佔位
    """
    # 手
    code_R = '0' * CODE_HAND
    code_L = '0' * CODE_HAND
    hands_map = extract_hands_landmarks(hands_results)
    if 'Right' in hands_map:
        ys_R = [lm.y for lm in hands_map['Right'][:CODE_HAND]]
        code_R = bits_from_ys(ys_R, CODE_HAND)
    if 'Left' in hands_map:
        ys_L = [lm.y for lm in hands_map['Left'][:CODE_HAND]]
        code_L = bits_from_ys(ys_L, CODE_HAND)

    # 臉
    code_F = '0' * CODE_FACE
    face_results = face_mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    if face_results.multi_face_landmarks:
        f = face_results.multi_face_landmarks[0].landmark
        ys_F = [f[i].y for i in range(min(CODE_FACE, len(f)))]
        if len(ys_F) < CODE_FACE:
            ys_F += [0.0] * (CODE_FACE - len(ys_F))
        code_F = bits_from_ys(ys_F, CODE_FACE)

    return code_R + code_L + code_F  # 110 位

def frame_keypoints_to_code_from_json(keypoints_list):
    """
    從 JSON 的扁平 keypoints（R 21*3 → L 21*3 → F 68*3）產生 110 位 code
    """
    def ys_n(start_offset, n):
        ys = []
        for i in range(n):
            y_idx = start_offset + 3 * i + 1
            ys.append(float(keypoints_list[y_idx]) if y_idx < len(keypoints_list) else 0.0)
        return ys

    R_base = 0
    L_base = CODE_HAND * 3            # 63
    F_base = (CODE_HAND * 2) * 3      # 126
    ys_R = ys_n(R_base, CODE_HAND)
    ys_L = ys_n(L_base, CODE_HAND)
    ys_F = ys_n(F_base, CODE_FACE)

    code_R = bits_from_ys(ys_R, CODE_HAND)
    code_L = bits_from_ys(ys_L, CODE_HAND)
    code_F = bits_from_ys(ys_F, CODE_FACE)
    return code_R + code_L + code_F

# ===================== JSON 匯入（單檔 / 資料夾） =====================
def import_single_json():
    path = filedialog.askopenfilename(
        title="選擇單一段落 JSON",
        filetypes=[("JSON Files", "*.json")]
    )
    if not path:
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        label = str(data.get("label", "")).strip()
        frames = data.get("frames", [])
        if not label or not frames:
            messagebox.showerror("錯誤", "JSON 內容缺少 label 或 frames")
            return

        # 取前 5 幀做穩定化 → 多數票 code
        codes = []
        for i in range(min(5, len(frames))):
            keypoints = frames[i].get("keypoints", [])
            if keypoints:
                codes.append(frame_keypoints_to_code_from_json(keypoints))
        if not codes:
            messagebox.showerror("錯誤", "無法從 frames 取得 keypoints")
            return
        vote_code = Counter(codes).most_common(1)[0][0]

        # 衝突檢查
        if vote_code in hand_gestures and hand_gestures[vote_code] != label:
            messagebox.showwarning(
                "衝突",
                f"此 code 已對應「{hand_gestures[vote_code]}」，本檔案標籤為「{label}」。保留原有不覆蓋。"
            )
            return

        hand_gestures[vote_code] = label
        with open(GESTURES_FILE, "w", encoding="utf-8") as f:
            json.dump(hand_gestures, f, indent=4, ensure_ascii=False)

        messagebox.showinfo("成功", f"✅ 新增：{label}（110位 code）")
    except Exception as e:
        messagebox.showerror("錯誤", f"讀取失敗：{e}")

def import_dataset_json_folder():
    folder = filedialog.askdirectory(title="選擇含多個段落 JSON 的資料夾")
    if not folder:
        return

    added = 0
    conflicted = []

    # label → 多個 code，最後再做一次多數票
    label_to_codes = {}

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            label = str(data.get("label", "")).strip()
            frames = data.get("frames", [])
            if not label or not frames:
                continue

            codes = []
            for i in range(min(5, len(frames))):
                keypoints = frames[i].get("keypoints", [])
                if keypoints:
                    codes.append(frame_keypoints_to_code_from_json(keypoints))
            if codes:
                label_to_codes.setdefault(label, []).append(Counter(codes).most_common(1)[0][0])

        except Exception as e:
            print(f"⚠️ 無法解析 {fname}：{e}")

    for label, codes in label_to_codes.items():
        vote_code = Counter(codes).most_common(1)[0][0]
        if vote_code in hand_gestures and hand_gestures[vote_code] != label:
            conflicted.append((vote_code, hand_gestures[vote_code], label))
        else:
            hand_gestures[vote_code] = label
            added += 1

    with open(GESTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(hand_gestures, f, indent=4, ensure_ascii=False)

    if conflicted:
        msg = "部分樣本的 110位 code 與既有詞衝突（已保留原對應）：\n"
        msg += "\n".join([f"code={c[:16]}... 原='{old}' 新='{new}'" for c, old, new in conflicted[:12]])
        messagebox.showwarning("有衝突", msg)

    messagebox.showinfo("完成", f"📥 已匯入 {added} 筆手勢樣本（110位 code）")

# ===================== 新增 / 刪除手勢 =====================
def capture_hand_gesture():
    """
    連拍 5 張：右手 21、左手 21、臉 68 的 y 值
    平均後各自二值化，串成 110 位 code
    """
    cap_R, cap_L, cap_F = [], [], []
    started = False
    frame_count = 0
    global instruction_text
    instruction_text = "📸 偵測中，請讓『雙手＋臉』同時入鏡..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = hands.process(img_rgb)
        face_results = face_mesh.process(img_rgb)

        if hand_results.multi_hand_landmarks or face_results.multi_face_landmarks:
            if not started:
                started = True
                frame_count = 0
                instruction_text = "✅ 偵測到，開始拍攝..."

            if frame_count < 5:
                # 手
                hands_map = extract_hands_landmarks(hand_results)
                if 'Right' in hands_map:
                    cap_R.append([lm.y for lm in hands_map['Right'][:CODE_HAND]])
                else:
                    cap_R.append([0.0] * CODE_HAND)

                if 'Left' in hands_map:
                    cap_L.append([lm.y for lm in hands_map['Left'][:CODE_HAND]])
                else:
                    cap_L.append([0.0] * CODE_HAND)

                # 臉
                if face_results.multi_face_landmarks:
                    f = face_results.multi_face_landmarks[0].landmark
                    ys = [f[i].y for i in range(min(CODE_FACE, len(f)))]
                    if len(ys) < CODE_FACE:
                        ys += [0.0] * (CODE_FACE - len(ys))
                    cap_F.append(ys)
                else:
                    cap_F.append([0.0] * CODE_FACE)

                frame_count += 1
                instruction_text = f"📷 拍攝中：{frame_count}/5"
                cv2.waitKey(1000)
            else:
                break
        else:
            instruction_text = "請讓『雙手＋臉』清楚入鏡…"
        cv2.waitKey(100)

    instruction_text = "完成！可繼續新增、辨識或刪除"
    if frame_count < 5:
        return None

    def avg_bits(mat, n):
        avg = np.mean(mat, axis=0) if mat else [0.0] * n
        return bits_from_ys(avg, n)

    code_R = avg_bits(cap_R, CODE_HAND)
    code_L = avg_bits(cap_L, CODE_HAND)
    code_F = avg_bits(cap_F, CODE_FACE)
    return code_R + code_L + code_F  # 110 位

def add_gesture():
    label = simpledialog.askstring("輸入手勢名稱", "這個手勢代表的詞語（例如：你好）：")
    if not label:
        return
    code = capture_hand_gesture()
    if code:
        # 衝突檢查
        if code in hand_gestures and hand_gestures[code] != label:
            messagebox.showwarning("衝突", f"此 code 已對應「{hand_gestures[code]}」。保留原有不覆蓋。")
            return
        hand_gestures[code] = label
        with open(GESTURES_FILE, "w", encoding="utf-8") as f:
            json.dump(hand_gestures, f, indent=4, ensure_ascii=False)
        messagebox.showinfo("成功", f"✅ 已新增：{label}（110位 code）")
    else:
        messagebox.showerror("錯誤", "❌ 擷取失敗，請再試一次。")

def delete_gesture():
    label = simpledialog.askstring("刪除手勢", "請輸入要刪除的詞語（中文）：")
    if not label:
        return
    found = False
    for code, word in list(hand_gestures.items()):
        if word == label:
            del hand_gestures[code]
            found = True
            break
    if found:
        with open(GESTURES_FILE, "w", encoding="utf-8") as f:
            json.dump(hand_gestures, f, indent=4, ensure_ascii=False)
        messagebox.showinfo("刪除成功", f"✅ 已刪除手勢：{label}")
    else:
        messagebox.showerror("未找到", f"❌ 找不到手勢：{label}")

# ===================== 即時影像迴圈 =====================
def toggle_recognition():
    global recognizing, instruction_text
    recognizing = not recognizing
    if recognizing:
        instruction_text = "🧠 手語辨識已啟動..."
        recog_button.config(text="🛑 停止辨識")
    else:
        instruction_text = "🔴 已停止辨識"
        recog_button.config(text="▶️ 開始辨識")

def update_frame():
    global instruction_text, sentence, last_time_seen
    ret, frame = cap.read()
    if not ret:
        video_label.after(10, update_frame)
        return

    # 手部
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    result_text = "未偵測"
    current_time = time.time()

    if recognizing:
        # 產生 110 位 code
        code = make_110_code_from_live(frame, results)
        # 繪製手部骨架（可選）
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        result_text = hand_gestures.get(code, "未知手勢")
        if result_text != "未知手勢" and (len(sentence) == 0 or sentence[-1] != result_text):
            sentence.append(result_text)
            last_time_seen = current_time
        elif current_time - last_time_seen > 1.5 and sentence:
            print("📝 句子辨識結果：", " ".join(sentence))
            sentence = []

    # 疊字
    frame = draw_text(frame, f"{instruction_text}", (30, 20))
    frame = draw_text(frame, f"當前辨識：{result_text}", (30, 60))
    frame = draw_text(frame, f"組句中：{' '.join(sentence)}", (30, 100))

    # 顯示到 Tkinter
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

# ===================== UI =====================
root = tk.Tk()
root.title("一體化手語系統 GUI（雙手+臉 110位 編碼）")
root.geometry("800x660")

video_label = tk.Label(root)
video_label.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

add_button = tk.Button(button_frame, text="➕ 新增手勢", command=add_gesture, bg="lightblue", font=("Arial", 12))
add_button.grid(row=0, column=0, padx=8)

recog_button = tk.Button(button_frame, text="▶️ 開始辨識", command=toggle_recognition, bg="lightgreen", font=("Arial", 12))
recog_button.grid(row=0, column=1, padx=8)

del_button = tk.Button(button_frame, text="🗑️ 刪除手勢", command=delete_gesture, bg="tomato", font=("Arial", 12))
del_button.grid(row=0, column=2, padx=8)

imp_one_btn = tk.Button(button_frame, text="📄 匯入單一 JSON", command=import_single_json, bg="#f0e68c", font=("Arial", 12))
imp_one_btn.grid(row=1, column=0, padx=8, pady=6)

imp_folder_btn = tk.Button(button_frame, text="📂 匯入資料夾 JSON", command=import_dataset_json_folder, bg="#ffd700", font=("Arial", 12))
imp_folder_btn.grid(row=1, column=1, padx=8, pady=6)

update_frame()
root.mainloop()

# 收尾
cap.release()
cv2.destroyAllWindows()
