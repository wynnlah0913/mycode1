import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog  # ★ 新增 filedialog
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont
from statistics import mean  # ★

GESTURES_FILE = "gestures.json"

if not os.path.exists(GESTURES_FILE):
    with open(GESTURES_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=4, ensure_ascii=False)

with open(GESTURES_FILE, "r", encoding="utf-8") as f:
    try:
        hand_gestures = json.load(f)
    except json.JSONDecodeError:
        hand_gestures = {}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
sentence = []
last_time_seen = time.time()
instruction_text = "請點選按鈕新增、辨識或刪除手勢"

# ========= 核心：把 keypoints frame 轉成你的「6位 code」 =========
def frame_keypoints_to_code(keypoints_list):
    """
    keypoints_list: 一個 frame 的扁平 keypoints 陣列，欄位順序為：
    R0_x, R0_y, R0_z, ..., R20_z, L0_x, L0_y, ..., L20_z, F1_x, ...
    我們取右手 R0~R5 的 y 值（取不到就用左手 L0~L5），
    以前 5 個點的平均 y 當門檻，y < center => '1' 否則 '0'。
    """
    def six_ys(start_offset):
        # start_offset=0 代表右手R的起點；=63 代表左手L的起點（21*3=63）
        ys = []
        for k in range(6):
            y_index = start_offset + (3 * k) + 1  # 每點是 x,y,z；+1 取 y
            if y_index < len(keypoints_list):
                ys.append(float(keypoints_list[y_index]))
            else:
                ys.append(0.0)
        return ys

    # 先試右手
    R_base = 0
    L_base = 21 * 3  # =63
    R_ys = six_ys(R_base)
    # 如果右手大多是 0，改用左手
    use_right = any(abs(v) > 1e-6 for v in R_ys)
    ys = R_ys if use_right else six_ys(L_base)

    center = mean(ys[:5])  # 與你既有程式一致：取前五個的平均做門檻
    bits = ['1' if y < center else '0' for y in ys]
    return ''.join(bits)

# ========= 匯入資料夾中的 JSON 樣本，批量更新 gestures.json =========
def import_dataset_json_folder():
    global instruction_text, hand_gestures
    folder = filedialog.askdirectory(title="選擇含多個段落 JSON 的資料夾")
    if not folder:
        return

    added, conflicted = 0, []
    # 將同一 label 的多個樣本合併為「投票/平均」：我們這裡採用「多數出現的 code」
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

            # 取前 N 幀（如 5 幀）做穩定化
            N = min(5, len(frames))
            codes = []
            for i in range(N):
                keypoints = frames[i].get("keypoints", [])
                if not keypoints:
                    continue
                codes.append(frame_keypoints_to_code(keypoints))

            if not codes:
                continue

            # 取多數票的 code
            from collections import Counter
            code = Counter(codes).most_common(1)[0][0]

            label_to_codes.setdefault(label, []).append(code)

        except Exception as e:
            print(f"⚠️ 無法讀取/解析：{fname}，原因：{e}")

    # 將每個 label 的多個 code 再做一次多數票
    for label, codes in label_to_codes.items():
        from collections import Counter
        vote = Counter(codes).most_common(1)[0][0]
        # 如果這個 code 已經對應到其他 label，就標記衝突（同一 code 兩個詞）
        if vote in hand_gestures and hand_gestures[vote] != label:
            conflicted.append((vote, hand_gestures[vote], label))
        else:
            hand_gestures[vote] = label
            added += 1

    with open(GESTURES_FILE, "w", encoding="utf-8") as f:
        json.dump(hand_gestures, f, indent=4, ensure_ascii=False)

    if conflicted:
        msg = "部分樣本的 code 與既有詞衝突（已保留原有對應）：\n"
        msg += "\n".join([f"code {c}：原='{old}' 新='{new}'" for c, old, new in conflicted[:10]])
        messagebox.showwarning("有衝突", msg)

    instruction_text = f"📥 已匯入 {added} 筆手勢樣本（依多數票建立 code→label）"
    messagebox.showinfo("完成", instruction_text)

# =========（可選）將單一 JSON 檔直接加入 gestures.json =========
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

        N = min(5, len(frames))
        codes = []
        for i in range(N):
            keypoints = frames[i].get("keypoints", [])
            if not keypoints:
                continue
            codes.append(frame_keypoints_to_code(keypoints))

        if not codes:
            messagebox.showerror("錯誤", "無法從 frames 取得 keypoints")
            return

        from collections import Counter
        code = Counter(codes).most_common(1)[0][0]

        # 衝突檢查
        if code in hand_gestures and hand_gestures[code] != label:
            messagebox.showwarning("衝突",
                                   f"此 code 已對應 '{hand_gestures[code]}'，本檔案標籤為 '{label}'。保留原有不覆蓋。")
            return

        hand_gestures[code] = label
        with open(GESTURES_FILE, "w", encoding="utf-8") as f:
            json.dump(hand_gestures, f, indent=4, ensure_ascii=False)

        messagebox.showinfo("成功", f"✅ 新增：{label} （code={code}）")

    except Exception as e:
        messagebox.showerror("錯誤", f"讀取失敗：{e}")

# ======== 下方為你原本的程式（略），只在 UI 位置多加兩顆按鈕 ========

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

def update_frame():
    global instruction_text, sentence, last_time_seen
    ret, frame = cap.read()
    if not ret:
        return
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    result_text = "未偵測"
    current_time = time.time()

    if recognizing and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [lm.y for lm in hand_landmarks.landmark[:6]]
            center = np.mean(landmarks[:5])
            code = "".join(['1' if y < center else '0' for y in landmarks])
            result_text = hand_gestures.get(code, "未知手勢")
            if result_text != "未知手勢" and (len(sentence) == 0 or sentence[-1] != result_text):
                sentence.append(result_text)
                last_time_seen = current_time
    elif recognizing:
        if current_time - last_time_seen > 1.5 and sentence:
            print("📝 句子辨識結果：", " ".join(sentence))
            sentence = []

    frame = draw_text(frame, f"{instruction_text}", (30, 20))
    frame = draw_text(frame, f"當前辨識：{result_text}", (30, 60))
    frame = draw_text(frame, f"組句中：{' '.join(sentence)}", (30, 100))

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

def capture_hand_gesture():
    captured = []
    started = False
    frame_count = 0
    global instruction_text
    instruction_text = "📸 偵測中，擺好姿勢..."
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            if not started:
                started = True
                frame_count = 0
                instruction_text = "✅ 偵測到手勢，開始拍攝..."
            if started and frame_count < 5:
                landmarks = [lm.y for lm in hand.landmark[:6]]
                captured.append(landmarks)
                frame_count += 1
                instruction_text = f"📷 拍攝中：{frame_count}/5"
                cv2.waitKey(1000)
            elif frame_count >= 5:
                break
        else:
            instruction_text = "請擺出你的手勢..."
        cv2.waitKey(100)
    instruction_text = "完成！可繼續新增、辨識或刪除"
    if len(captured) == 5:
        avg = np.mean(captured, axis=0)
        center = np.mean(avg[:5])
        code = "".join(['1' if y < center else '0' for y in avg])
        return code
    return None

def add_gesture():
    label = simpledialog.askstring("輸入手勢名稱", "這個手勢代表的詞語（例如：你好）：")
    if not label:
        return
    code = capture_hand_gesture()
    if code:
        hand_gestures[code] = label
        with open(GESTURES_FILE, "w", encoding="utf-8") as f:
            json.dump(hand_gestures, f, indent=4, ensure_ascii=False)
        messagebox.showinfo("成功", f"✅ 已新增手勢：{label}")
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

def toggle_recognition():
    global recognizing, instruction_text
    recognizing = not recognizing
    if recognizing:
        instruction_text = "🧠 手語辨識已啟動..."
        recog_button.config(text="🛑 停止辨識")
    else:
        instruction_text = "🔴 已停止辨識"
        recog_button.config(text="▶️ 開始辨識")

# ===================== UI =====================
root = tk.Tk()
root.title("一體化手語系統 GUI（含刪除功能）")
root.geometry("760x620")

video_label = tk.Label(root)
video_label.pack()

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

add_button = tk.Button(button_frame, text="➕ 新增手勢", command=add_gesture, bg="lightblue", font=("Arial", 12))
add_button.grid(row=0, column=0, padx=10)

recog_button = tk.Button(button_frame, text="▶️ 開始辨識", command=toggle_recognition, bg="lightgreen", font=("Arial", 12))
recog_button.grid(row=0, column=1, padx=10)

del_button = tk.Button(button_frame, text="🗑️ 刪除手勢", command=delete_gesture, bg="tomato", font=("Arial", 12))
del_button.grid(row=0, column=2, padx=10)

# ★ 新增兩顆按鈕：資料集匯入
imp_one_btn = tk.Button(button_frame, text="📄 匯入單一 JSON", command=import_single_json, bg="#f0e68c", font=("Arial", 12))
imp_one_btn.grid(row=1, column=0, padx=10, pady=6)

imp_folder_btn = tk.Button(button_frame, text="📂 匯入資料夾 JSON", command=import_dataset_json_folder, bg="#ffd700", font=("Arial", 12))
imp_folder_btn.grid(row=1, column=1, padx=10, pady=6)

recognizing = False
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
