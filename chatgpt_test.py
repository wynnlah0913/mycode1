import cv2
import mediapipe as mp
import numpy as np

# Mediapipe 初始化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 簡單的手勢規則 -> 對應中文詞
gesture_map = {
    "ONE": "我",
    "TWO": "你",
    "FIVE": "好",
    "FIST": "謝謝"
}

def classify_gesture(hand_landmarks):
    """根據手部關鍵點判斷手勢 (簡單規則版)"""
    landmarks = hand_landmarks.landmark
    tips = [4, 8, 12, 16, 20]  # 拇指到小指指尖
    finger_open = []

    # 拇指：比較x座標（左/右手可能會反轉，這裡只做簡化示範）
    if landmarks[tips[0]].x < landmarks[tips[0] - 2].x:
        finger_open.append(1)
    else:
        finger_open.append(0)

    # 其他四指：比較指尖和指根的y座標
    for tip in tips[1:]:
        if landmarks[tip].y < landmarks[tip - 2].y:
            finger_open.append(1)
        else:
            finger_open.append(0)

    # 根據手指開合情況判斷手勢
    if finger_open == [0,1,0,0,0]:
        return "ONE"
    elif finger_open == [0,1,1,0,0]:
        return "TWO"
    elif finger_open == [1,1,1,1,1]:
        return "FIVE"
    elif finger_open == [0,0,0,0,0]:
        return "FIST"
    else:
        return None

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows 建議加 CAP_DSHOW 避免黑屏
    sentence = []

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 翻轉鏡像，轉換顏色
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            current_word = ""

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    gesture = classify_gesture(hand_landmarks)
                    if gesture and gesture in gesture_map:
                        current_word = gesture_map[gesture]
                        if not sentence or sentence[-1] != current_word:
                            sentence.append(current_word)

            # 顯示辨識結果
            cv2.putText(frame, f"當前詞: {current_word}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "輸出句子: " + "".join(sentence), (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Chinese Sign Language Demo", frame)

            if cv2.waitKey(5) & 0xFF == 27:  # 按 ESC 離開
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
