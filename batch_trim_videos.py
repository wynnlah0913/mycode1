import os
from moviepy.editor import VideoFileClip
from tkinter import filedialog, Tk

def trim_video(filepath, start_time, end_time):
    try:
        clip = VideoFileClip(filepath)

        # 補充：強制設定 fps 以避免寫出時錯誤（尤其是 VFR 影片）
        fps = clip.fps if clip.fps else 30

        # 裁切片段
        trimmed_clip = clip.subclip(start_time, end_time)

        # 暫存檔案名（避免立即覆蓋）
        temp_path = filepath + "_temp.mp4"

        # 寫入剪輯後影片（加入 faststart，修復播放卡住）
        trimmed_clip.write_videofile(
            temp_path,
            codec='libx264',
            audio_codec='aac',
            fps=fps,
            ffmpeg_params=["-movflags", "faststart"]
        )

        # 關閉資源
        clip.close()
        trimmed_clip.close()

        # 用剪輯後檔案覆蓋原始影片
        os.remove(filepath)
        os.rename(temp_path, filepath)

        print(f"✔ 處理完成：{os.path.basename(filepath)}")
    except Exception as e:
        print(f"✘ 發生錯誤處理 {filepath}：{e}")

def process_folder(folder_path):
    supported_exts = ('.mp4', '.mov', '.avi', '.mkv')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_exts):
            full_path = os.path.join(folder_path, filename)
            trim_video(full_path, start_time=20, end_time=30)

if __name__ == "__main__":
    print("請選擇影片資料夾...")
    root = Tk()
    root.withdraw()
    selected_folder = filedialog.askdirectory(title="選擇影片資料夾")

    if selected_folder:
        print(f"📂 開始處理資料夾：{selected_folder}")
        process_folder(selected_folder)
        print("✅ 所有影片處理完成！")
    else:
        print("⚠ 未選擇資料夾，程式結束。")

