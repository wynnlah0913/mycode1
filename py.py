from moviepy.editor import VideoFileClip

print("✅ MoviePy 可以 import")

try:
    import tkinter
    print("✅ Tkinter 可以 import")
except Exception as e:
    print("❌ Tkinter 出錯：", e)

import subprocess
try:
    subprocess.run(["ffmpeg", "-version"], check=True)
    print("✅ ffmpeg 正常")
except Exception as e:
    print("❌ ffmpeg 出錯：", e)
