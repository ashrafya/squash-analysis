import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_utils import count_frames
from config import VIDEO_PATH

path = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
frames, fps = count_frames(path)
print(f"Frames: {frames}  |  FPS: {fps:.2f}  |  Duration: {frames/fps:.1f}s")
