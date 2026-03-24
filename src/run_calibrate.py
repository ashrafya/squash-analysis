"""
run_calibrate.py — standalone 3-D court calibration tool

Usage:
    python run_calibrate.py               # calibrate from first video frame
    python run_calibrate.py --frame 500   # use frame 500 (avoids dark intros)
    python run_calibrate.py --force       # force redo even if calibration exists

After running, saves:
    assets/calibration_3d.npz  (used automatically by the pipeline)
    assets/homography.npy      (legacy 2-D fallback, kept for backward compat)
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from calibrate import get_calibration
from video_utils import load_video
from config import VIDEO_PATH


def main():
    parser = argparse.ArgumentParser(description="3-D court calibration for squash analysis")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to use for calibration (default: 0 = first frame)")
    parser.add_argument("--force", action="store_true",
                        help="Force recalibration even if a saved calibration exists")
    parser.add_argument("--video", default=VIDEO_PATH,
                        help=f"Path to video file (default: {VIDEO_PATH})")
    args = parser.parse_args()

    print(f"Loading video: {args.video}")
    cap, fps, n_frames = load_video(args.video)

    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"ERROR: Could not read frame {args.frame} from video.")
        sys.exit(1)

    h, w = frame.shape[:2]
    print(f"Using frame {args.frame}  ({w}×{h} px)")

    calib = get_calibration(frame, force_recalibrate=args.force)

    print("\nCalibration complete.")
    print(f"  Camera centre (world): x={calib.C[0]:.3f} m, y={calib.C[1]:.3f} m, z={calib.C[2]:.3f} m")
    print("  assets/calibration_3d.npz updated.")
    print("  Run the pipeline normally (python main.py) to use it.")


if __name__ == "__main__":
    main()
