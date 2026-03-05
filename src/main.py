import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",     action="store_true", help="Show live tracking animation while processing.")
    parser.add_argument("--calibrate", action="store_true", help="Force recalibration of the court homography.")
    parser.add_argument("--reuse",     action="store_true", help="Skip tracking and reuse positions from the last run.")
    parser.add_argument(
        "--tracker",
        choices=["mediapipe", "yolo"],
        default="yolo",
        help="Tracking backend: 'mediapipe' (default, Option A) or 'yolo' (Option B — YOLOv8-pose, requires: pip install ultralytics).",
    )
    args = parser.parse_args()

    if args.tracker == "yolo":
        from extract_pose_yolo import main
    else:
        from extract_pose import main

    main(debug=args.debug, calibrate=args.calibrate, reuse=args.reuse)
