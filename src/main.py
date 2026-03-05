import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

import argparse
from extract_pose import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",     action="store_true", help="Show live tracking animation while processing.")
    parser.add_argument("--calibrate", action="store_true", help="Force recalibration of the court homography.")
    parser.add_argument("--reuse",     action="store_true", help="Skip tracking and reuse positions from the last run.")
    args = parser.parse_args()
    main(debug=args.debug, calibrate=args.calibrate, reuse=args.reuse)
