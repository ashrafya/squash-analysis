import argparse
from extract_pose import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug",     action="store_true", help="Show live tracking animation while processing.")
    parser.add_argument("--calibrate", action="store_true", help="Force recalibration of the court homography.")
    args = parser.parse_args()
    main(debug=args.debug, calibrate=args.calibrate)
