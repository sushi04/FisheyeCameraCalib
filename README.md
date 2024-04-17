# FisheyeCameraCalib

# Video_calibration.py
This script captures frames from a video stream obtained from an RTSP link, detects a checkerboard pattern in the frames, and performs camera calibration using the detected corners.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## Setup

1. Install Python 3.x if not already installed.
2. Install required packages using pip:

## Usage

1. Run the script `calibrate_camera.py`.
2. Enter the camera ID when prompted.
3. Ensure that the RTSP link is accessible and providing a video stream.
4. The script will capture frames from the video stream, detect checkerboard corners, and perform camera calibration.
5. Once calibration is completed, the calibration data will be saved in a `.npz` file (`calibration_data_{camId}.npz`).

## Additional Notes

- Ensure that the camera is properly calibrated and the checkerboard pattern is clearly visible in the captured frames for accurate calibration results.
- Adjust the parameters such as the RTSP link, output directory, checkerboard dimensions, and calibration flags as needed for your specific setup.

# Videostream.py


