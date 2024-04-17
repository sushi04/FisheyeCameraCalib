# FisheyeCameraCalib

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## Setup

1. Install Python 3.x if not already installed.
2. Install required packages using pip:

# Video_calibration.py
This script captures frames from a video stream obtained from an RTSP link, detects a checkerboard pattern in the frames, and performs camera calibration using the detected corners.


## Usage

1. Run the script `video_calibration.py`.
2. Enter the camera ID when prompted.
3. Ensure that the RTSP link is accessible and providing a video stream.
4. The script will capture frames from the video stream, detect checkerboard corners, and perform camera calibration.
5. Once calibration is completed, the calibration data will be saved in a `.npz` file (`calibration_data_{camId}.npz`).

## Additional Notes

- Ensure that the camera is properly calibrated and the checkerboard pattern is clearly visible in the captured frames for accurate calibration results.
- Adjust the parameters such as the RTSP link, output directory, checkerboard dimensions, and calibration flags as needed for your specific setup.

# Videostream.py
This Script undistorts the video stream using calibration data stored by the video_calibration.py.

## Usage

1. Ensure that you have a camera calibrated and have saved the calibration data (`calibration_data_{camId}.npz`).
2. Update the script with the correct file path to the calibration data.
3. Set the correct RTSP link for the video stream.
4. Run the script `videostreaming.py`.
5. The script will capture the video stream, undistort each frame in real-time using the provided calibration data, and display the undistorted video.
6. Press the 'q' key to exit the program.

## Additional Notes

- Ensure that the camera is properly calibrated and the calibration data is accurate for effective undistortion.
- Adjust the RTSP link and other parameters as needed for your specific camera setup.
- The script may need adjustments depending on the camera model and calibration process used.
- Make sure your system supports RTSP video streaming and has access to the provided RTSP link.


