import cv2
import numpy as np
import os
import glob
import queue
import threading
import emoji

q = queue.Queue()
stop_flag = threading.Event()

def Receive(rtsp_link):
    print("Starting to receive")
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec"
    cap = cv2.VideoCapture(rtsp_link)
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        q.put(frame)
    cap.release()
    q.put(None)  # Signal the end of the stream

def Display():
    print("Start displaying")
    frame_count = 0
    while not stop_flag.is_set():
        if not q.empty():
            frame = q.get()
            if frame is None:
                break
            
            # Display different messages based on frame count
            font = cv2.FONT_HERSHEY_SIMPLEX
            if frame_count % 75 >= 70:
                message = "Capturing..."
                cv2.putText(frame, message, (20, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            
            elif frame_count % 75 >= 50 and frame_count % 75 <70:
                message = "Stand Steady"
                cv2.putText(frame, message, (20, 50), font, 1, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                message = "Move"
                cv2.putText(frame, message, (20, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            
            
            # Display frame with instructions
            cv2.imshow("frame1", frame)
            
            frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_flag.set()  # Signal to stop if 'q' is pressed
            break
    
    cv2.destroyAllWindows()
    print("Stopped displaying")



def Videostream(rtsp_link):
    stop_flag.clear()
    p1 = threading.Thread(target=Receive, args=(rtsp_link,))
    p2 = threading.Thread(target=Display)
    p1.start()
    p2.start()

    return p1, p2

def stop_videostream(p1, p2):
    stop_flag.set() 
    p1.join()
    p2.join() 
    print("Video stream stopped")

def save_calibration_images(rtsp_link, output_dir, frame_limit=2250, capture_interval=75):
    p1, p2 = Videostream(rtsp_link)
    
    print('Proceeding to capture')
    
    cap = cv2.VideoCapture(rtsp_link)
    frame_count = 0
    
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % capture_interval == 0:
            frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
            cv2.imwrite(frame_path, frame)
            text = 'Saved frame:' + str(int(frame_count / capture_interval))
            frame = cv2.putText(frame, text, (100, 700), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2, cv2.LINE_4)
         
        
        frame_count += 1
        if frame_count >= frame_limit:
            print('Frame limit reached.')
            break
        
        del frame

    stop_videostream(p1, p2)
    cap.release()
    cv2.destroyAllWindows()

def calculate_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, cam_matrix, dist_coeffs, image_paths):
    reprojection_errors = []

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], cam_matrix, dist_coeffs)
        imgpoints2 = imgpoints2.reshape(-1, 1, 2)
        if imgpoints2.shape == imgpoints[i].shape:
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            reprojection_errors.append((os.path.basename(image_paths[i]), error))
        else:
            print(f"Size mismatch for image {i}: imgpoints[i].shape = {imgpoints[i].shape}, imgpoints2.shape = {imgpoints2.shape}")
            reprojection_errors.append((os.path.basename(image_paths[i]), float('inf')))  
    
    return reprojection_errors

def calibrate_camera(output_dir, checkerboard):
    subpix_criteria = (cv2.TermCriteria_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_CHECK_COND
   
    objpoints = []
    imgpoints = []
    img_shape = None
    
    images = sorted(glob.glob(os.path.join(output_dir, '*.jpg')))
    if not images:
        print("No calibration images found.")
        return None, None
    
    for img_path in images:
        img = cv2.imread(img_path)
        DIM = img.shape[::-1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            if img_shape is None:
                img_shape = gray.shape[::-1]
            objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
            objp[0, :, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)

            objpoints.append(objp)
            corner2 = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(corner2)
                        
            cv2.drawChessboardCorners(img, checkerboard, corners, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(500)  
    cv2.destroyAllWindows()
    
    imgpoints = np.array(imgpoints, dtype=np.float64)  
    objpoints = np.array(objpoints, dtype=np.float64)  

    N_OK = len(objpoints)
    cam_matrix = np.zeros((3, 3))
    dist_coeffs = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    
    rpe, cam_matrix, dist_coeffs, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, gray.shape[::-1], cam_matrix, dist_coeffs, rvecs, tvecs, calibration_flags, 
        (cv2.TermCriteria_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )
    
    print("Camera Matrix:\n", cam_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    print("Reprojection Error:\n", rpe)
    reprojection_errors = calculate_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, cam_matrix, dist_coeffs, images)

    return cam_matrix, dist_coeffs, DIM, rpe, reprojection_errors

def remove_bad_images(sorted_list, output_dir, count):
    try:
        count = int(count)  
    except ValueError:
        print("The count must be a number. Exiting...")
        return
    
    top5_error_images = sorted_list[:count]
    for image_name, error in top5_error_images:
        try:
            img_path = os.path.join(os.path.abspath(output_dir), image_name)
            img_path = os.path.abspath(img_path)
            print(img_path)
            os.remove(img_path)
        except FileNotFoundError:
            print(f"File not found: {image_name}")
        except Exception as e:
            print(f"Error deleting {image_name}: {e}")

def stream_and_capture(rtsp_link, output_dir):
    start = input('\nTo start capturing the images, please press (y/n): ')
    if start.lower() == 'y':
        save_calibration_images(rtsp_link, output_dir)
        print('Image capture completed.')
    print('Process completed.')

def run_calibration(output_dir, checkerboard, camId, rpe_old):
    cam_matrix, dist_coeffs, DIM, rpe, reprojection_errors = calibrate_camera(output_dir, checkerboard)
    if cam_matrix is not None and dist_coeffs is not None:
        np.savez(f"calibration_data_{camId}.npz", camMatrix=cam_matrix, coeffs=dist_coeffs, DIM=DIM, rpe=rpe)
        print('Individual reprojection errors:')
    error_list = [(img_name, error) for img_name, error in reprojection_errors]
    sorted_list = sorted(error_list, reverse=True, key=lambda x: x[1])
    for img_name, error in sorted_list:
        print(f"{img_name}: {error}")
    print('\nPrevious reprojection error was:', rpe_old)
    print('New calibration error is:', rpe)

def main():
    ip1 = input('Enter the third section digits of IP 192.168.__.__:')
    ip1 = int(ip1)
    ip2 = input(f'Enter last fourth section digits of IP 192.168.{ip1}.__:')
    ip2 = int(ip2)
    print(f'\nrtsp link is : rtsp://192.168.{ip1}.{ip2}:8554/h264')
    rtsp_link = f'rtsp://192.168.{ip1}.{ip2}:8554/h264'  

    dim1 = input('Enter the number of checkerboard horizontal squares:')
    dim2 = input('Enter the number of checkerboard vertical squares:')
    dim1 = int(dim1) - 1
    dim2 = int(dim2) - 1

    checkerboard = (dim1, dim2)
    print(checkerboard)
    
    camId = str(ip1) + str(ip2)
    
    output_dir = f"Images_{camId}"
    
    try:
        os.mkdir(output_dir)
        print(f"Directory '{output_dir}' created.")
    except FileExistsError:
        print(f"Directory '{output_dir}' already exists.")

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = "video_codec"
    
    stream = input('\nDo you want to stream and capture images for calibration (y/n):')
    if stream == 'y':
        stream_and_capture(rtsp_link, output_dir)

    proceed = input('\nShould we proceed for calibration (y/n):')
    if proceed == 'y':
        cam_matrix, dist_coeffs, DIM, rpe, reprojection_errors = calibrate_camera(output_dir, checkerboard)
        if cam_matrix is not None and dist_coeffs is not None:
            np.savez(f"calibration_data_{camId}.npz", camMatrix=cam_matrix, coeffs=dist_coeffs, DIM=DIM, rpe=rpe)
            rpe_old = rpe
            print('Individual reprojection errors:')
            error_list = [(img_name, error) for img_name, error in reprojection_errors]
            sorted_list = sorted(error_list, reverse=True, key=lambda x: x[1])
            for img_name, error in sorted_list:
                print(f"{img_name}: {error}")
            num_images = len(sorted_list)
    
        print(f'You have {num_images} in your directory. Keep at least 10 to 20 images for good calibration')
        remove = input(f'\nYour reprojection error for calibration was: {rpe}. Are you satisfied?\nIf not satisfied enter (y) for removing images (y): ')
        if remove.lower() == 'y':
            count = input('\nEnter number of images you want to remove (in digits only):')
            if count.isdigit():
                remove_bad_images(sorted_list, output_dir, count)
                run_calibration(output_dir, checkerboard, camId, rpe_old)
            else:
                print('Enter valid digit between 0 to 10')
    
    print('\nCalibration done', emoji.emojize(":grinning_face_with_big_eyes:"), "\U0001F389")

if __name__ == "__main__":
    main()
