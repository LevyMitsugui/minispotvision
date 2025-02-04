import cv2
import numpy as np
import os

cam_calib_path = "./camera_calibration.txt"

def undistort_frame(frame, mtx, dist):
    
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    h,  w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst

def get_cam_calib(calib_path):
    try:
        calib_file = open(cam_calib_path, "r")
    except:
        print("\n\nError: Could not open camera calibration file.")
        print("       Wrong or unavailableCalibration file name or path.\n\n")
        exit()
    else:
        lines = calib_file.readlines()
        calib_file.close()


    for idx, line in enumerate(lines):
        if line.startswith("Camera Calibration Parameters"):
            print(lines[idx+1])
        elif line.startswith("Camera Matrix (Intrinsic Parameters):"):
            result_list = []
            for i in range(3):
                result_list.append(list(map(float, lines[idx+1+i].split())))
            mtx = np.array(result_list)
            print("\nCamera Matrix (Intrinsic Parameters):\n\n",mtx)
        elif line.startswith("Distortion Coefficients:"):
            dist = np.array(list(map(float, lines[idx+1].split())))
            print("\n\n\nDistortion Coefficients:\n\n",dist)
            print("\n")

    return mtx, dist

def cls():
    os.system('cls' if os.name=='nt' else 'clear')
cls()


mtx, dist = get_cam_calib(cam_calib_path)

#Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

ret, frame = cap.read()
if ret:
    
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)
    print(f"Image saved as {image_path}")
else:
    print("Error: Couldn't capture frame.")

cap.release()

dst = undistort_frame(frame, mtx, dist)
cv2.imwrite('calibresult.png', dst)