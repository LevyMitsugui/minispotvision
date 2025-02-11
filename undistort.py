import cv2
import numpy as np

def undistort_frame(frame, mtx=None, dist=None):
    if mtx is None:
        raise Exception("mtx could not be loaded")
    if dist is None:
        raise Exception("dist could not be loaded")

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

def get_camera_calib(calib_path):
    try:
        calib_file = open(calib_path, "r")
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