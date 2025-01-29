import numpy as np
import cv2 as cv
import glob
import os

square_size = 0.028

# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Input and output folders
calibfolder = "/home/Levy/Documents/calibimages"
images = glob.glob(os.path.join(calibfolder, "*.jpg"))

output_folder = "/home/Levy/Documents/calibresults"
os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

for idx, fname in enumerate(images):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        cv.drawChessboardCorners(img, (9, 6), corners2, ret)

        output_path = os.path.join(output_folder, f"processed_{idx}.jpg")
        cv.imwrite(output_path, img)  # Save processed image
        print(f"Saved: {output_path}")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)  

print("\nCamera Calibration Parameters:")
print("Reprojection Error:", ret)
print("Camera Matrix (Intrinsic Parameters):\n", mtx)
print("Distortion Coefficients:\n", dist)
print("Rotation Vectors:\n", rvecs)
print("Translation Vectors:\n", tvecs)

calib_file = "./camera_calibration.txt"
with open(calib_file, "w") as f:
    f.write("Camera Calibration Parameters\n")
    f.write(f"Reprojection Error: {ret}\n\n")
    f.write("Camera Matrix (Intrinsic Parameters):\n")
    np.savetxt(f, mtx, fmt="%.6f")
    f.write("\nDistortion Coefficients:\n")
    np.savetxt(f, dist, fmt="%.6f")
    f.write("\nRotation Vectors:\n")
    for rvec in rvecs:
        np.savetxt(f, rvec, fmt="%.6f")
        f.write("\n")
    f.write("\nTranslation Vectors:\n")
    for tvec in tvecs:
        np.savetxt(f, tvec, fmt="%.6f")
        f.write("\n")

print(f"\nCalibration parameters saved to: {calib_file}")

total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print(f"\nReprojection Error: {mean_error:.6f}")