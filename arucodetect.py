import imutils
import time
import cv2
import undistort as ud
calib_path = "./camera_calibration.txt"

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
mtx, dist = ud.get_camera_calib(calib_path)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()
time.sleep(2.0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break
    frame = imutils.resize(frame, width=720)
    frame = ud.undistort_frame(frame, mtx, dist)
    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary,
    parameters=aruco_params)
    if len(corners) > 0:
        ids = ids.flatten()

        for marker_corner, marker_id in zip(corners, ids):
            corners = marker_corner.reshape((4, 2))
            top_left, top_right, bottom_right, bottom_left = corners

            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))

            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

            cv2.putText(frame, str(marker_id),
                        (top_left[0], top_left[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()