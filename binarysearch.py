import cv2
import numpy as np
import time

start_time = time.time()
cap = cv2.VideoCapture('test/'+'vanishingbicycle.mp4') # Sesuaikan dengan nama file video
object_detected = False
checked = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
low = 0
mid = None
high = total_frames;
booleanValue = None;
while (low <= high) :
    checked += 1
    mid = (low + high) // 2;
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    if not ret:
        break
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(f'Processing frame {frame_number}')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_color = np.array([10, 100, 20])  # Sesuaikan dengan warna objek
    # upper_color = np.array([25, 255, 255]) # Sesuaikan dengan warna objek

    # mask = cv2.inRange(hsv, lower_color, upper_color)
    # Convert the image to HSV format
    # Convert the image to HSV format
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for blue color
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Now you can use lower_blue and upper_blue in cv2.inRange()
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        booleanValue = True;
    else:
        booleanValue = False;
    if (booleanValue):
        low = mid + 1;
    elif (low == mid):
        print(mid);
        current_time_in_milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        break
    else :
        high = mid;
    current_time_in_milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
    small_frame = cv2.resize(frame, (320, 320))
    cv2.imshow('Frame', small_frame)
    cv2.waitKey(1)
end_time = time.time()
current_time_in_seconds = current_time_in_milliseconds / 1000

hours, remainder = divmod(current_time_in_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"{str(int(hours)).zfill(2)}:{str(int(minutes)).zfill(2)}:{str(int(seconds)).zfill(2)}")
runtime = end_time - start_time
print(f'The script ran for {runtime} seconds')
print(f'checked : {checked}')
cap.release()
cv2.destroyAllWindows()