import cv2
import numpy as np
import time

start_time = time.time()

cap = cv2.VideoCapture('video.mp4')


object_detected = False
missing_frames = 0

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# cap.set(cv2.CAP_PROP_POS_MSEC, 28000) 
# cap.set(cv2.CAP_PROP_POS_FRAMES, 870)
low = 0
mid = None
high = total_frames;
booleanValue = None;
while (low <= high) :
    mid = (low + high) // 2;
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    if not ret:
        break
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(f'Processing frame {frame_number}')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        booleanValue = True;
    else:
        booleanValue = False;
    print(booleanValue)
    if (booleanValue):
        low = mid + 1;
    elif (low == mid):
        print(mid);
        current_time_in_milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        break
    else :
        high = mid;
    small_frame = cv2.resize(frame, (320, 320))
    cv2.imshow('Frame', small_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
end_time = time.time()

current_time_in_seconds = current_time_in_milliseconds / 1000

hours, remainder = divmod(current_time_in_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"{str(int(hours)).zfill(2)}:{str(int(minutes)).zfill(2)}:{str(int(seconds)).zfill(2)}")
runtime = end_time - start_time
print(f'The script ran for {runtime} seconds')
# cap.release()
cv2.destroyAllWindows()

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)
# return -low;
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(f'Processing frame {frame_number}')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        object_detected = True
        missing_frames = 0
    else:
        if object_detected:
            missing_frames += 1

        if missing_frames == 5:
            print('The red object is missing!')
            current_time_in_video = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # convert to seconds
            print(f'The red object is missing! Current time in video: {current_time_in_video} seconds')
            object_detected = False
            break

    small_frame = cv2.resize(frame, (320, 320))
    cv2.imshow('Frame', small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()

runtime = end_time - start_time
print(f'The script ran for {runtime} seconds')

cap.release()
cv2.destroyAllWindows()
