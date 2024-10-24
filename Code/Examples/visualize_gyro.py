# importing libraries
import cv2
import numpy as np

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture("C:/Users/dexte/github/RoboSkin/Assets/Video demos/flat_detection.avi")
gyro=np.load("C:/Users/dexte/github/RoboSkin/Assets/Video demos/flat_detection_gyro.npy")
# Check if camera opened successfully
if (cap.isOpened()== False):
	print("Error opening video file")

# Read until video is completed
i=0
while(cap.isOpened()):
	
# Capture frame-by-frame
	ret, frame = cap.read()
	if ret == True:
	# Display the resulting frame
		cv2.putText(frame,str(gyro[i]),(100,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0))
		cv2.imshow('Frame', frame)
		i+=1
	# Press Q on keyboard to exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
			break

# Break the loop
	else:
		break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
