# import necessary packages
import numpy as np
import cv2 as cv
import time
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
import argparse
import pandas as pd
import cv_drawing_functions
import imutils
from imutils.video import FPS
from collections import deque
import cv2.aruco as aruco

def findAruco(img, marker_size=4, total_markers=50, draw=True):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
	arucoDict = aruco.Dictionary_get(key)
	arucoParam = aruco.DetectorParameters_create()
	bbox, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
	if draw:
		aruco.drawDetectedMarkers(img, bbox)
 
	cv.imshow("Gray Frame", gray)
	return bbox, ids

def determine_center(corners):
    
	(topLeft, topRight, bottomRight, bottomLeft) = corners
	# convert each of the (x, y)-coordinate pairs to integers
	topRight = (int(topRight[0]), int(topRight[1]))
	bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
	bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
	topLeft = (int(topLeft[0]), int(topLeft[1]))
	
	# compute and draw the center (x, y)-coordinates of the
	# ArUco marker
	cX = int((topLeft[0] + bottomRight[0]) / 2.0)
	cY = int((topLeft[1] + bottomRight[1]) / 2.0)

	return cX, cY

def main():
	# construct the argument parse and parse the arguments on script call
	ap = argparse.ArgumentParser()
	# For tracking ball from .mp4 video
	ap.add_argument("-v", "--video",
		help="optional path for video file")
	# Buffer size corresponds to length of deque
	# Larger buffer = longer lasting bar path
	ap.add_argument("-b", "--buffer", type=int, default=10000,
		help="max buffer size")
	# Allow for Optical Flow estimation
	ap.add_argument("-o", "--optical", type=bool, default=False, 
    	help="allow for Optical Flow estimation when undetected AruCo tag.")
	args, unknown = ap.parse_known_args()
	args_dict = vars(args)

 
	# Check if no video was supplied and set to camera
	# else, grab a reference to the video file
	if not args_dict.get("video", False):
		camera_source = 0
	else:
		camera_source = args_dict["video"]
  
	# Set Optical Flow Estimation based on script call argument
	allow_OF_estimation = args_dict["optical"]
	# Set Optical Flow parameters
	lk_params = dict(winSize=(20, 20),
					maxLevel=4,
					criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
 
	camera = cv.VideoCapture(camera_source)
	time.sleep(2)
	(ret, frame) = camera.read()
	old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
	# Initialize deque for list of tracked coords using buffer size
	coords = deque(maxlen=args_dict["buffer"])
 
	marker_detected= False
	stop_code=False

	# Create DataFrame to hold coordinates and time
	data_columns = ['x', 'y', 'time']
	data_df = pd.DataFrame(data = None, columns=data_columns, dtype=float)

	while not camera_source:
		(ret, frame) = camera.read()

		key = cv.waitKey(1)
		# if the 's' key is pressed, break from the loop
		if key == ord("s"):
			break
	
		cv_drawing_functions.textBGoutline(frame, f'Press S to Begin Tracking', (30,80), scaling=0.5,text_color=(cv_drawing_functions.MAGENTA ))
		cv.imshow("Barbell Velocity Tracker - Main Menu", frame)
 
	start_time = time.time()
	while True:
		(ret, frame) = camera.read()
		gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
  
		# Attempt to Detect AruCo
		bbox, ids = findAruco(frame)
  
		# Check current time
		current_time = time.time() - start_time
  
		stop_code=False
		# loop over the detected ArUCo corners
		if len(bbox) > 0:
    		# flatten the ArUco IDs list
			ids = ids.flatten()
   
			# Allow for Optical Flow in the future
			marker_detected= True
			stop_code=True
  
			for (markerCorner, markerID) in zip(bbox, ids):
				# extract the marker corners (which are always returned
				# in top-left, top-right, bottom-right, and bottom-left
				# order)
				corners = markerCorner.reshape((4, 2))
				cX, cY = determine_center(corners)
				cv.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
    
				data_df.loc[data_df.size/3] = [cX , cY, current_time]
				coords.appendleft((cX, cY))
		if marker_detected and stop_code==False and allow_OF_estimation:
			# print('detecting')
			old_corners = np.array(corners, dtype=np.float32)
			new_corners, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_corners, None, **lk_params)
			corners = new_corners
			new_corners = new_corners.astype(int)
			cX, cY = determine_center(new_corners)
			cv_drawing_functions.textBGoutline(frame, f'Detection: Optical Flow', (30,80), scaling=0.5, text_color=cv_drawing_functions.GREEN)
			cv.rectangle(frame, new_corners[0], new_corners[2], cv_drawing_functions.GREEN, 2)
			cv.circle(frame, (new_corners[0]), 3, cv_drawing_functions.GREEN, 2)
			cv.circle(frame, (cX, cY), 25, cv_drawing_functions.VIOLET, 5)
		
			data_df.loc[data_df.size/3] = [cX , cY, current_time]
			# update the position queue
			coords.appendleft((cX, cY))
   
		# loop over deque for tracked position
		for i in range(1, len(coords)):
		
			# Ignore drawing if curr/past position is None
			if coords[i - 1] is None or coords[i] is None:
				continue
			# Compute line between positions and draw
			cv.line(frame, coords[i - 1], coords[i], (0, 0, 255), 2)


		old_gray = gray_frame.copy()
  
		#print(fps.fps())
		key = cv.waitKey(1)
		# if the 'q' key is pressed, break from the loop
		if key == ord("q"):
			break
  
		cv.imshow("frame", frame)
		cv.resizeWindow("frame", (1000, 800))

if __name__ == '__main__':
	main()