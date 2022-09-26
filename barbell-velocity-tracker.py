# import necessary packages
import numpy as np
import cv2 as cv
import time
from pyzbar import pyzbar
from pyzbar.pyzbar import ZBarSymbol
import argparse
import pandas as pd
import AiPhile

def qr_detection(frame):
	# QR code detector function 
 
	# convert the color image to grayscale image
    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # create QR code object
    detectedQRcode = pyzbar.decode(grayscale)
    for decodedQRcode in detectedQRcode: 
        x, y, w, h = decodedQRcode.rect
        
        # cv.rectangle(image, (x,y), (x+w, y+h), ORANGE, 4)
        points = decodedQRcode.polygon
        if len(points) > 4:
            hull = cv.convexHull(
                np.array([points for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points

        return hull

def detect_qr(processed_frame, display_frame, data_df, current_time):
	"""Detects a QR code within a given frame, draws border around QR code and returns decoded information.

	Args:
		frame (OutputArray): A single frame from the inputted video.
		data_df (DataFrame): A DataFrame containing X, Y, and time data.
		current_time (float): Current time expressed as a float since epoch start. 

	Returns:
		frame (OutputArray): The same frame un-changed from the input.
	"""
	
	barcodes = pyzbar.decode(processed_frame, symbols=[ZBarSymbol.QRCODE])
	for barcode in barcodes:
		print(barcode)
		x, y, w, h = barcode.rect
  
		barcode_info = barcode.data.decode('utf-8')
		cv.rectangle(processed_frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
		
		# Save positions in DataFrame
		data_df.loc[data_df.size/3] = [x , y, current_time]
		
		font = cv.FONT_HERSHEY_DUPLEX
		cv.putText(processed_frame, barcode_info, (x + 6, y - 6), font, 2.0, (255, 0, 0), 1)
	
	return processed_frame, data_df

def main():
	# construct the argument parse and parse the arguments on script call
	ap = argparse.ArgumentParser()
	# For tracking ball from .mp4 video
	ap.add_argument("-v", "--video",
		help="optional path for video file")
	args, unknown = ap.parse_known_args()
	args_dict = vars(args)
 
	# Check if no video was supplied and set to camera
	# else, grab a reference to the video file
	#if not args_dict.get("video", False):
		#camera = cv.VideoCapture(0)
	#else:
		#camera = cv.VideoCapture(args_dict["video"])
 
	# allow the camera/video file to warm up
	time.sleep(2.0)
 
	# Create DataFrame to hold coordinates and time
	data_columns = ['x', 'y', 'time']
	data_df = pd.DataFrame(data = None, columns=data_columns, dtype=float)
 
	#
	ref_point = []
	click = False
	points =()
	camera = cv.VideoCapture(0)
	_, frame = camera.read()
	old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

	lk_params = dict(winSize=(20, 20),
					maxLevel=4,
					criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
	
	camera = cv.VideoCapture(0)
	point_selected = False
	points = [()]
	old_points = np.array([[]])
	qr_detected= False
	# stop_code=False

	frame_counter =0
	start_time =time.time()
	# keep looping until the 'q' key is pressed
	while True:
		frame_counter +=1
		ret, frame = camera.read()
		img = frame.copy()
		# img = cv.resize(img, None, fx=2, fy=2,interpolation=cv.INTER_CUBIC)
		cv.imshow('old frame ', old_gray)
		cv.imshow('img', img)

		gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		# display the image and wait for a keypress
		clone = frame.copy()
		hull_points = qr_detection(frame)
		# print(old_points.size)
		stop_code=False
		if hull_points:
			pt1, pt2, pt3, pt4 = hull_points
			qr_detected= True
			stop_code=True
			old_points = np.array([pt1, pt2, pt3, pt4], dtype=np.float32)
			frame =AiPhile.fillPolyTrans(frame, hull_points, AiPhile.MAGENTA, 0.4)
			AiPhile.textBGoutline(frame, f'Detection: Pyzbar', (30,80), scaling=0.5,text_color=(AiPhile.MAGENTA ))
			cv.circle(frame, pt1, 3, AiPhile.GREEN, 3)
			cv.circle(frame, pt2, 3, (255, 0, 0), 3)
			cv.circle(frame, pt3, 3,AiPhile.YELLOW, 3)
			cv.circle(frame, pt4, 3, (0, 0, 255), 3)
		if qr_detected and stop_code==False:
			# print('detecting')
			new_points, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
			old_points = new_points 
			new_points=new_points.astype(int)
			n = (len(new_points))
			frame =AiPhile.fillPolyTrans(frame, new_points, AiPhile.GREEN, 0.4)
			AiPhile.textBGoutline(frame, f'Detection: Optical Flow', (30,80), scaling=0.5,text_color=AiPhile.GREEN)
			cv.circle(frame, (new_points[0]), 3,AiPhile.GREEN, 2)

		old_gray = gray_frame.copy()
		# press 'r' to reset the window
		key = cv.waitKey(1)
		if key == ord("s"):
			cv.imwrite(f'reference_img/Ref_img{frame_counter}.png', img)

		# if the 'c' key is pressed, break from the loop
		if key == ord("q"):
			break
		fps = frame_counter/(time.time()-start_time)
		AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
		cv.imshow("image", frame)
     
	'''
	while True:
		# Grab current video frame
		(grabbed, frame) = camera.read()
  
		# Check current time
		current_time = time.time() - start
	
		if args_dict.get("video") and not grabbed:
			break

		# Resize frame
		frame = imutils.resize(frame, width=800)
		# preprocessing using opencv
		gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		# blur = cv.GaussianBlur(gray, (5, 5), 0)
		# ret, bw_im = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
  		
		display_frame, data_df = detect_qr(gray, frame, data_df, current_time)
		cv.imshow('Barcode/QR Code Reader', display_frame)
   
		if cv.waitKey(1) & 0xFF == 27:
			break
		
	'''
	camera.release()
	cv.destroyAllWindows()
 
	# Export plot and DataFrame
	data_df.to_csv('Data_Set.csv', sep=",")
		
if __name__ == '__main__':
	main()