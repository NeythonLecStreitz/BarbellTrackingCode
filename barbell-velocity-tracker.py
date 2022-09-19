# import necessary packages
import numpy as np
import cv2 as cv
import imutils
import time
from pyzbar import pyzbar
import argparse
import pandas as pd

def read_qr(frame):
	"""Detects a QR code within a given frame, draws border around QR code and returns decoded information

	Args:
		frame (_type_): A single frame from the inputted video.

	Returns:
		frame (_type_): _description_
	"""
	
	barcodes = pyzbar.decode(frame)
	for barcode in barcodes:
		x, y, w, h = barcode.rect
  
		barcode_info = barcode.data.decode('utf-8')
		cv.rectangle(frame, (x, y),(x+w, y+h), (0, 255, 0), 2)
			
		
		font = cv.FONT_HERSHEY_DUPLEX
		cv.putText(frame, barcode_info, (x + 6, y - 6), font, 2.0, (255, 255, 255), 1)
	
	return frame

def frame_preprocess(frame):
    
    # Resize frame to 800/600
	frame = imutils.resize(frame, width=800)
 
	# Convert to grayscale
	grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
	# Gaussian Blur
	blurred = cv.GaussianBlur(grayscale, (11, 11), 0)
 
	# Binary Threshold
	ret, processed_frame = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
 
	return processed_frame

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
	if not args_dict.get("video", False):
		camera = cv.VideoCapture(0)
	else:
		camera = cv.VideoCapture(args_dict["video"])
 
	# allow the camera/video file to warm up
	time.sleep(2.0)
 
	# Create DataFrame to hold coordinates and time
	data_columns = ['x', 'y', 'time']
	data_df = pd.DataFrame(data = None, columns=data_columns, dtype=float)

	# Read time at video start
	start = time.time()
	
	while True:
		(grabbed, frame) = camera.read()
	
		if args_dict.get("video") and not grabbed:
			break

		processed_frame = frame_preprocess(frame)
  		
		frame = read_qr(processed_frame)
		cv.imshow('Barcode/QR Code Reader', frame)
   
		if cv.waitKey(1) & 0xFF == 27:
			break
		
	camera.release()
	cv.destroyAllWindows()
		
if __name__ == '__main__':
	main()