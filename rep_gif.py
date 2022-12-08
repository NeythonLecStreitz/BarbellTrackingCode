import pandas as pd
import cv2 as cv
import time
import imageio
import cv_drawing_functions


def showWindow(winname, img, x, y):
	cv.namedWindow(winname)        # Create a named window
	cv.moveWindow(winname, x, y)   # Move it to (x,y)
	img = cv.flip(img, 0)
	imS = cv.resize(img, (480, 854))
	
	cv_drawing_functions.textBGoutline(imS, "Press 'q' to end looping.", (100,100), scaling=.75,text_color=(cv_drawing_functions.MAGENTA ))
	cv.imshow(winname,imS)
	cv.setWindowProperty(winname, cv.WND_PROP_TOPMOST, 1)
 

def generate_rep_loop(video_path, data_df, rep):
    
	camera = cv.VideoCapture(video_path)
	time.sleep(2)
 
 
	# Get start and end time of rep
	if rep == 0:
		start_time = 0
		end_time = data_df.at[0, 'Time']
	else:
		start_time = data_df.at[rep-1, 'Time']
		end_time = data_df.at[rep, 'Time']
	
	camera.set(cv.CAP_PROP_POS_FRAMES, start_time)
	while True:
		(ret, frame) = camera.read()
  
		# if video supplied and no frame grabbed, video ended so break
		if not ret:
			break
		
		# Retrieve frame number and if within range, show frame
		current_frame = camera.get(cv.CAP_PROP_POS_FRAMES)
		if current_frame >= end_time:
			camera.set(cv.CAP_PROP_POS_FRAMES, start_time)
  
  
		showWindow(f'Rep {rep} Loop:',frame, 900, 0)
		# Press q to exit early
		key = cv.waitKey(1)
		if key == ord("q"):
			break

	cv.destroyAllWindows()