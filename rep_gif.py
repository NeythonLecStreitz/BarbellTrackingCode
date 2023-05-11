import pandas as pd
import cv2 as cv
import time
import imageio
import cv_drawing_functions


def showWindow(winname, img, x, y):
	'''
	Shows the looped repetition in a moved and scaled window.
 	'''	
 
	cv.namedWindow(winname)        # Create a named window
	cv.moveWindow(winname, x, y)   # Move it to (x,y)
	#img = cv.flip(img, 0)
	imS = cv.resize(img, (480, 854))
	
	cv_drawing_functions.textBGoutline(imS, "Press 'q' to end looping.", (100,900), scaling=.75,text_color=(cv_drawing_functions.MAGENTA ))
	cv.imshow(winname, imS)
	cv.setWindowProperty(winname, cv.WND_PROP_TOPMOST, 1)


def generate_rep_loop(video_path, data_df, rep_str):
	'''
	Plays a loop of a single repetition based on the frame numbers from the barbell velocity tracking output (data_df).
	'''
	camera = cv.VideoCapture(video_path)
	time.sleep(2)
 
	time_df = data_df.groupby('Rep')['Time'].max()
	print(time_df)
 
	if rep_str == 'All Reps':
		start_time = 0
		end_time = time_df.iloc[-1]
	elif int(rep_str[-1]) == 1:
		start_time = 0
		end_time = time_df.iloc[0]
	else:
		rep = int(rep_str[-1]) - 1
		prev_rep = rep - 1
		start_time = time_df.iloc[prev_rep]
		end_time = time_df.iloc[rep]
  
	print(f"Start Time: {start_time}")
	print(f"End Time: {end_time}")
	
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
  
  
		showWindow(f'{rep_str} Loop:',frame, 900, 0)
		# Press q to exit early
		key = cv.waitKey(1)
		if key == ord("q"):
			break

	cv.destroyAllWindows()