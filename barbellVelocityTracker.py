#------------------------ Import Necessary Libaries -------------------------

# General
import os
import time
from collections import deque
from datetime import date

# Statistics/Plotting
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output


# Computer Vision
import cv2 as cv
import cv2.aruco as aruco
import ffmpeg

# Custom functions
import cv_drawing_functions
import output_plots

	
#--------------------------------- Functions -------------------------------------    
def is_rack_derack(history):
	"""
 	Checks current displacement to see if there is more movement in the x direction than y. 
	This indicates deracking/racking/setting up, so it clears history to not confuse the rep detection algorithm.
  	"""

	pos = 0
	x_displacement = 0
	y_displacement = 0

	while pos < len(history):
		x_displacement += abs(history[pos][0])
		y_displacement += abs(history[pos][1])
		pos += 1

	if (abs(x_displacement) - abs(y_displacement)) >= 0:
		print('Racking currently...')
		return True

	return False

def calculate_velocity(coord_deque, mmpp, velocity_list, rep_rest_time, reps, analyzed_rep, change_in_phase):
	"""
 	Calculates current velocity by taking pixel distance between current and previous coordinate point and multiplying by mmpp.
	Since each frame takes 1/FPS seconds, we have mm/s every 1/FPS.
	"""
	rep_rest_threshold = 80.0
	rep = False
	calculated_velocity = (0, 0, 0)
	inflection = False
 
	curr_x, last_x = coord_deque[0][0], coord_deque[1][0]
	curr_y, last_y = coord_deque[0][1], coord_deque[1][1]
	
	x_disp = last_x - curr_x
	y_disp = last_y - curr_y
	
	y_distance = y_disp * mmpp
	x_distance = x_disp * mmpp

	#if abs(y_distance) > barbell_perimeter / 32
	if abs(y_distance) > 2:
		rep_rest_time = 0.0
		analyzed_rep = False
	 
		distance = math.sqrt(x_disp ** 2 + y_disp ** 2) * mmpp
		velocity = distance * vid_fps / 1000
		y_velocity = y_distance * vid_fps / 1000
		
		# Minute differences don't count as movement...
		if -0.05 < y_velocity < 0.05:
			y_velocity = 0
			rep_rest_time += 1 / vid_fps * 1000

		velocity_list.append((int(x_distance), int(y_distance), y_velocity))
		if is_inflection(y_velocity, change_in_phase):
			inflection = True
			change_in_phase = not change_in_phase
			rep, calculated_velocity = analyze_for_rep(velocity_list, reps)
	else:
		# Count how many milliseconds we're at 0 velocity
		rep_rest_time += 1 / vid_fps * 1000
		# analyze for last rep when we appear to rest for a threshold time
		if (rep_rest_time > rep_rest_threshold) and not analyzed_rep:
			analyzed_rep = True
			if is_rack_derack(velocity_list):
				velocity_list = []
			rep, calculated_velocity = analyze_for_rep(velocity_list, reps)
  
	return velocity_list, rep, calculated_velocity, rep_rest_time, analyzed_rep, change_in_phase, inflection
	

def showInMovedWindow(winname, img, x, y, out, tracking_toggled=False, add_text=False):
	'''
	Creates a dummy window for displaying the velocity tracking video. 
	Without this resized window, an iPhone video cannot be properly viewed on a laptop screen.
	'''
	
	# Create dummy window, move it to (x, y), flip and resize
	cv.namedWindow(winname)
	cv.moveWindow(winname, x, y)   
	img = cv.flip(img, 0)
	imS = cv.resize(img, (480, 854))
 
	# Allows for inclusion of text on the screen.
	if add_text:
		# If tracking on, show in Magenta
		if tracking_toggled:
			cv_drawing_functions.textBGoutline(imS, f'{add_text}', (25,100), scaling=.75,text_color=(cv_drawing_functions.MAGENTA ))
		# If tracking off, show in Blue
		else:
			cv_drawing_functions.textBGoutline(imS, f'{add_text}', (25,100), scaling=.75,text_color=(cv_drawing_functions.BLUE ))
	
	out.write(imS)
	cv.imshow(winname,imS)
 
	return out
 
def showStats(data_df):
	'''
	Displays current repetition statistics in a separate window while video is playing.
	'''

	cv.namedWindow("Velocity Stats:")        # Create a named window
	cv.moveWindow("Velocity Stats:", 700, 300) 
 
	# Create All-Black Image
	blank_frame = np.full((300,320,3), fill_value=(255, 255,255), dtype=np.uint8)
	stats = [
			("Reps", "{}".format(data_df.iloc[0]['Rep']), (0, 255, 0)),
			("Last AVG Con Velocity", "{:.2f} m/s".format(float(data_df['Avg Velocity'])), (0, 255, 0)),
			("Last PEAK Con Velocity", "{:.2f} m/s".format(float(data_df['Peak Velocity'])), (0, 255, 0)),
			("Last Displacement", "{:.2f} mm".format(float(data_df['Displacement'])), (0, 255, 0)),
		]
 
	if data_df.iloc[0]['Rep'] > 1:
		stats.append(("AVG Velocity Loss", "{:.2f} %".format(float(data_df['Avg Velocity Loss'])), (0, 255, 0))) 
		stats.append(("PEAK Velocity Loss", "{:.2f} %".format(float(data_df['Peak Velocity Loss'])), (0, 255, 0)))

	# loop over the info tuples and draw them on our frame
	for (i, (k, v, c)) in enumerate(stats):
		text = "{}: {}".format(k, v)
		cv_drawing_functions.textBGoutline(blank_frame, text, (10, 300 - ((i * 40) + 20)), scaling=.5, text_color=(cv_drawing_functions.BLACK))
 
	cv.imshow("Velocity Stats:", blank_frame)
 
def show_set_avg(data_df):
	'''
	Displays aggregate statistics across the entire set after the video finishes playing.
	'''
	
	cv.namedWindow("Set Velocity Averages")
	cv.moveWindow("Set Velocity Averages", 1000, 300)
	
	blank_frame = np.full((300, 320, 3), fill_value=(255, 255, 255), dtype=np.uint8)
	stats = [
			("Total Reps", "{}".format(int(data_df['Rep'].max())), (0, 255, 0)),
			("Average Rep Velocity", "{:.2f} m/s".format(float(data_df['Avg Velocity'].mean())), (0, 255, 0)),
			("Average Peak Velocity", "{:.2f} m/s".format(float(data_df['Peak Velocity'].mean())), (0, 255, 0)),
			("Max Velocity Loss", "{:.2f} %".format(float(data_df['Avg Velocity'].max()) * 100), (0, 255, 0))
		]
	
	
	for (i, (k, v, c)) in enumerate(stats):
		text = "{}: {}".format(k, v)
		cv_drawing_functions.textBGoutline(blank_frame, text, (10, 300 - ((i * 40) + 20)), scaling=.5, text_color=(cv_drawing_functions.BLACK))
	
	cv.imshow("Set Velocity Averages", blank_frame)
 
def findAruco(img, marker_size=6, total_markers=50):
	'''
	Detects AruCo tag within given frame based on provided AruCo dictionary.
	Returns all AruCo tag candidates, but for our purposes should only be one tag.
 	'''
    
    # Convert frame to grayscale
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
	arucoDict = aruco.Dictionary_get(key)
	arucoParam = aruco.DetectorParameters_create()
 
	# Looks for markers from the given AruCo dictionary.
	bbox, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
 
	# Return only the accepted AruCo detections
	return bbox, ids

def determine_center(corners):
	'''
	Simply takes given AruCo corner detection and finds the centermost point of the rectangle.
 	'''	
 
 
	(topLeft, topRight, bottomRight, bottomLeft) = corners
	# convert each of the (x, y)-coordinate pairs to integers
	topRight = (int(topRight[0]), int(topRight[1]))
	bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
	bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
	topLeft = (int(topLeft[0]), int(topLeft[1]))
	
	# compute the cX and cY values of the AruCo center
	cX = int((topLeft[0] + bottomRight[0]) / 2.0)
	cY = int((topLeft[1] + bottomRight[1]) / 2.0)

	return cX, cY

def calculate_velocity_averages(avg_velocities, peak_velocities, reps):
	'''
	After completion of a single repetition, calculates rep aggregates including average velocity, peak velocity
	and if rep 2 or above, calculates velocity loss from previous repetition.
 	'''
	
	# If first rep, calculate only average, peak, and first velocities.
	if reps == 1:
		avg_velocity = avg_velocities[0]
		peak_velocity = peak_velocities[0]
		# first_velocity = avg_velocity
		return avg_velocity, peak_velocity, None, None
  
	# If rep >= 2, calculate average, peak, and velocity losses.
	else:
		avg_velocity = avg_velocities[-1]
		peak_velocity = peak_velocities[-1]
		avg_vel_loss = (avg_velocities[0] - avg_velocities[-1]) / avg_velocities[0] * 100
		peak_vel_loss = (peak_velocities[0] - peak_velocities[-1]) / peak_velocities[0] * 100
	
		return avg_velocity, peak_velocity, avg_vel_loss, peak_vel_loss
	

def is_inflection(y_velocity, is_eccentric_portion):
	'''
	Determines if an inflection has occured. Used in the automatic detection of partial and full repetitions.
 	'''
	
	# If in eccentric portion, velocity should be positive. If velocity is negative, then no longer in eccentric (thus, inflection point).
	if is_eccentric_portion:
		if y_velocity < 0:
			return True
		else:
			return False
	# If in concentric portion, velocity should be negative. If velocity is positive, then no longer in concentric (thus, inflection point).
	else:
		if y_velocity >= 0:
			return True
		else:
			return False

def analyze_for_rep(velocity_list, reps):
	'''
	Determines current phase of the lift and upon completion of both phases, automatically detects a full repetition.
	'''
 
	pos = 0
	eccentric_phase = False
	concentric_phase = False
	is_eccentric_portion = False
	displacement = 0
	eccentric_displacement = 0
	concentric_displacement = 0
	horizontal_displacement = 0
	velocities = []
	error = 0
	vector_threshold = 8

	# Only analyze if sufficient frames have been analyzed for velocity.
	if len(velocity_list) < 2 * vector_threshold:
		return(False, (0.0, 0.0, 0))

	# Method
	# 1. determine whether in eccentric vs. concentric phase by looking at last 8 points
	# 2. keep reading and ensure each point matches initial direction up until inflection point
	# 3. Read all points after inflection up until next inflection or end of history
	# 4. Use criteria to determine if it's a rep
	for calculated_frame in range(1, vector_threshold):
		displacement += velocity_list[-calculated_frame][2]
		
	if displacement > 0:
		is_eccentric_portion = True
	elif displacement < 0:
		is_eccentric_portion = False
	else:
		# need more data to determine if it's a rep
		return(False, (0.0, 0.0, 0))

	# For every frame with calculated velocity, loop through to verify in correct phase, and switch if change in direction detected.
	while True:
		pos += 1

		if pos > len(velocity_list):
			break

		# Continue reading points in eccentric phase until inflection occurs (we transition to concentric)
		if not eccentric_phase:
			# Check for inflection after at least 150mm of displacement has occured.
			if eccentric_displacement >= 150 and is_inflection(velocity_list[-pos][2], is_eccentric_portion):
				eccentric_phase = True
			else:
				# Checks for inflection without sufficient displacement (change in direction happening haphazardly)
				if is_inflection(velocity_list[-pos][2], is_eccentric_portion):
					if error > 3:
						break
					error += 1
					continue
				else:
					# No inflection detected yet, continue adding eccentric_displacement
					eccentric_displacement += abs(velocity_list[-pos][1])
					horizontal_displacement += abs(velocity_list[-pos][0])
					if is_eccentric_portion:
						# Store frame-by-frame velocity to list
						velocities.append(abs(velocity_list[-pos][2]))
					continue
		
		# Transition to concentric portion of the lift.
		if not concentric_phase:
			# Check for inflection if at least 150mm of displacement has occured or we are at end of velocity history.
			if (concentric_displacement >= 150 and is_inflection(velocity_list[-pos][2], not is_eccentric_portion)) or (pos == len(velocity_list) and concentric_displacement >= 150):
				concentric_phase = True
			else:
				# If no inflection yet, continue adding to concentric displacement total.
				concentric_displacement += abs(velocity_list[-pos][1])
				if not is_eccentric_portion:
					# Store frame-by-frame velocity to list
					velocities.append(abs(velocity_list[-pos][2]))
				continue

		# Once both phases have occured and the difference between them is less than 100mm (essentially starting and ending at same place...), rep detected.
		if eccentric_phase and concentric_phase and abs(concentric_displacement - eccentric_displacement) < 100:
			print("Found rep {}! first: {} mm, second: {} mm".format(reps + 1, eccentric_displacement, concentric_displacement))
	
			# Store last/current displacement based on last phase detected
			if is_eccentric_portion:
				last_displacement = eccentric_displacement
			else:
				last_displacement = concentric_displacement

			avg_vel = sum(velocities) / len(velocities)
			peak_vel = max(velocities)
			return(True, (avg_vel, peak_vel, last_displacement))

	return(False, (0.0, 0.0, 0))

def pixel_to_mm(bbox):
	aruco_perimeter = cv.arcLength(bbox, True)
	
	mmpp = aruco_perimeter /barbell_perimeter
	
	return mmpp

def convert_to_mp4(mp4_file):
    name, ext = os.path.splitext(mp4_file)
    out_name = name + "_converted" + ".mp4"
    ffmpeg.input(mp4_file).output(out_name).run()
    print("Finished converting {}".format(mp4_file))

# -------------------------------- Driving Function ----------------------------------------------
def main(video_path='na', set_weight=0, save_data=False, save_folder=''):
	'''
	Driver function for the entire barbell velocity process. Responsible for retrieving video frames, detecting or estimating the aruCo tag coordinates,
	and drawing the bar path to the frame.
	'''
    
	print("STARTING UP!")
	# Check if no video was supplied and set to camera
	# else, grab a reference to the video file
	if video_path == 'na':
		camera_source = 0
	else:
		camera_source = video_path
		save_folder = os.path.split(video_path)[0]
  
	out = cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'mpv4'), 30, (480, 854))
  
	# Set Optical Flow parameters
	lk_params = dict(winSize=(20, 20),
					maxLevel=4,
					criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.01))
 
	# Initialize camera and grayframe
	camera = cv.VideoCapture(camera_source)
 
	time.sleep(2)
	(ret, frame) = camera.read()
	old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
 
	# Initialize deque for list of tracked coords and velocity lists
	coords = deque(maxlen=10000)
	velocity_list = []
	avg_velocities = []
	peak_velocities = []
 
	# Set Boolean Variables
	marker_detected= False
	stop_code=False
	change_in_phase = False
	analyzed_rep = False
	first_detection = False
	start_tracking = False
	walk_out = False
 
	# Set Numerical Variables
	rep_rest_time = 0.0
	reps = 0
	avg_vel, peak_vel, displacement, avg_vel_loss, peak_vel_loss = 0, 0, 0, pd.NA, pd.NA
	update_mmpp = 0
 	
	# Initialize Barbell radius
	global barbell_perimeter 
	barbell_perimeter = 300 # MM
 
	# Check video FPS
	global vid_fps
	vid_fps = int(camera.get(cv.CAP_PROP_FPS))

	# Create DataFrame to hold coordinates and time
	data_columns = ['Rep', 'Center-Coord', 'Avg Velocity', 'Peak Velocity', 'Avg Velocity Loss', 'Peak Velocity Loss', 'Displacement', 'Time']
	data_df = pd.DataFrame(data = None, columns=data_columns)
	
	temp_col = ['cX', 'cY', 'Reps']
	coord_df = pd.DataFrame(data=None, columns = temp_col)
  
	while True:
		(ret, frame) = camera.read()
  
		# if video supplied and no frame grabbed, video ended so break
		if camera_source and not ret:
			break
   
		gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		# Attempt to Detect AruCo
		bbox, ids = findAruco(frame)
  
		stop_code=False
		# loop over the detected ArUCo corners
		if len(bbox) > 0:
			# flatten the ArUco IDs list
			ids = ids.flatten()
   
			# Allow for Optical Flow in the future
			marker_detected= True
			stop_code=True
  
			for (markerCorner, markerID) in zip(bbox, ids):
				# extract the AruCo tag corners and determine center coordinate
				corners = markerCorner.reshape((4, 2))
				cX, cY = determine_center(corners)
	
				# Append the first coords twice so that first displacement == 0
				if first_detection == False:
					coords.appendleft((cX, cY))
					first_detection = True
	
				# Draw bounding box on AruCo tag and center circle
				aruco.drawDetectedMarkers(frame, bbox)
				cv.circle(frame, (cX, cY), 4, cv_drawing_functions.RED, 1)
	
# -------------------- Optical Flow --------------------
		if marker_detected and stop_code==False:
			old_corners = np.array(corners, dtype=np.float32)
			new_corners, status, error = cv.calcOpticalFlowPyrLK(old_gray, gray_frame, old_corners, None, **lk_params)
   
			# Calculate Center coords
			corners = new_corners
			new_corners = new_corners.astype(int)
   
			cX, cY = determine_center(new_corners)
			pts = new_corners.reshape((-1,1,2))
			
			# Draw bounding box around aruCo tag and center circle
			cv.polylines(frame,[pts],True,(0,255,0), 2)
			cv.circle(frame, (cX, cY), 4, cv_drawing_functions.RED, 1)
		
# -------------------- Velocity and Statistic Calculation --------------------
		if marker_detected:
			# Append coordinate point to deque
			coords.appendleft((cX, cY))
   
			# Find MMPP conversion every three frames
			if update_mmpp == 0:
				mmpp = pixel_to_mm(corners)
			update_mmpp = (update_mmpp + 1) % 3

			# Calculate velocity and returns 
			velocity_list, rep, calculated_velocity, rep_rest_time, analyzed_rep, change_in_phase, inflection = calculate_velocity(coords, mmpp, velocity_list, rep_rest_time, reps, analyzed_rep, change_in_phase)

			# If rep detected, add to reps and display current velocity averages per rep
			if rep:
				velocity_list = []
				reps += 1 # Add a rep
				avg_velocities.append(calculated_velocity[0])
				peak_velocities.append(calculated_velocity[1])
				displacement = calculated_velocity[2]

				avg_vel, peak_vel, avg_vel_loss, peak_vel_loss = calculate_velocity_averages(avg_velocities, peak_velocities, reps)

				# Save Data to DataFrame
				# rep_time = time.time() - start_time
				frame_num = camera.get(cv.CAP_PROP_POS_FRAMES)
				data_df.loc[len(data_df.index)] = [reps, (cX, cY), avg_vel, peak_vel, avg_vel_loss, peak_vel_loss, displacement, frame_num]
				showStats(data_df.tail(1))
    
			if start_tracking:
				coord_df.loc[len(coord_df.index)] = [cX, cY, reps]
				
				if walk_out == False:
					frame_num = camera.get(cv.CAP_PROP_POS_FRAMES)
					data_df.loc[len(data_df.index)] = [reps, (cX, cY), avg_vel, peak_vel, avg_vel_loss, peak_vel_loss, displacement, frame_num]	
					walk_out = True
# -------------------- Path Tracking --------------------

		# loop over deque for tracked position
		for i in range(1, len(coords)):
			
			# Ignore drawing if curr/past position is None
			if coords[i - 1] is None or coords[i] is None:
				continue
			# Compute line between positions and draw
			cv.line(frame, coords[i - 1], coords[i], (0, 0, 255), 5)

# -------------------- Keyboard Interrupts --------------------
  
		# if the 'q' key is pressed, end tracking (and video)
		key = cv.waitKey(1)
		if key == ord("q"):
			break
		elif key == ord('s'):
			print('Tracking toggled.')
			start_tracking = not start_tracking
		
# -------------------- Show Frame to screen --------------------

		# Save current gray frame as old (for Optical Flow)
		old_gray = gray_frame.copy()
  
		# Show video frame in new window (to resize properly)
		if not start_tracking:
			out = showInMovedWindow('Barbell Velocity Tracker:',frame, 200, 0, out, tracking_toggled=False, add_text="Press 's' to toggle tracking ON.")
		else: 
			out = showInMovedWindow('Barbell Velocity Tracker:',frame, 200, 0, out, tracking_toggled=True, add_text="Press 's' to toggle tracking OFF.")

# -------------------- Save Data and Close Windows --------------------
	# Release camera and destroy webcam video
	out.release()
	camera.release()
 
	cv.destroyWindow("Barbell Velocity Tracker:")
	cv.destroyWindow("Velocity Stats:")
 
	while True:
		key = False
		show_set_avg(data_df)
  
		# if the 'q' key is pressed, end tracking (and video)
		key = cv.waitKey(0)
		if key:
			break
	cv.destroyAllWindows()
   
	if save_data:
		# Save complete dataset.
		today = date.today()
		date_format = today.strftime("%b-%d-%Y")
		filepath = save_folder + f"/velocity_data_{set_weight}lbs_{date_format}.csv"
		filepath2 = save_folder + f"/coord_data_{set_weight}lbs_{date_format}.csv"
		data_df.to_csv(filepath, sep=",")
		coord_df.to_csv(filepath2, sep=",")

	# output_plots.create_dash_env(data_df, coord_df, video_path, set_weight)
	return data_df, coord_df

if __name__ == '__main__':
	main()
 