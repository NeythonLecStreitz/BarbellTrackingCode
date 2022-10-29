#------------------------ Import Necessary Libaries -------------------------

# General
import os
import time
from collections import deque

# Statistics/Plotting
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Computer Vision
import cv2 as cv
import cv2.aruco as aruco

# Custom functions
import cv_drawing_functions
import velocity_stops
	
#--------------------------------- Functions -------------------------------------
def is_rack_derack(history):
	# When there is a pause or inflection, let's check our previous points to see
	# if there is more movement in the x direction than the y.
	# This would indicate deracking/racking/setting up and we want to clear the history
	# of any small movements that would confuse the algorithm from detecting the start
	# of a rep, whether it be a concentric or eccentric first exercise.
	pos = 0
	x_displacement = 0
	y_displacement = 0

	while pos < len(history):
		x_displacement += abs(history[pos][0])
		y_displacement += abs(history[pos][1])
		pos += 1

	if (abs(x_displacement) - abs(y_displacement)) >= 0:
		return True

	return False

def calculate_velocity(coord_deque, mmpp, velocity_list, rep_rest_time, reps, analyzed_rep, change_in_phase):
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
	
	if abs(y_distance) > barbell_perimeter / 32:
		rep_rest_time = 0.0
		analyzed_rep = False
     
		distance = math.sqrt(x_disp ** 2 + y_disp ** 2) * mmpp
		velocity = distance * vid_fps / 1000
		y_velocity = y_distance * vid_fps / 1000
		
		# Minute differences don't count as movement...
		if -0.02 < y_velocity < 0.02:
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
				print("Detected significant x movement over y, resetting history...")
				velocity_list = []
			rep, calculated_velocity = analyze_for_rep(velocity_list, reps)
  
	return velocity_list, rep, calculated_velocity, rep_rest_time, analyzed_rep, change_in_phase, inflection
	

def showInMovedWindow(winname, img, x, y, vid_writer=None, save_data=False, add_text=False):
	cv.namedWindow(winname)        # Create a named window
	cv.moveWindow(winname, x, y)   # Move it to (x,y)
	img = cv.flip(img, 0)
	imS = cv.resize(img, (480, 854))
 
	if add_text:
		cv_drawing_functions.textBGoutline(imS, f'{add_text}', (100,100), scaling=.75,text_color=(cv_drawing_functions.MAGENTA ))
	cv.imshow(winname,imS)
 
	if vid_writer is not None:
		vid_writer.write(imS)
 
def showStats(data_df, rpe, rir):
		
	cv.namedWindow("Velocity Stats:")        # Create a named window
	cv.moveWindow("Velocity Stats:", 800, 300) 
 
    # Create All-Black Image
	blank_frame = np.zeros((300,300,3), dtype=np.uint8)
	stats = [
		("Reps", "{}".format(data_df.iloc[0]['Rep']), (0, 255, 0)),
        ("Last AVG Con Velocity", "{:.2f} m/s".format(float(data_df['Avg Velocity'])), (0, 255, 0)),
        ("Last PEAK Con Velocity", "{:.2f} m/s".format(float(data_df['Peak Velocity'])), (0, 255, 0)),
        ("Last Displacement", "{:.2f} mm".format(float(data_df['Displacement'])), (0, 255, 0))
    ]
 
	if data_df.iloc[0]['Rep'] > 1:
		stats.append(("AVG Velocity Loss", "{:.2f} %".format(float(data_df['Avg Velocity Loss'])), (0, 255, 0))) 
		stats.append(("PEAK Velocity Loss", "{:.2f} %".format(float(data_df['Peak Velocity Loss'])), (0, 255, 0)))
  
	if rpe != 'N/A':
		stats.append(("Approx. RPE/RIR", "{}/{}".format(rpe, rir), (0, 255, 0)))

    # loop over the info tuples and draw them on our frame
	for (i, (k, v, c)) in enumerate(stats):
		text = "{}: {}".format(k, v)
		cv_drawing_functions.textBGoutline(blank_frame, text, (10, 300 - ((i * 40) + 20)), scaling=.5, text_color=(cv_drawing_functions.BLUE))
 
	cv.imshow("Velocity Stats:", blank_frame)

def findAruco(img, marker_size=6, total_markers=50):
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
	arucoDict = aruco.Dictionary_get(key)
	arucoParam = aruco.DetectorParameters_create()
	bbox, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParam)
 
	# cv.imshow("Gray Frame", gray)
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
	# radius = topLeft[1] - cY

	return cX, cY

def calculate_velocity_averages(avg_velocities, peak_velocities, reps):
	# rep_list: (average velocity, peak velocity, ending Y displacement)
	
	# If first rep, calculate only average, peak, and first velocities.
	if reps == 1:
		avg_velocity = avg_velocities[0]
		peak_velocity = peak_velocities[0]
		# first_velocity = avg_velocity
		return avg_velocity, peak_velocity, None, None
  
	# If second + rep, calculate average, peak, and velocity losses.
	else:
		avg_velocity = avg_velocities[-1]
		peak_velocity = peak_velocities[-1]
		avg_vel_loss = (avg_velocities[0] - avg_velocities[-1]) / avg_velocities[0] * 100
		peak_vel_loss = (peak_velocities[0] - peak_velocities[-1]) / peak_velocities[0] * 100
	
		return avg_velocity, peak_velocity, avg_vel_loss, peak_vel_loss
	

def is_inflection(y_velocity, is_eccentric_portion):
	
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
			# Check for inflection after at least 200mm of displacement has occured.
			if eccentric_displacement > 150 and is_inflection(velocity_list[-pos][2], is_eccentric_portion):
				# print("First Phase Displacement: {}".format(first_phase_disp))
				eccentric_phase = True
			else:
				# Checks for inflection without sufficient displacement (change in direction happening haphazardly)
				if is_inflection(velocity_list[-pos][2], is_eccentric_portion):
					if error > 3:
						print("Sufficient movement error detected...")
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
			# Check for inflection if at least 200mm of displacement has occured or we are at end of history (movement about to stop)
			# or we're on the last point in history
			if (concentric_displacement >= 200 and is_inflection(velocity_list[-pos][2], not is_eccentric_portion)) or (pos == len(velocity_list) and concentric_displacement >= 200):
				concentric_phase = True
			else:
				# If no inflection yet, continue adding to concentric displacement total.
				concentric_displacement += abs(velocity_list[-pos][1])
				if not is_eccentric_portion:
					# Store frame-by-frame velocity to list
					velocities.append(abs(velocity_list[-pos][2]))
				continue

		# Once both phases have occured and the difference between them is less than 100mm (essentially starting and ending at same place...)
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


def generate_plots(data_df, save_data=True, save_folder=''):
    # Generate Rep vs. Velocity Plot
	fig1 = plt.figure(1)
	plt.plot(data_df['Rep'], data_df['Avg Velocity'], color='r', label='Avg Vel')
	plt.plot(data_df['Rep'], data_df['Peak Velocity'], color='g', label='Peak Vel')
	plt.title('Velocity as Reps Increase')
	plt.xlabel('Rep')
	plt.ylabel('Velocity m/s')
	plt.legend()
	if save_data:
		save_to = save_folder + "/Reps_VS_Velocity.svg"
		plt.savefig(save_to, transparent=True)
 
	# Generate Velocity Loss plots
	fig2 = plt.figure(2)
	plt.plot(data_df['Rep'], data_df['Avg Velocity Loss'], color='r', label='Avg Vel Loss')
	plt.plot(data_df['Rep'], data_df['Peak Velocity Loss'], color='g', label='Peak Vel Loss')
	plt.title("Velocity Loss as Reps Increase")
	plt.xlabel('Rep')
	plt.ylabel('Velocity Loss %')
	plt.legend()
	if save_data:
		save_to = save_folder + "/Reps_VS_Velocity.svg"
		plt.savefig(save_to, transparent=True)
	plt.show()


def pixel_to_mm(bbox):
    aruco_perimeter = cv.arcLength(bbox, True)
    
    mmpp = aruco_perimeter /barbell_perimeter
    
    return mmpp

# -------------------------------- Driving Function ----------------------------------------------
def main(video_path='na', use_rpe=False, save_data=True, plot_data=False, save_rpe=False):
	# Check if no video was supplied and set to camera
	# else, grab a reference to the video file
	if video_path == 'na':
		camera_source = 0
	else:
		camera_source = video_path
		save_folder = os.path.split(video_path)[0]
  
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
	begin_tracking = False
	show_path = True
 
	# Set Numerical Variables
	rep_rest_time = 0.0
	reps = 0
	avg_vel, peak_vel, avg_vel_loss, peak_vel_loss = 0, 0, pd.NA, pd.NA
	update_mmpp = 0
	rpe, rir = 'N/A', 'N/A'
 
	# Create video writing object if saving video to file
	if save_data:
		vid_res = (480, 854)
		result = cv.VideoWriter('tracked_video.mp4', 
							cv.VideoWriter_fourcc(*'mp4v'),
							30, vid_res)
 
 	
	# Initialize Barbell radius
	global barbell_perimeter 
	barbell_perimeter = 300 # MM
 
	# Check video FPS
	global vid_fps
	vid_fps = int(camera.get(cv.CAP_PROP_FPS))

	
	# Create DataFrame to hold coordinates and time
	data_columns = ['Rep', 'Center-Coord', 'Avg Velocity', 'Peak Velocity', 'Avg Velocity Loss', 'Peak Velocity Loss', 'Displacement', 'Time']
	data_df = pd.DataFrame(data = None, columns=data_columns)
 
	# Load RPE Data
	if use_rpe:
		file = save_folder + '/rpe_data.csv'
		rpe_df = pd.read_csv(file)


	start_time = time.time()
	while True:
		(ret, frame) = camera.read()
  
		# if video supplied and no frame grabbed, video ended so break
		if camera_source and not ret:
			break
   
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
				# extract the AruCo tag corners and determine center coordinate
				corners = markerCorner.reshape((4, 2))
				cX, cY = determine_center(corners)
	
				# Append the first coords twice so that first displacement == 0
				if begin_tracking == False:
					coords.appendleft((cX, cY))
					begin_tracking = True
	
				# Draw bounding box on AruCo tag and center circle
				aruco.drawDetectedMarkers(frame, bbox)
				cv.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
	
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
			cv.circle(frame, (cX, cY), 25, cv_drawing_functions.RED, 1)
		
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
			if use_rpe:
				rir, rpe, velocity_stop = velocity_stops.calculate_rpe(rpe_df, avg_vel)

			# Save Data to DataFrame
			data_df.loc[len(data_df.index)] = [reps, (cX, cY), avg_vel, peak_vel, avg_vel_loss, peak_vel_loss, displacement, current_time]
			showStats(data_df.tail(1), rpe, rir)

# -------------------- Path Tracking --------------------

		# loop over deque for tracked position
		if show_path:
			for i in range(1, len(coords)):
			
				# Ignore drawing if curr/past position is None
				if coords[i - 1] is None or coords[i] is None:
					continue
				# Compute line between positions and draw
				cv.line(frame, coords[i - 1], coords[i], (0, 0, 255), 2)

# -------------------- Keyboard Interrupts --------------------
  
		# if the 'q' key is pressed, end tracking (and video)
		key = cv.waitKey(1)
		if key == ord("q"):
			break
		elif key == ord("p"):
			show_path = not show_path
		
# -------------------- Show Frame to screen --------------------

		# Save current gray frame as old (for Optical Flow)
		old_gray = gray_frame.copy()
  
		# Show video frame in new window (to resize properly)
		if not save_data:
			showInMovedWindow('Barbell Velocity Tracker:',frame, 200, 0)
		else:
			showInMovedWindow('Barbell Velocity Tracker:', frame, 200, 0, vid_writer=result)

# -------------------- Save Data and Close Windows --------------------
	# Release camera and destroy webcam video
	camera.release()
	result.release()
	cv.destroyAllWindows()


	if save_data:
     
		# Save this set as a Rep-to-Failure set.
		if save_rpe:
			filepath = save_folder + "/rpe_data.csv"
			# Generate RPE/RIR values into new dataframe
			rpe_df = velocity_stops.generate_rpe_table(data_df)
			rpe_df.to_csv(filepath, sep=',')
   
		# Save complete dataset.
		filepath = save_folder + "/velocity_data.csv"
		data_df.to_csv(filepath, sep=",")
 
	if plot_data:
		generate_plots(data_df, save_data, save_folder)


if __name__ == '__main__':
	main()
 