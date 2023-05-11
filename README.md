# Raise The Bar: Video-Based Barbell Path and Velocity Tracker

<p align="center">
Raise The Bar is a desktop tool for monitoring and displaying the range of motion and velocity statistics for a barbell during performance of the barbell Back squat. 
Statistics on average concentric velocity, peak velocity, displacement, and bar path are outputted during and after a set.
Uses AruCo tags and the OpenCV python library for optically tracking the barbell through a pre-recorded or webcam-streamed video. 
Also uses a combination of PySimpleGUI, Dash, and Plotly for app interface and statistics plotting.
Developed for my CS Senior comprehensive project.
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/70167258/207141756-217b32fb-82f6-44b5-a3af-33199319f12b.png" />
</p>

<p align="center">
<a href="#Motivation">Motivation</a> &nbsp;&bull;&nbsp;
<a href="#Overview">Overview</a> &nbsp;&bull;&nbsp;
<a href="#Usage">Usage</a> &nbsp;&bull;&nbsp;
<a href="#Tips">Tips</a> &nbsp;&bull;&nbsp;
<a href="#Future">Future</a> &nbsp;&nbsp; 
</p>

# Motivation
Do you lift hard enough? 
The question of lifting intensity is a difficult one to measure. 
In general, lifting intensity is measured as a percentage of your 1-Repetition Max (1RM). 
Thus, if the most you can squat is 300 lbs, your 80% is 240 lbs. 
Yet, studies have shown wide daily fluctations in 1RM meaning this method is problematic.
As a team sport athlete, a competition over the weekend might negatively impact your performance on the following Monday.
Using velocity, we can objectively measure lifting intensity and performance preparedness week-to-week.
By setting velocity standards, we can auto-regulate the total weight lifted to better align with your preparedness.

While Velocity Based Training (VBT) is a great option for auto-regulating training and objectifying intensity, current methods of implementation are inaccesibile to most. 
The "gold standard" of velocity tracking, Linear Position Transducers, cost at the very least $300 per device, not to mention the need for a phone or tablet to sync to the device, and/or any monthly app fees. 
On the opposite end, software based apps for velocity tracking are typically free, but either have quite poor accuracy or require extensive manual operation.

Therefore, the purpose of this project is to show a proof of concept for creating an accurate and accesibile video-based velocity and bar path tracker.

# Overview
Velocity tracking works in three main steps: tag identification/localization, velocity calculating, and user interface.
* The AruCo tag identification process happens via the OpenCV python library tools for AruCo tag detection.
    The frame is first grayscaled before the candidate markers are retrieved.

* Velocity calculation happens by taking the tag displacement times the video FPS over 1000. 
    This gives the mm/s velocity of the barbell.
    To convert pixel displacement to mm displacement, we calculate the pixel perimeter of the AruCo tag and divide by the real perimeter of the tag.
    Velocity calculations are stored and then averaged upon completion of a repetition.

* User interface was kept simple to prioritize velocity tracking features.
    To start the application, PySimpleGUI was used to build out a file browser to select the video to track.
    After completion of the barbell tracking, Dash and Plotly were used to show the bar path of each repetition as well as a bar chart of the rep velocities across the entire set. 
    OpenCV is also used briefly to show a looped video of each individual repetition.

### Algorithm Overview
The overall algorithm proceeds as follows:
1. Retrieve next frame and identify AruCo tag in frame.
2. If no AruCo tag found (likely due to motion blur), use Optical Flow estimation to localize frame based on previous frame.
3. Calculate tag displacement between past and current frame.
4. Convert displacement from pixels to mm and find velocity in mm/s. 
5. If two inflections have been detected (one at the bottom and one at the top), count full repetition and caculate aggregate velocity statistics across the entire rep.
6. Continue looping until end of video.

### File Overview

**barbellVelocityTracker:** responsbile for driving the barbell tracking, tag identification, bar path functions, and velocity calculations.

**output_plots:** responsible for outputting barbell path graphs and velocity bar charts in a Dash app.

**rep_gif:** responsible for creating looped videos of individual repetitions for the output_plots file.

**cv_drawing_functions:** helper functions for drawing text on frames and holding CV color options.

**generate_aruco:** generates a 6x6_50 aruCo tag to be printed by the user. Is started by the video_gui file if the user selects the button.



# Usage
### Dependencies
Use the package manager pip to install all required libraries.
```bash
pip install -r requirements.txt
```

### Starting the App
To begin tracking, run the app.py file in the dashboard folder.
```bash
python .\dashboard\app.py
```

A dashboard will appear with several options. 

![image](https://github.com/NeythonLecStreitz/BarbellTrackingCode/assets/70167258/17c2e55e-dfcb-4061-b219-90db7fdaaf7e)


To begin, create a new session.
You will be asked to name the session and indicate how many videos you will be uploading.

### Printing a Tag
To print an AruCo tag, press the **About** button and select **PRINT**.
The default size should work but you may try a slightly smaller tag.
Place the tag on some sturdy back (cardboard works) and tape to the end of the barbell to track.
Remember that when cutting the tag out, the tag needs some white space to highlight the black border.
For ease-of-use, try taping a small cardboard paper tube to the end of the tag to easily slip the tag on-and-off the barbell.


### Tracking a Video
Once you have created a session and uploaded the videos you want to track, navigate to the **WORKOUT** tab.
A sample video is included in this repository, ```sample_squat.mov```.

In this tab, select the specific video you'd like to track, and indicate the weight and number of reps for the set.
Press the **TRACK** button and wait until a video beings playing.

<p align="center">
  <img src="https://github.com/NeythonLecStreitz/BarbellTrackingCode/assets/70167258/8067c7b4-806d-489b-b6d6-94fed6c1b794" />
</p>

In the future, weight input will also be used for power output.
After a few seconds, the video will appear and begin to play.
You should notice a red contrail begin to follow the tag.

**IMPORTANT:**

To facilitate the output plots after conclusion of the set, at the start of the first repetition (as you are about to go down), press **s** on your keyboard to intitiate tracking. This helps differentiate the walk out portion from individual repetitions. At the conclusion of your last repetition (as you hit the top), press **s** once more on your keyboard to stop the tracking.
If you do not need the output plots and just want to track in real-time, the algorithm will still work automatically (so you do not need to press **s**).
The text at the top of the frame should be blue if you have NOT toggled tracking, and magenta if tracking IS toggled.


### Viewing Output Plots

Upon conclusion of the set, if you selected to save the data, two new files should have been created: cooord_data and velocity_data.
coord_data holds information on specific coordinates of the bar throughout the entire set.
velocity_data simply holds information on repetition average velocities and the frame number of the end of each repetition.
To download this data, press **DOWNLOAD DATA**.

The **BAR VELOCITY** tab includes velocity statistics as a histogram across the entire set. In a quality set, you should see velocity decreasing as the repetitions increase.

The **BAR PATH** tab includes the bar path for each repetition including the entire set. Use the dropdown menu to the right to select specific repetitions. Additionally, you can press the **SHOW REP** button to view a looped video of that specific repetition.
To note is that the bar path plots might look slightly skewed because they are in 2D while the actual bar path in the video is likely at a slight angle.

The **VIDEO** tab includes the ability to watch an untracked and tracked version of the set. 

Velocity Tracking          |  Bar Path                 | Video
:-------------------------:|:-------------------------:|:-------------------------:
![image](https://github.com/NeythonLecStreitz/BarbellTrackingCode/assets/70167258/9c57a724-fb13-42ba-808c-8c3848ae8e9e) |  ![image](https://github.com/NeythonLecStreitz/BarbellTrackingCode/assets/70167258/388dbac5-e7ae-4a4f-805f-8647c0ded1b3) | ![image](https://github.com/NeythonLecStreitz/BarbellTrackingCode/assets/70167258/3ab9c465-d9fa-4384-b4b2-a1a4548aa1cf)
 

# Tips
* Tape the AruCo tag to a piece of sturdy cardboard at a medium size. Make sure to keep some white space on the paper.
* Record in 60 FPS, side-on as much as possible. 
* Set your phone down as having someone record may cause the phone to sway and incorrectly detect movement of the bar.
* Perform your set as strictly as possible with very little commotion during walk-out/racking phases.
* Do not add extra bounces, pauses, swaying side-to-side, or otherwise movement that might confuse detection of the repetition.

# Future
* Improving velocity tracking accuracy to function even at higher velocities.
* Updating the GUI to be more beginner-friendly with information on VBT practices.
* Building a more robust AruCo tag "device" to fit on the barbell.
* Adding information on power output.
* Highlighting specific areas of lower speed within a single repetition to uncover sticking/weak points.
* Adding support for other barbell-based exercises (bench press, deadlift, ...).
