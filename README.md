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
**video_gui:** driver for all other files, asks user for video file, save folder, tracking settings, and intitiates the barbellVelocityTracker file.

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
To begin tracking, run the video_gui.py file.
```bash
python .\video_gui.py
```

A simple GUI will appear with several options. 

![image](https://user-images.githubusercontent.com/70167258/207139464-c85499aa-ade1-4401-ba67-9f53020fc195.png)

In general it is best to keep all options at default.
This includes saving the set data to .csv files and showing the Dash plots after conclusion of the tracking.

### Printing a Tag
To print an AruCo tag, press the **Generate AruCo** button and print out the tag.
The default size should work but you may try a slightly smaller tag.
Place the tag on some sturdy back (cardboard works) and tape to the end of the barbell to track.
Remember that when cutting the tag out, the tag needs some white space to highlight the black border.
For ease-of-use, try taping a small cardboard paper tube to the end of the tag to easily slip the tag on-and-off the barbell.


### Tracking a Video
In the top right corner of the GUI, select **browse** and find the folder with the video to track.
Select the video and it will be highlighted black.
A sample video is included in this repository, ```sample_squat.mov```.

When ready, press **Track barbell** to start.
You will be prompted to enter a folder to save the data into and to input the weight of the barbell (to help name the resulting data).

![image](https://user-images.githubusercontent.com/70167258/207139575-65c9028f-9347-47cf-b73b-a6390e836ca5.png)

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

If you selected to view output plots, in whatever IDE you use, a Dash app should begin running. On my machine it looks as such:
```
Dash is running on http://127.0.0.1:8050/

 * Serving Flask app 'output_plots'
 * Debug mode: on
```

Follow the address and all output plots including bar path for every rep and a bar chart of the entire set should appear.
Pressing the **Show rep loop** button will begin playing a looped version of either the walk out or an individual repetition.
Unfortunately, this is without a contrail showing the bar path.
To note is that the bar path plots might look slightly skewed because they are in 2D while the actual bar path in the video is likely at a slight angle.




Velocity Tracking          |  Bar Path
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/70167258/207140929-67e10329-db3c-4d5d-8a20-3ef873dd3263.png)  |  ![image](https://user-images.githubusercontent.com/70167258/207141248-f96d0a5b-8cbe-4cf1-bdc8-cd152973a3d5.png)

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
