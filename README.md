# Raise The Bar: Video-Based Barbell Path and Velocity Tracker

<p align="center">
Raise The Bar is a desktop tool for monitoring and displaying the range of motion and velocity statistics for a barbell during performance of the barbell Back squat. 
Statistics on average concentric velocity, peak velocity, displacement, and bar path are outputted during and after a set.
Uses AruCo tags and the OpenCV python library for optically tracking the barbell through a pre-recorded or webcam-streamed video. 
Also uses a combination of PySimpleGUI, Dash, and Plotly for app interface and statistics plotting.
Developed for my CS Senior comprehensive project.
</p>

<p align="center">
<a href="#Motivation">Motivation</a> &nbsp;&bull;&nbsp;
<a href="#Overview">Overview</a> &nbsp;&bull;&nbsp;
<a href="#Usage">Usage</a> &nbsp;&bull;&nbsp;
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
The overall algorithm proceeds as follows:
1. Retrieve next frame and identify AruCo tag in frame.
2. If no AruCo tag found (likely due to motion blur), use Optical Flow estimation to localize frame based on previous frame.
3. Calculate tag displacement between past and current frame.
4. Convert displacement from pixels to mm and find velocity in mm/s. 
5. If two inflections have been detected (one at the bottom and one at the top), count full repetition and caculate aggregate velocity statistics across the entire rep.
6. Continue looping until end of video.

The AruCo tag identification process happens via the OpenCV python library tools for AruCo tag detection.
The frame is first grayscaled before the candidate markers are retrieved.

Velocity calculation happens by taking the tag displacement times the video FPS over 1000. 
This gives the mm/s velocity of the barbell.
To convert pixel displacement to mm displacement, we calculate the pixel perimeter of the AruCo tag and divide by the real perimeter of the tag.
Velocity calculations are stored and then averaged upon completion of a repetition.

User interface was kept simple to prioritize velocity tracking features.
To start the application, PySimpleGUI was used to build out a file browser to select the video to track.
After completion of the barbell tracking, Dash and Plotly were used to show the bar path of each repetition as well as a bar chart of the rep velocities across the entire set. 
OpenCV is also used briefly to show a looped video of each individual repetition.


# Usage



#
