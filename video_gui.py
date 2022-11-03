import PySimpleGUI as sg
import os
import barbellVelocityTracker
import generate_aruCo


filename = ''
use_rpe = False
fnames = []

# --------------------------------- The GUI ---------------------------------

# First the window layout...2 columns
version = '25 October 2022'

folder_col = [[sg.Text('Select Video'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
              [sg.Listbox(values=[], enable_events=True, size=(45,10),key='-FILE LIST-')],
              [sg.Button('Track Barbell', key='-START-'), sg.Button('Generate AruCo', key='-ARUCO-'), sg.Button('Exit')]]

options_col = [[sg.Text('Options', font=('Arial', 15, 'bold'))],
               [sg.CBox('Use Rep-to-Failure Data?',key='-USE RPE-')],
               [sg.CBox('Save Video and Data to Folder?',key='-SAVE DATA-')],
               [sg.CBox('Generate Velocity Plots?',key='-PLOT DATA-')],
               [sg.CBox('Save as Rep-to-Failure Set?',key='-SAVE RPE-')],
               [sg.Text('', font=('Arial', 10), text_color='red', key='-ERROR-')]]
# ----- Full layout -----
layout = [[sg.Column(folder_col), sg.VSeperator(), sg.Column(options_col)]]

# ----- Make the window -----
window = sg.Window('Barbell Velocity and Bar Path Tracker', layout, grab_anywhere=True)

while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break
    if event == '-FOLDER-':         # Folder name was filled in, make a list of files in the folder
        folder = values['-FOLDER-']
        img_types = (".mp4",".mov", ".mkv",".csv", ".txt", ".py")
        # get list of files in folder
        try:
            flist0 = os.listdir(folder)
        except:
            continue
        fnames = [f for f in flist0 if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith(img_types)]
        window['-FILE LIST-'].update(fnames)
    elif event == '-FILE LIST-':    # A file was chosen from the listbox
        try:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
        except:
            continue
    elif event =='-ARUCO-':
        # Generate AruCo tag from 6x6_50 Dictionary.
        generate_aruCo.generate_markers(marker_size=6, total_markers=50, grid_size=(1, 1))
    elif event == '-START-':
        begin_tracking = True
        if values['-USE RPE-']:
            if 'rpe_data.csv' in fnames:
                use_rpe = True
            else:
                window['-ERROR-'].update("RPE dataset not found in folder.\nMake sure its named 'rpe_data.csv'")
                begin_tracking=False
        
        if begin_tracking:        
            if filename != '':
                window.hide()
                barbellVelocityTracker.main(video_path=filename, use_rpe=use_rpe, save_data=values['-SAVE DATA-'], plot_data=values['-PLOT DATA-'], save_rpe=values['-SAVE RPE-'])
                window.UnHide()
            else:
                window['-ERROR-'].update("No video file selected...")
            

window.close()