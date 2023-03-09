import PySimpleGUI as sg
import os
import barbellVelocityTracker
import generate_aruCo
import output_plots
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def start_gui():
    '''
    Driver for the entire app.
    Responsible for creating the initial GUI for selecting video files and where to save calculated data.
    Also starts the barbell tracking once the user has selected the settings they want.
    '''
    
    
    filename = ''
    fnames = []
    data_df, temp_df = 0, 0
    # --------------------------------- The GUI ---------------------------------
    folder_col = [[sg.Text('Select Video'), sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
                [sg.Listbox(values=[], enable_events=True, size=(45,10),key='-FILE LIST-')],
                [sg.Button('Track Barbell', key='-START-'), sg.Button('Generate AruCo', key='-ARUCO-'), sg.Button('Exit')]]

    options_col = [[sg.Text('Barbell Velocity and Bar Path Tracker', font=('Arial', 12, 'bold'))],
                [sg.Text("Please select the video to analyze by pressing 'browse'.")],
                [sg.Text("To print an AruCo tag, press 'Generate AruCo'.")],
                [sg.Text('Select save options below.')],
                [sg.Text('\nOptions', font=('Arial', 10, 'bold'))],
                [sg.CBox('Save Set Data to Folder?',key='-SAVE DATA-', default=True)],
                [sg.CBox('Show advanced plots?', key='-PLOT-', default=True)],
                [sg.Text('', font=('Arial', 10), text_color='red', key='-ERROR-')]]
    # ----- Full layout -----
    layout = [[sg.Column(options_col), sg.VSeperator(), sg.Column(folder_col)]]

    # ----- Make the window -----
    window = sg.Window('Barbell Velocity and Bar Path Tracker', layout, grab_anywhere=True)

    while True:
        event, values = window.read()
        if event in (None, 'Exit', sg.WIN_CLOSED):
            break
        elif event == '-FOLDER-':         # Folder name was filled in, make a list of files in the folder
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
        elif event =='-ARUCO-':         # AruCo tag generation was chosen
            # Generate AruCo tag from 6x6_50 Dictionary.
            generate_aruCo.generate_markers(marker_size=6, total_markers=50, grid_size=(1, 1))
        elif event == '-START-':        # Start Tracking was chosen
            if filename != '':
                window.hide()
                layout = [
                        [sg.Text('Where would you like to save your data?', font=('Arial', 10))],
                        [sg.In(size=(25,1), enable_events=True , key='-SAVE FOLDER-'), sg.FolderBrowse()],
                        [sg.Text('Please enter set weight:', font=('Arial', 10))],
                        [sg.Text('Weight (lbs)'), sg.In(size=(10,1), enable_events=True , key='-WEIGHT-')],
                        [sg.Button('Begin Tracking', key='BEGIN')]]
                save_window = sg.Window('', layout)

                while True:
                    save_event, save_values = save_window.Read()
                    if save_event is None or save_event == 'Exit':
                        break
                    elif save_event == 'BEGIN':
                        save_window.close()
                        barbellVelocityTracker.main(video_path=filename, set_weight=save_values['-WEIGHT-'], save_data=values['-SAVE DATA-'], save_folder=save_values['-SAVE FOLDER-'])
                        window.UnHide()
                        break
                    elif save_event == sg.WIN_CLOSED:
                        break
                    
                save_window.close()
            else:
                window['-ERROR-'].update("No video file selected...")
             
    window.close()
    
    return None
    
if __name__ == '__main__':
    temp = start_gui()