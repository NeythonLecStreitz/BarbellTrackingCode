from dash import Dash, dcc, html, Input, Output, State, ctx, Patch, MATCH, ALL
import dash_bootstrap_components as dbc
import dash_player as dp
import pandas as pd
import re
import os
import json

from datetime import datetime
import time

import upload as upload
import data

import sys        
sys.path.append("C:/Users/neyth/Desktop/SeniorComps/BarbellTrackingCode")       
import barbellVelocityTracker as rtb
import rep_gif

# Number of sets uploaded
num_sets = 0

# Save Path
UPLOAD_DIRECTORY = "C:\\Users\\neyth\\Desktop\\SeniorComps\\BarbellTrackingCode\\uploads"

# Set Dataframes
global data_df
data_df = None


# Get current date
DATE = datetime.now().strftime('%A %x')

# stylesheet with the .dbc class
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc_css])

def update_set_dropdown():
	print("Checking folder..")
	dropdown_items = []
	# Initiate dropdown menu with header
	dropdown_items.append(dbc.DropdownMenuItem("Session Selector", header=True))
 
	# Look in upload directory for each set
	i = 0
	for folder in os.listdir(UPLOAD_DIRECTORY):
		print(f"Folder: {folder}")
		folder_path = os.path.join(UPLOAD_DIRECTORY, folder)
		# Get json data for each set
		for file in os.listdir(folder_path):
			print(f"File: {file}")
			split_tup = os.path.splitext(file)
			if (split_tup[1] == ".json"):
				print("Found first JSON file")
				json_path = os.path.join(folder_path, file)
				# Use json data to name set in dropdown
				with open(json_path) as json_file:
					session_data = json.load(json_file)
					session_name = folder.replace("_", " ")
					session_display = f"{session_name} - {session_data['date']}"
					
					dropdown_items.append(dbc.DropdownMenuItem(session_display,
						   id={
								'type': 'dropdown-session',
								'index': i
							}
					))
				break
		i = i+1
	 
	# Add final dropdown selection for creating new set
	dropdown_items.append(dbc.DropdownMenuItem(divider=True))
	dropdown_items.append(dbc.DropdownMenuItem("Create New Session", n_clicks=0, id='open-set-modal'))
	return dropdown_items

header = html.H4(
	"Raise The Bar: Barbell Velocity and Bar Path Tracker", className="bg-primary text-white p-2 mb-2 text-center"
)

# Workout Settings
w_no_session = html.Div(children=[
					html.H4("Welcome to RTB."),
					html.P(f"{DATE}", className='text-center'),
					html.Br(),
					html.P("Please select or create a session to start tracking."),
					html.P("Not sure how to start? Press ABOUT for more info."),
					html.P(id='start-session', style={'display':'None'}),
	 				html.P(id='number-sets-uploaded', style={'display':'None'}),
					html.Div(id='force-close-modal', style={'display':'None'}),
					html.Div(id='sets-tracked', style={"display":"None"}),
	 				html.Div(id='set-accordion')
				],
				style={"display":"block", "left":"25%", "text-align":"center"}, id='w-no-session')

dropdown_items = update_set_dropdown()
w_dropdown = dbc.ButtonGroup(children=[
						dbc.DropdownMenu(children=dropdown_items, group=True, label='Session Selector', id='set-selector'),
						dbc.Button("About", id='open-backdrop', n_clicks=0)
	  				], id='btn-group')

# Set Settings
dropdown = html.Div(
				children=[
					html.Div(id='sets-dropdown-input'), 
					dcc.Dropdown(id='set-dropdown'), 
		   			html.Div(id='set-subdir'),
					html.Div(id='set-subdir-name')], 
				className="mb-4", id='sets-dropdown-div'
				)


set_volume = dbc.Row(
	[
		dbc.Col(
			[
				dbc.Label("Weight", style={"font-weight": "bold"}),
				dbc.Input(
					id="set-weight",
					type="number",
					min=0,
					max=1000,
					step=2.5,
					placeholder="Enter Set Weight",
				),
			],
			width=6,
		),
		dbc.Col(
			[
				dbc.Label("Repetitions", style={"font-weight": "bold"}),
				dbc.Input(
					id="set-reps",
					type="number",
					min=0,
					max=100,
					step=1,
					placeholder="Enter Set Repetitions",
				),
			],
			width=6,
		),
	],
	className="mb-4",
)

update_graphs = dbc.ButtonGroup([
					dbc.Button("Track Set", id='update-graph-button', n_clicks=0),
					dbc.Button("Download Data", id='download-data', n_clicks=0)
					], size="md",
				)

set_settings = [dropdown, set_volume, update_graphs]

# Welcome Modal -------------------------------------------
print_aruco_card = dbc.Card(
	[
		dbc.CardImg(src="/static/generated_aruCo_1x1-1.png", top=True, style={'height':'40%', 'width':'40%'}),
		dbc.CardBody(
			[
				html.H4("1. Print AruCo Tag", className="card-title"),
				html.P(
					"RTB works by detecting an AruCo tag on the barbell. "
					"AruCo tags are like simplified QR-codes. "
					"Print the tag with some white-space padding.",
					className="card-text",
				),
				dbc.Button("Print AruCo Tag", color="primary", id='btn-print-aruco'),
				dcc.Download(id='print-aruco')
			]
		),
	]
)
tape_aruco_card = dbc.Card(
	[
		dbc.CardImg(src="/static/taped_tag.PNG", top=True, style={'height':'30%', 'width':'30%'}),
		dbc.CardBody(
			[
				html.H4("2. Taping Tags", className="card-title"),
				html.P(
					"It is possible to simply tape the tag to the barbell cap. "
					"However, for reusability, we recommend taping to a piece of cardboard. "
					"Then, tape the cardboard to a paper roll, allowing for easy on-off.",
					className="card-text",
					)
			]
		),
	]
)
record_video_card = dbc.Card(
	[
		dbc.CardBody(
			[
				html.H4("3. Recording Sets", className="card-title"),
				html.P(
					"RTB works best when video is recorded at high-quality. "
					"Furthermore, having the camera at chest height and completely "
					"parallel is crucial to accurate tracking.",
					className="card-text",
				),
				dcc.Upload(dbc.Button("Upload Video", color="primary")),
			]
		),
	]
)

how_to_cards = dbc.Row(
	[
		dbc.Col(print_aruco_card, width=4, style={"height":100}),
		dbc.Col(tape_aruco_card, width=4, style={"height":100}),
		dbc.Col(record_video_card, width=4, style={"height":100}),
	]
)

welcome_modal = html.Div(
	[
		dbc.Modal(
			[
				dbc.ModalHeader([
					dbc.ModalTitle("Raise The Bar: Video-Based Barbell Velocity Tracking", className='text-center'),
		   			dbc.Button(
						"Close",
						id="close-backdrop",
						className="float-right",
						n_clicks=0,
					)], close_button=False),
				html.H2("Do you lift HARD enough?", className='text-center'),
				html.P(
					"The question of lifting intensity is a difficult one to answer. "
		   			"Top-end strength fluctuates daily based on a myriad of factors. ", className='text-center'),
				html.P(html.B("Using velocity, we can objectively measure lifting intensity and performance preparedness week-to-week."), className='text-center'),
				html.P(
					"RTB is a desktop tool for monitoring and displaying the range of motion"
		   			"and velocity statistics for a barbell during performance of the barbell Back squat.", className='text-center'),
				html.Hr(),
				html.H4("How it Works:", className='text-center'),
				dbc.ModalBody(how_to_cards),
			],
			fullscreen=True,
			id="modal-backdrop",
			is_open=False,
		),
	],
	style = {
		"display":"flex",
		"align-items":"center",
		"justify-content":"center"},
)
# --------------------------------------------------------
# New Session Modal
new_session_modal = html.Div(
	[
		dbc.Modal(
			[
				dbc.ModalHeader(dbc.ModalTitle("New Session"), close_button=True),
				dbc.ModalBody(
						[
							dbc.Label("Session Name"),
							dbc.Input(placeholder="Please name your workout...", value='', type="text", id='session-name'),
							html.Br(),
							dbc.Label("Number of Sets"),
							dbc.Input(placeholder="Can be updated later...", value=0, min=0,max=100,step=1, type="number", id='set-num'),
						]),
				dbc.ModalFooter(
					dbc.Button(
						"Next",
						id="open-upload-modal",
						className="ms-auto",
						n_clicks=0,
					)
				),
			],
			id="set-modal",
			centered=True,
			is_open=False,
		),
	]
)

upload_sets_modal = dbc.Modal(
	[
		dbc.ModalHeader(dbc.ModalTitle("Upload Set Videos"), close_button=False),
		dbc.ModalBody("Please upload at least one set:"),
		dcc.Upload(
			id='upload-set',
			children=html.Div([
				'Drag and Drop or ',
				html.A('Select Files')
			]),
			style={
				'width': '96%',
				'height': '60px',
				'lineHeight': '60px',
				'borderWidth': '1px',
				'borderStyle': 'dashed',
				'borderRadius': '5px',
				'textAlign': 'center',
				'margin': '10px'
			},
			# Allow multiple files to be uploaded
			multiple=True,
		),
		dbc.ModalFooter(
			dbc.Button(
				"Close",
				id="close-upload-modal",
				className="ms-auto",
				n_clicks=0,
			)
		),
		
	],
	id="upload-modal",
	is_open=False,
	className='text-center'
)


new_session = html.Div(
	[
		new_session_modal,
		upload_sets_modal,
	]
)

# Settings Tabs (Left side)
workout_control_tab = dbc.Tab(dbc.Card([w_no_session, w_dropdown, welcome_modal, new_session]), label="Menu")
set_control_tab = dbc.Tab(dbc.Card(
	set_settings,
	body=True,
), label="Workout", disabled=True, id='set-control-tab')
control_tabs = dbc.Card(dbc.Tabs([workout_control_tab, set_control_tab]))

# Data Tabs (Right side)
data_store = dcc.Store(id='statistics')

stats_tab = dbc.Tab(children=
					[
						html.Br(),
		 				html.P("Please select or start a new session to view statistics.", id='stats-tab-info'),
						
			 		], label="Stats", className='text-center', id='stats-tab')
video_tab = dbc.Tab(children=
					[
						html.Div(dbc.Button("View Untracked Video", id='track-toggle'), className="d-grid gap-2"),
						dbc.Col(html.Video(src="assets/tracked_output.mp4", controls=True, loop=True, width=480 ,height=854, id='tracked-vid', style={'display':'block'})),
						dbc.Col(html.Video(src="assets/untracked_output.mp4", controls=True, loop=True, width=480 ,height=854, id='untracked-vid', style={'display':'none'}))
					], label="Video", disabled=True, id='video-tab')
velocity_tab = dbc.Tab([dcc.Graph(id="velocity-hist")], label="Bar Velocity", disabled=True, id='velocity-tab')
bar_path_tab = dbc.Tab(children=
					   	[
							dbc.Row(
								[
									dbc.Col(dcc.Graph(id='scatter-chart'), width=9),
									dbc.Col(children=[
											html.H4("Bar Path Settings"),
											html.Br(),
											html.Label("Select Bar Path to Show"),
			 								dcc.RadioItems(options=[], value="All Reps", id='bar-path-selection'),
											html.Br(),
											dbc.Button("Show Rep Video", id='rep-gif'),
											html.Div(id='dummy-output'),
											html.Hr(),
											html.Div(children=[], id='bar-path-stats')
										], width=3)
								]
							)
		  				], label="Bar Path", disabled=True, id='bar-path-tab')
data_tabs = dbc.Card(dbc.Tabs([stats_tab, velocity_tab, bar_path_tab, video_tab, data_store]))


# Dashboard Layout
app.layout = dbc.Container(
	[
		header,
		dbc.Row(
			[
				dbc.Col([control_tabs],
					width=4,
				),
				dbc.Col([data_tabs], width=8),
			]
		),
	],
	fluid=True,
	className="dbc",
)

'''
@app.callback(
	Output("upload-set", "filename"), 
 	Output("upload-set", "contents"),
	Input({"type": "dropdown-session", "index": ALL}, "n_clicks"),
	prevent_initial_call=True,
)
def use_saved_set(ids):
	print("reason why buggin...")
	button_id = ctx.triggered_id
	
	return None, None
'''

@app.callback(
	Output("dummy-output", "style"),
	Input("rep-gif", "n_clicks"),
	State("bar-path-selection", "value"),
)
def show_rep_gif(n, rep):
	if n:
		rep_gif.generate_rep_loop("assets/tracked_output.mp4", data_df, rep)
		
	return {"display":"none"}

@app.callback(
	Output("untracked-vid", "style"),
	Output("tracked-vid", 'style'),
	Input("track-toggle", "n_clicks"),
	State('untracked-vid', 'style'),
	State('tracked-vid', 'style')
)
def change_video(n, untracked, tracked):
	if n:
		if untracked['display'] == 'none':
			return {"display":'block'}, {"display":"none"}
		else:
			return {'display':'none'}, {'display':"block"}
	else:
		return {'display':'none'}, {'display':"block"}
		

# Toggle Modal for uploading new videos
@app.callback(
	Output("modal-backdrop", "is_open"),
	[Input("open-backdrop", "n_clicks"), Input("close-backdrop", "n_clicks")],
	[State("modal-backdrop", "is_open")],
)
def toggle_about_modal(n1, n2, is_open):
	if n1 > 0 or n2 > 0:
		return not is_open
	if n1 == 0:
		return False
	return is_open

@app.callback(
	Output("set-modal", "is_open"),
	[
		Input("open-set-modal", "n_clicks"),
		Input("open-upload-modal", "n_clicks"),
	],
	[State("set-modal", "is_open")],
)
def toggle_session_modal(n0, n1, is_open):
	if n0 or n1:
		if n0 == 0:
			return False
		else:
			return not is_open
	return is_open


@app.callback(
	Output("upload-modal", "is_open"),
	[
		Input("open-upload-modal", "n_clicks"),
		Input("close-upload-modal", "n_clicks")
	],
	[State("upload-modal", "is_open")],
)
def toggle_upload_modal(n2,n1,is_open):
	if n2 or n1:
		return not is_open
	return is_open


# Download AruCo tag on press
@app.callback(
	Output('print-aruco', 'data'),
	Input('btn-print-aruco', 'n_clicks'),
	prevent_initial_call=True,
)
def download_aruco(n_clicks):
	return dcc.send_file('C:/Users/neyth/Desktop/SeniorComps/BarbellTrackingCode/generated_aruCo_1X1.pdf')
		

@app.callback(
	Output("bar-path-stats", "children"),
	Output("scatter-chart", "figure"),
	Input("bar-path-selection", "value"),
	prevent_initial_call = True
)
def update_bar_path_figure(selection):
	if not data_df.empty:
		fig = data.generate_set_bar_paths(data_df, reps_wanted=selection)
  
		temp_df = data_df.groupby("Rep").max()
		if selection == 'All Reps':
			stats = [
						html.Label([html.B("Set at a Glance")]),
						html.P(f"Avg Velocity: {data_df['Avg Velocity'].mean():.4f} m/s"),
						html.P(f"Avg Peak Velocity: {data_df['Peak Velocity'].mean():.4f} m/s"),
						html.P(f"Avg Displacement: {data_df['Displacement'].mean():.2f} mm"),
					]
		else:
			rep_int = int(selection[-1]) - 1
			stats = [
						html.Label([html.B("Rep at a Glance:")]),
						html.P(f"Avg Velocity: {temp_df.iloc[rep_int]['Avg Velocity']:.4f} m/s"),
						html.P(f"Peak Velocity: {temp_df.iloc[rep_int]['Peak Velocity']:.4f} m/s"),
						html.P(f"Displacement: {temp_df.iloc[rep_int]['Displacement']:.2f} mm")
					]
		return stats, fig
	else:
		return [], None


'''
HANDLE UPLOADING OF VIDEOS

'''

@app.callback(
	Output("set-accordion", "children"),
	Output("sets-dropdown-div", "children"),
	Input("w-no-session", "children"),
	Input("sets-tracked", "children"),
	State("sets-dropdown-input", "children"), 
 	State('set-subdir', "children"),
	prevent_initial_call=True)
def update_set_list(tracked, session, sets, subdir):
	options = []
	accordion_items = []
	i = 1
	if sets:
		for file in sets:
			filePath = os.path.join(UPLOAD_DIRECTORY, subdir, file)
			if "json" not in filePath:
				json_path = filePath.replace(".MOV", "_data.json")
				with open(json_path) as jpath:
					session_data = json.load(jpath)

				status = session_data['status']
				if status == 'tracked':
					set_weight = session_data['weight']
					set_reps = session_data['reps']
					options.append({"label":f"Set {i}		({status})		({set_weight}x{set_reps})", "value": filePath})
				else:
					options.append({"label":f"Set {i}		({status})", "value": filePath})
	 
				
				accordion_items.append(dbc.AccordionItem(
						[
							html.H6(file),
							html.P("Status: Untracked"),
							html.A("Download Video", href=filePath)
						], title=f"Set {i}: {file.split('.')[0]}"
					))
				i = i+1
				
   
	accordion = [dbc.Accordion(accordion_items, start_collapsed=True)]
	return accordion, [
		html.H4(id='set-subdir-name'),
		html.Div(id='sets-dropdown-input', style={'display':'none'}),
		dbc.Label("Select a Set"),
		dcc.Dropdown(placeholder='Sets', options=options, id='set-dropdown'),
  		html.Div(id='set-subdir', style={"display":"none"})] 

@app.callback(
	Output("stats-tab-info", "children"),
	Output("set-selector", "children"),
	Output("set-control-tab", "disabled"),
	Output("stats-tab", "disabled"),
	Output("video-tab", "disabled"),
	Output("velocity-tab", "disabled"),
	Output("bar-path-tab", "disabled"),
	Output("w-no-session", "children"),
	Input("close-upload-modal", "n_clicks"),
	State('start-session', 'children'), 
 	State("session-name", "value"), 
  	State("number-sets-uploaded", "children"),
 	prevent_initial_call=True,
)
def activate_session(modal, start_session, session_name, sets_uploaded):
	print('Activating session')
	new_sets = update_set_dropdown()
	tracked = 0
	stats_tab = "Select a set from the Workout Tab to track/view statistics."
	return stats_tab, new_sets, False,False,False,False,False, [
		html.H2(f"{session_name}"),
		html.H6(DATE),
		html.Hr(),
		html.P(sets_uploaded),
		html.P(f"{tracked} sets tracked.", id='sets-tracked'),
		html.Div(children=[], id='set-accordion'),
		html.Br(),
		html.Div(id='start-session', style={'display':'None'}),
		html.Div(id="number-sets-uploaded", style = {'display':"None"}),
		html.Div(id="sets-tracked", style = {'display':"None"})
	]
@app.callback(
	Output("set-subdir-name", "children"),
	Output("set-subdir", "children"),
	Output('sets-dropdown-input', 'children'),
	Output("start-session", "children"),
	Output("number-sets-uploaded", "children"),
	Input("upload-set", "filename"), 
 	Input("upload-set", "contents"),
	State('session-name', 'value'), 
 	State("set-num", "value"),
)
def update_output(uploaded_filenames, uploaded_file_contents, subdir_raw, set_num,):
	subdir = None
	print("Update Output buggin!")
	"""Save uploaded files and regenerate the file list."""
	if uploaded_filenames is not None and uploaded_file_contents is not None:
		subdir = re.sub(r'\W+', '_', subdir_raw)
		for name, data in zip(uploaded_filenames, uploaded_file_contents):
			upload.save_file(subdir, name, data, set_num)

	if subdir:
		files = upload.uploaded_files(subdir)
	else:
		files = []
	
	sets_uploaded = f"{int(len(files) / 2)} / {set_num} sets uploaded."
	if len(files) < 2:
		return subdir_raw, subdir, None, False, 0
	else:
		return  subdir_raw, subdir, files, True, sets_uploaded


# Update Data
@app.callback(
	Output("sets-tracked", "children"),
	Output("scatter-chart", "figure", allow_duplicate=True),
	Output("velocity-hist", "figure"),
	Output("statistics", 'data'),
	Input("update-graph-button", "n_clicks"),
 	State("set-dropdown", "value"), 
  	State("set-weight", "value"), 
   	State("set-reps", "value"),
 	prevent_initial_call=True)
def update_data(n_clicks, file_path, set_weight, set_reps):
	if ctx.triggered_id == "update-graph-button" and n_clicks > 0:
		# Run tracking...
		velocity_df, coord_df = rtb.main(file_path, set_weight=set_weight)
		velocity_df.to_csv('vel.csv', sep=",")
		coord_df.to_csv('coords.csv', sep=",")
		# Prepare data
		global data_df
		data_df = data.prepare_data(coord_df, velocity_df)
		# Get Statistics
		fig_stats = data.prepare_statistics(data_df)
		# Get velocity data
		fig_vel = data.generate_velocity_histogram(velocity_df)
		# Get bar path data
		fig_bar = data.generate_set_bar_paths(data_df, 'All Reps')
  
		# Update tracked status
		json_file_path = file_path.replace(".MOV", '_data.json')
		with open(json_file_path, 'r') as open_json:
			session_data = json.load(open_json)
   
		with open(json_file_path, 'w', encoding='utf-8') as f:
			data_json = {
					"name": session_data['name'], 
					"date": DATE,
					"file_path": session_data['file_path'],
					"sets": session_data['sets'],
					"status": "tracked",
	 				"weight": set_weight,
		 			"reps": set_reps}
			json.dump(data_json, f, ensure_ascii=False, indent=4)

		return 1, fig_bar, fig_vel, fig_stats, 
	return None, None, None


@app.callback(
	Output("bar-path-selection", "options"),
	Output("stats-tab", "children"),
	Input('statistics', 'data'),
	prevent_initial_call=True
)
def update_stats_tab(stats_dict):
	if stats_dict:
		rep_list = ["All Reps"]
		for rep in range(1, stats_dict['Total Reps'] + 1):
			rep_list.append(f"Rep {rep}")

		return rep_list, [
			html.Br(), 
			html.H2("Set Statistics"), 
			html.Hr(), 
			html.P("Total Reps: {}".format(stats_dict['Total Reps'])),
			html.P("Average Rep Velocity: {:.2f} m/s".format(stats_dict["Average Rep Velocity"])),
			html.P("Average Peak Velocity: {:.2f} m/s".format(stats_dict['Average Peak Velocity'])),
			html.P("Max Velocity Loss: {:2f} %".format(stats_dict['Max Velocity Loss'])),
			html.Br(),
			dbc.Label("Set Notes:"),
			dbc.Textarea(className="mb-3", placeholder="Type here..."),
   		]
	return [], [html.Br(), html.P("Select a set from the Workout Tab.")]

if __name__ == "__main__":
	app.run(debug=True, use_reloader=False)