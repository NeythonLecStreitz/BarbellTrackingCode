from dash import Dash, dcc, html, Input, Output, ctx
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import rep_gif

def generate_allrep_velocity(data_df):
	'''
	Creates a bar chart showing average and peak velocity of each repetition over the entire set.
 	'''
    
    
	data_df = data_df[data_df['Rep'] > 0]
	data_df['Rep'] = pd.to_numeric(data_df["Rep"])
	data_df['Avg Velocity'] = pd.to_numeric(data_df['Avg Velocity'])
	data_df['Peak Velocity'] = pd.to_numeric(data_df['Peak Velocity'])
	set_avg = float(data_df['Avg Velocity'].mean())
	
	fig = px.bar(data_df, x='Rep', y=['Avg Velocity', 'Peak Velocity'], title='Set Velocity Statistics',text_auto='.3f', barmode='overlay')
	fig.add_hline(y=set_avg, line_width=3, line_dash="dash", line_color="salmon",  
				annotation_text=f"Set Avg: {set_avg:.3f}", 
				annotation_position="bottom right")
	fig.update_xaxes(title_text="Rep", dtick=1)
	fig.update_yaxes(title_text="Velocity (m/s)")
	fig.update_layout(legend_title_text='Velocity Type')
 
	return fig

def create_dash_env(data_df, coord_df, video_path, set_weight):
	'''
	Creates a Dash environment for displaying the different output plots for bar path and rep velocity statistics.
	'''
    
	# Preprocess velocity and coordinate data by merging the two on Rep #.
	coord_df['Reps'] = coord_df['Reps'] + 1
	coord_df = pd.merge(coord_df, data_df[['Avg Velocity', 'Peak Velocity', 'Avg Velocity Loss', 'Peak Velocity Loss', 'Rep']], left_on='Reps', right_on='Rep')
	name_list = [f'Rep: {x}' for x in coord_df['Reps']]
 
	# Retrieve the starting coordinates of each repetition.
	first_coords = coord_df.groupby('Reps').first()
	rep_list = ["All Reps"] + ["{}".format(i + 1) for i in range(int(coord_df['Reps'].max()))]
 
 
	# Build the app layout including dropdowns and buttons.
	app = Dash(__name__)

	app.layout = html.Div([
		html.H2('Set Velocity and Bar Path Statistics'),
		html.B("Please select an option:"),
		html.P("Bar Path: Shows the relative coordinate bar path of all reps."),
		html.P("Velocity Data: Shows a bar chart of average and peak velocity of all reps over the entire set."),
		dcc.RadioItems(
			id="radio",
			options=['Bar Path', 'Velocity Data'],
			value="Bar Path"),
		html.Div([dcc.Dropdown(
					id="rep_type",
					options=rep_list,
					value="All Reps",
					clearable=False,),],
           style={'display': 'block'}, id='dropdown'),
		html.Button("Show Walk Out GIF", id='show_loop', style = dict(display='block', background_color='blue')),
		dcc.Graph(id="graph")

	])

	@app.callback(
		Output("dropdown", "style"),
		Output("graph", "figure"),
		Output("show_loop", 'style'),
		Output("show_loop", 'children'),
		Input("rep_type", "value"),
  		Input('radio', 'value'),
		Input('show_loop', 'n_clicks'))
	def update_bar_chart(rep_type, plot_type, n_clicks):
		# Update the Dash app based on user input.
		df = coord_df
  
		# If bar path, show line plot as graph.
		if plot_type == 'Bar Path':
			style = {'display': 'block'}
			if rep_type == 'All Reps':
				style2 = {'display': 'block'}
				title = 'Show Walk Out GIF'
				# Show looped repetition video if user pressed button.
				if ctx.triggered_id == 'show_loop':
					rep_gif.generate_rep_loop(video_path, data_df, 0)
       
				df["Reps"] = df["Reps"].astype(str)
				min_x = df['cX'].min() - 150
				max_x = df['cX'].max() + 150

				# Show each repetition as a separate line on the same plot.
				fig = px.line(df, x="cX", y="cY", title=f'All Reps Bar Path', color='Reps', hover_name=name_list, 
					hover_data={'Reps': None, 'cX': None, 'cY': None, 'Avg Velocity':':.3f', 'Peak Velocity': ':.3f'})
		
				# Annotate the starting coordinate of each repetition with its corresponding rep number
				for index, row in first_coords.iterrows():
					x_coord = row['cX'].astype(int)
					y_coord = row['cY'].astype(int)
					rep = row['Rep'].astype(int)
					fig.add_annotation(x=x_coord, y=y_coord, text=str(rep), showarrow=False, 
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))

				fig.update_layout(legend=dict(
					orientation="h",
					entrywidth=50,
					yanchor="bottom",
					y=1.02,
					xanchor="right",
					x=1,
					font=dict(
						family="Gravitas One",
						size=20,
						color="black"
					),
					bordercolor="Black",
					borderwidth=2
				))
		
		
				# Add annotations regarding aggregate velocity statistics for the entire set.
				# Includes total reps, velocity average, peak velocity, max velocity loss.
				text_x = max_x - 75
				text_y = df['cY'].median() + 200
				fig.add_annotation(x=text_x, y=text_y, text=f"Total reps: {data_df['Rep'].max()}", showarrow=False, 
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
				fig.add_annotation(x=text_x, y=text_y - 50, text="Velocity Average: {:.3f} m/s".format(float(data_df['Avg Velocity'].mean())), showarrow=False, 
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
				fig.add_annotation(x=text_x, y=text_y - 100, text="Peak Velocity: {:.3f} m/s".format(float(data_df['Peak Velocity'].max())), showarrow=False,
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
				fig.add_annotation(x=text_x, y=text_y - 150, text="Max Velocity Loss: {:.3f}%".format(float(data_df['Avg Velocity'].max() - data_df['Avg Velocity'].min())), showarrow=False, 
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
				fig.add_annotation(x=text_x, y=text_y - 400, text="Rep # indicates start point", showarrow=False, 
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))

				fig.update_xaxes(range = [min_x, max_x], showticklabels=False)
				fig.update_yaxes(showticklabels=False)
				fig.update_layout(legend_title_text='Rep')
			# Change plot to only show an individual repetition instead of all.
			else:
				title = "Show Rep GIF"
				style2 = dict(display='block')
    

				mask = df["Reps"] == rep_type
				rep_df = df[mask].reset_index()
				min_x = df['cX'].min() - 100
				max_x = df['cX'].max() + 200
				cutoff_index = rep_df['cY'].astype(int).idxmin()
				concentric = rep_df.iloc[cutoff_index:]
    
				# Draw the entire bar path onto the plot.
				fig = px.line(rep_df, x='cX', y='cY',
                  			hover_data={'Reps': None, 'cX': None, 'cY': None, 'Avg Velocity':':.3f', 'Peak Velocity': ':.3f'})
    
				# Draw only the concentric portion of the plot over the existing full bar path.
				# This creates two different colors for the eccentric and concentric portions, but maintains only the hover information from the original line.
				fig.add_trace(
					go.Scatter(
						x=concentric['cX'],
						y=concentric['cY'],
						mode="lines",
						line=go.scatter.Line(color="red"),
						showlegend=False,
      					hoverinfo='skip'
           			)
				)
				fig.update_layout(title=f'Rep {rep_type} Bar Path', xaxis_title="", yaxis_title="")
				fig.add_annotation(x=df[mask]['cX'].iloc[0], y=df[mask]['cY'].iloc[0], text=rep_type, showarrow=False,
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
				fig.update_xaxes(range = [min_x, max_x], showticklabels=False)
				fig.update_yaxes(showticklabels=False)


				# Add annotations regarding aggregate velocity statistics for the entire set.
				# Includes total reps, velocity average, peak velocity, max velocity loss.
				text_x = max_x - 75
				text_y = df['cY'].median() + 200
				fig.add_annotation(x=text_x, y=text_y, text=f"Rep: {rep_type}", showarrow=False, 
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
				fig.add_annotation(x=text_x, y=text_y - 50, text="Velocity Average: {:.3f} m/s".format(float(df[mask]['Avg Velocity'].iloc[0])), showarrow=False, 
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
				fig.add_annotation(x=text_x, y=text_y - 100, text="Peak Velocity: {:.3f} m/s".format(float(df[mask]['Peak Velocity'].iloc[0])), showarrow=False,
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
				fig.add_annotation(x=text_x, y=text_y - 300, text='Blue Line: Eccentric Phase', showarrow=False,
                       	font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
				fig.add_annotation(x=text_x, y=text_y - 350, text='Red Line: Concentric Phase', showarrow=False,
                       	font=dict(family="Gravitas One, monospace",size=20,color="#000000"))

				if rep_type != '1':
					fig.add_annotation(x=text_x, y=text_y - 150, text="Avg Velocity Loss: {:.3f}%".format(float(df[mask]['Avg Velocity Loss'].iloc[0])), showarrow=False,
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))
					fig.add_annotation(x=text_x, y=text_y - 200, text="Peak Velocity Loss: {:.3f}%".format(float(df[mask]['Peak Velocity Loss'].iloc[0])), showarrow=False,
						font=dict(family="Gravitas One, monospace",size=20,color="#000000"))

				# Display rep loop if user pressed button.
				if ctx.triggered_id == 'show_loop':
						rep_gif.generate_rep_loop(video_path, data_df, int(rep_type))

		# If users chooses velocity data, display bar chart instead of barbell path line plot.
		elif plot_type == 'Velocity Data':
			title = ''
			style = {'display': 'none'}
			style2 = {'display': 'none'}
			fig = generate_allrep_velocity(data_df)
			# fig = generate_allrep_power(data_df, set_weight)
		
		return style, fig, style2, title

	app.run_server(debug=True, use_reloader=False)
 