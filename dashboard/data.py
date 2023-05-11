import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def prepare_data(coord_df, velocity_df):
 
	# Merge Velocity and Coordinate data on Rep
	coord_df.rename(columns={'Reps':'Rep'}, inplace=True)
	coord_df['Rep'] = coord_df['Rep'] + 1
	data_df = pd.merge(coord_df, velocity_df[['Avg Velocity', 'Peak Velocity', 'Avg Velocity Loss', 'Peak Velocity Loss', 'Displacement', 'Time', 'Rep']], on='Rep')
	data_df[["cX", "cY"]] = data_df[["cX", "cY"]].apply(pd.to_numeric)
	print(data_df)
 
	# Add Phase variable based on before/after minimum height
	data_df['Phase'] = pd.NA
	for index, row in data_df.iterrows():
			 
		# Get index of minimum cY value
		cutoff = data_df[data_df['Rep'] == row['Rep']]['cY'].idxmin()
	
		# If index is less than the index of min cY, then Eccentric (going down)
		if index < cutoff:
			data_df.loc[index, 'Phase'] = 'Eccentric'
	 
		# Else, index is greater than min cY, so Concentric (going back up)
		else:
			data_df.loc[index, 'Phase'] = 'Concentric'

	return data_df

def prepare_statistics(data_df):

	stats_dict = {
				"Total Reps": int(data_df['Rep'].max()),
				"Average Rep Velocity": float(data_df['Avg Velocity'].mean()),
				"Average Peak Velocity": float(data_df['Peak Velocity'].mean()),
				"Max Velocity Loss": float(data_df['Avg Velocity'].max()) * 100
	}
	return stats_dict

def generate_velocity_histogram(velocity_df):
      
  set_avg = velocity_df['Avg Velocity'][1:].mean()
  velocity_df['Velocity Difference'] = velocity_df['Peak Velocity'] - velocity_df['Avg Velocity']
  data_df = velocity_df[velocity_df['Rep'] > 0]

  fig = go.Figure(data= 
                    [
                      go.Bar(
                        name='Average Velocity', 
                        x=data_df['Rep'], y=data_df['Avg Velocity'], 
                        text='Avg Velocity',
                        hovertemplate =
                          '<b>Rep</b>: %{x}'+
                          '<br><b>Velocity</b>: %{y:.4f} m/s<br>',
                        ),
                    go.Bar(
                        name='Peak Velocity', 
                        x=data_df['Rep'], y=data_df['Velocity Difference'], 
                        text='Peak Velocity',
                        customdata = data_df['Peak Velocity'],
                        hovertemplate =
                          '<b>Rep</b>: %{x}'+
                          '<br><b>Velocity:</b> %{customdata:.4f}<br>',
                        ),
                    ]
                  )
  fig.update_layout(barmode='stack')
  fig.update_traces(marker_line_color='rgb(0, 0, 0)', marker_line_width=1.5)
  fig.update_layout(title_text='Average and Peak Velocity Across Set', title_x=0.5)
  fig.add_hline(y=set_avg, line_width=5, line_dash="dash", line_color="black",  
          annotation_text=f"Set Avg: {set_avg:.3f}", 
          annotation_position="bottom right")
  fig.update_xaxes(title_text="Rep", dtick=1, showgrid=False)
  fig.update_yaxes(title_text='Velocity (m/s)', dtick=0.05)
  return fig

def generate_set_bar_paths(data_df, reps_wanted='All Reps'):
	if reps_wanted == 'All Reps':
		plot_title = 'All Reps Bar Path'
	else:
		rep = reps_wanted[-1]
		plot_title = f'Rep {rep} Bar Path'
		data_df = data_df[data_df['Rep'] == int(rep)]

	min_x = data_df['cX'].min() - 150
	max_x = data_df['cX'].max() + 150
	fig = px.line(
			data_df, 
			x="cX", y="cY", 
			title=plot_title,
			line_dash='Phase',
			color='Rep',
			hover_name="Rep", 
			hover_data=
				{
					'cX':False,
					'cY':False,
					'Rep':False,
					'Phase': True,
					'Avg Velocity': ':.2f'
				}
	)
	
	fig.update_xaxes(range = [min_x, max_x], visible=False)
	fig.update_yaxes(visible=False)
	# strip down the rest of the plot
	fig.update_layout(
	showlegend=True,
	plot_bgcolor="white",
	margin=dict(t=50,l=10,b=10,r=10)
	)

	first_coords = data_df.groupby('Rep').first()
	fig.add_trace(
		go.Scatter(
			x=first_coords['cX'],
			y=first_coords['cY'],
			marker=dict(color="black", size=12),
			mode="markers",
			hovertemplate= '<b>%{text}</b>',
			text = ['Rep {}'.format(i + 1) for i in range(len(first_coords))],
			name="Rep Start",
	))

	return fig