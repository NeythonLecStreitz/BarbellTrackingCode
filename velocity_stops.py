import pandas as pd


# Assuming this set was to failure, the final rep had 0 in the tank, thus we correlate that with 10 RPE and build down from there.
def generate_rpe_table(rtf_data, filepath):
	rtf_data = rtf_data[['Rep', 'Avg Velocity']]
	length = rtf_data.shape[0]

	rtf_data['Reps Left'] = rtf_data.apply(lambda row: int(length - row['Rep']), axis=1)
	rtf_data['RPE'] = rtf_data.apply(lambda row: int(10 - row['Reps Left']) 
									 if (row['Reps Left'] < 6 ) else pd.NA, axis=1)
	
	rtf_data.to_csv(filepath, sep=',')
	return rtf_data

def calculate_rpe(rtf_data, current_data):
    
    # Get current velocity
	current_vel = current_data['Avg Velocity'][0]
	
	# Get first occurance of row with velocity at current velocity
	curr_row = rtf_data.loc[rtf_data['Avg Velocity'] - current_vel < 0.01].iloc[0]
	velocity_stop = curr_row['Avg Velocity']
 
	# Get corresponding reps in reserve (RIR) and Rate of Percieved Effort (RPE)
	rir = curr_row['Reps Left']
	rpe = curr_row['RPE']
 
	return rir, rpe, velocity_stop