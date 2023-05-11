import os
import base64
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import json
from datetime import datetime

UPLOAD_DIRECTORY = "C:\\Users\\neyth\\Desktop\\SeniorComps\\BarbellTrackingCode\\uploads"
DATE = datetime.now().strftime('%m/%d/%Y')

def save_file(subdir, name, content, set_num):
	"""Decode and store a file uploaded with Plotly Dash."""
	data = content.encode("utf8").split(b";base64,")[1]
	subdir_path = os.path.join(UPLOAD_DIRECTORY, subdir)
	# Check whether the specified path exists or not
	isExist = os.path.exists(subdir_path)
	if not isExist:
		# Create a new directory because it does not exist
		print("Creating subdirectory...")
		os.makedirs(subdir_path)
  
	path = os.path.join(subdir_path, name)
	with open(path, "wb") as fp:
		fp.write(base64.decodebytes(data))
  
	split_tup = os.path.splitext(name)
	data_file_name = f"{split_tup[0]}_data.json"
	data_path = os.path.join(subdir_path, data_file_name)
	with open(data_path, 'w', encoding='utf-8') as f:
		data = {
      			"name": name, 
         		"date": DATE,
           		"file_path": path,
             	"sets": set_num,
              	"status": "untracked",
               	"weight": 0,
                "reps": 0}
		json.dump(data, f, ensure_ascii=False, indent=4)
		print("JSON DUMPED!")
  


def uploaded_files(subdir):
	"""List the files in the upload directory."""
	files = []
	path = os.path.join(UPLOAD_DIRECTORY,subdir)
	for filename in os.listdir(path):
		file_path = os.path.join(path, filename)
		if os.path.isfile(file_path):
			files.append(filename)
	return files

def file_download_link(subdir, filename):
	"""Create a Plotly Dash 'A' element that downloads a file from the app."""
	location = os.path.join(UPLOAD_DIRECTORY,subdir, filename)
	return html.A(filename, href=location)