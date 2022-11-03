import numpy as np
import cv2, PIL
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import argparse

def generate_markers(marker_size=4, total_markers=50, grid_size=(1, 1)):

	key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
	aruco_dict = aruco.Dictionary_get(key)

	fig = plt.figure()
	nx, ny = grid_size
	for i in range(1, nx*ny+1):
		ax = fig.add_subplot(ny,nx, i)
		img = aruco.drawMarker(aruco_dict,i, 700)
		plt.imshow(img, cmap = mpl.cm.gray, interpolation = "nearest")
		ax.axis("off")

	plt.savefig(f"generated_aruCo_{nx}X{ny}.pdf")
	plt.show()
    
	
def main():
	# construct the argument parse and parse the arguments on script call
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--marker_size", type=int, default=5,
				help="number of bits in marker")
	ap.add_argument("-t", "--total_markers", type=int, default=50,
				help="number of total markers in the dictionary")
	
	ap.add_argument("-g", "--grid_size", type=int, default=1,
				help="number of markers to print in a grid")
	args, unknown = ap.parse_known_args()
	args_dict = vars(args)
 
	marker_size = args_dict["marker_size"]
	total_markers = args_dict["total_markers"]
	grid_size = args_dict["grid_size"]
 
	generate_markers(marker_size, total_markers, (grid_size, grid_size))
	
if __name__ == '__main__':
	main()