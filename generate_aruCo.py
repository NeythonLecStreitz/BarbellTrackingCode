import numpy as np
import cv2, PIL
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

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
    generate_markers(marker_size=4, total_markers=50, grid_size=(4, 4))
    
if __name__ == '__main__':
	main()