import math

import numpy as np
import scipy.ndimage

from vhog3d import hog3d

if __name__ == "__main__":
	
	# Test 1
	vox_volume = np.full((3,3,3),3)
	cell_size = 3
	block_size = 3
	theta_histogram_bins = 5
	phi_histogram_bins = 5
	print("Test 1 : ", hog3d(vox_volume, cell_size, block_size, theta_histogram_bins, phi_histogram_bins))


	# Test 2
	vox_volume = np.full((256,256,256),3)
	cell_size = 3
	block_size = 3
	theta_histogram_bins = 5
	phi_histogram_bins = 5
	print("Test 2 : ", hog3d(vox_volume, cell_size, block_size, theta_histogram_bins, phi_histogram_bins))

	
