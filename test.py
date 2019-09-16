import math

import numpy as np
import scipy.ndimage

from vhog3d import hog3d

import pdb
pdb.set_trace()

if __name__ == "__main__":
	
	# Test 1
	vox_volume = np.full((27,27,27),0.5)
	cell_size = 3
	block_size = 3
	theta_histogram_bins = 5
	phi_histogram_bins = 5
	hog3d(vox_volume, cell_size, block_size, theta_histogram_bins, phi_histogram_bins)
	print("Test 1 : ", theta)


	# Test 2
	vox_volume = np.full((256,256,256),0.5)
	cell_size = 3
	block_size = 3
	theta_histogram_bins = 5
	phi_histogram_bins = 5
	grad_vec = hog3d(vox_volume, cell_size, block_size, theta_histogram_bins, phi_histogram_bins)
	print("Test 2 : ", grad_vec)

	
