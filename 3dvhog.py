import numpy as np
import math

def hog3d(vox_volume, cell_size, block_size, theta_histogram_bins, phi_histogram_bins, step_size=block_size):

	"""
	Inputs

	vox_volume : a 	[x x y x z] numpy array defining voxels with values in the range 0-1
	cell_size : size of a 3d cell (int)
	block_size : size of a 3d block defined in cells
	theta_histogram_bins : number of bins to break the angles in the xy plane - 180 degrees
	phi_histogram_bins : number of bins to break the angles in the xz plane - 360 degrees
	step_size : OPTIONAL integer defining the number of cells the blcoks should overlap by. If same/larger than block size, blocks will not overlap
	"""

	c = cell_size
	b = block_size

	sx, sy, sz = vox_volume.shape
	
	num_x_cells = math.floor(sx / cell_size)
	num_y_cells = math.floor(sy / cell_size)
	num_z_cells = math.floor(sz / cell_size)

	#Get cell positions
	x_cell_positions = np.linspace(0, (num_x_cells*cell_size), num_x_cells)
	y_cell_positions = np.linspace(0, (num_y_cells*cell_size), num_y_cells)
	z_cell_positions = np.linspace(0, (num_z_cells*cell_size), num_z_cells) 

	#Get block positions
	x_block_positions = x_cell_positions[0 : num_x_cells : block_size]
	y_block_positions = y_cell_positions[0 : num_y_cells : block_size]
	z_block_positions = z_cell_positions[0 : num_z_cells : block_size]

	#Check if last block in each dimension has enough voxels to be a full block
	if (x_block_positions[-1] > (sx -(cell_size * block*size)):
		x_block_positions = x_block_positions[:-1]
	if (y_block_positions[-1] > (sy -(cell_size * block*size)):
		y_block_positions = y_block_positions[:-1]
	if (z_block_positions[-1] > (sz -(cell_size * block*size)):
		z_block_positions = z_block_positions[:-1]

	

