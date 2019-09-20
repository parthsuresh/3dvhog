import numpy as np
import math
from scipy.ndimage import convolve


def hog3d(vox_volume, cell_size, block_size, theta_histogram_bins, phi_histogram_bins, step_size=None):
    """
    Inputs

    vox_volume : a 	[x x y x z] numpy array defining voxels with values in the range 0-1
    cell_size : size of a 3d cell (int)
    block_size : size of a 3d block defined in cells
    theta_histogram_bins : number of bins to break the angles in the xy plane - 180 degrees
    phi_histogram_bins : number of bins to break the angles in the xz plane - 360 degrees
    step_size : OPTIONAL integer defining the number of cells the blocks should overlap by.
	"""

    if step_size is None:
        step_size = block_size

    c = cell_size
    b = block_size

    sx, sy, sz = vox_volume.shape

    num_x_cells = math.floor(sx / cell_size)
    num_y_cells = math.floor(sy / cell_size)
    num_z_cells = math.floor(sz / cell_size)

    # Get cell positions
    x_cell_positions = np.array(list(range(0, (num_x_cells * cell_size), cell_size)))
    y_cell_positions = np.array(list(range(0, (num_y_cells * cell_size), cell_size)))
    z_cell_positions = np.array(list(range(0, (num_z_cells * cell_size), cell_size)))

    # Get block positions
    x_block_positions = (x_cell_positions[0: num_x_cells: block_size])
    y_block_positions = (y_cell_positions[0: num_y_cells: block_size])
    z_block_positions = (z_cell_positions[0: num_z_cells: block_size])

    # Check if last block in each dimension has enough voxels to be a full block. If not, discard it.
    if x_block_positions[-1] > ((sx + 1) - (cell_size * block_size)):
        x_block_positions = x_block_positions[:-2]
    if y_block_positions[-1] > ((sy + 1) - (cell_size * block_size)):
        y_block_positions = y_block_positions[:-2]
    if z_block_positions[-1] > ((sz + 1) - (cell_size * block_size)):
        z_block_positions = z_block_positions[:-2]

    # Number of blocks
    num_x_blocks = len(x_block_positions)
    num_y_blocks = len(y_block_positions)
    num_z_blocks = len(z_block_positions)

    # Create 3D gradient vectors
    # X filter and vector
    x_filter = np.zeros((3, 3, 3))
    x_filter[0, 1, 1], x_filter[2, 1, 1] = 1, -1
    x_vector = convolve(vox_volume, x_filter, mode='constant', cval=0)

    # Y filter and vector
    y_filter = np.zeros((3, 3, 3))
    y_filter[1, 0, 0], y_filter[1, 2, 0] = 1, -1
    y_vector = convolve(vox_volume, y_filter, mode='constant', cval=0)

    # Z filter and vector
    z_filter = np.zeros((3, 3, 3))
    z_filter[1, 1, 0], z_filter[1, 1, 2] = 1, -1
    z_vector = convolve(vox_volume, z_filter, mode='constant', cval=0)

    magnitudes = np.zeros([sx, sy, sz])
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                magnitudes[i, j, k] = (x_vector[i, j, k] ** 2 + y_vector[i, j, k] ** 2 + z_vector[i, j, k] ** 2) ** (
                    0.5)

    # Voxel Weights
    kernel_size = 3
    voxel_filter = np.full((kernel_size, kernel_size, kernel_size), 1 / (kernel_size * kernel_size * kernel_size))
    weights = convolve(vox_volume, voxel_filter, mode='constant', cval=0)
    weights = weights + 1

    # Gradient vector
    grad_vector = np.zeros((sx, sy, sz, 3))
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                grad_vector[i, j, k, 0] = x_vector[i, j, k]
                grad_vector[i, j, k, 1] = y_vector[i, j, k]
                grad_vector[i, j, k, 2] = z_vector[i, j, k]

    theta = np.zeros((sx, sy, sz))
    phi = np.zeros((sx, sy, sz))
    for i in range(sx):
        for j in range(sy):
            for k in range(sz):
                theta[i, j, k] = math.acos(grad_vector[i, j, k, 2])
                phi[i, j, k] = math.atan2(grad_vector[i, j, k, 1], grad_vector[i, j, k, 0])
                phi[i, j, k] += math.pi

    # Binning
    b_size_voxels = int(c * b)
    t_hist_bins = math.pi / theta_histogram_bins
    p_hist_bins = (2 * math.pi) / phi_histogram_bins

    block_inds = np.zeros((num_x_blocks * num_y_blocks * num_z_blocks, 3))
    i = 0
    for z_block in range(num_z_blocks):
        for y_block in range(num_y_blocks):
            for x_block in range(num_x_blocks):
                block_inds[i] = np.array(
                    [x_block_positions[x_block], y_block_positions[y_block], z_block_positions[z_block]])
                i += 1

    num_blocks = len(block_inds)
    error_count = 0
    features = []
    for i in range(num_blocks):
        print("Processing block: {:d} of {:d}".format(i + 1, num_blocks))
        full_empty = vox_volume[int(block_inds[i, 0]):int(block_inds[i, 0] + b_size_voxels),
                     int(block_inds[i, 1]):int(block_inds[i, 1] + b_size_voxels),
                     int(block_inds[i, 2]):int(block_inds[i, 2] + b_size_voxels)]

        if np.sum(full_empty) != 0 and np.sum(full_empty) != full_empty.size:
            feature = np.zeros((b, b, b, theta_histogram_bins, phi_histogram_bins))
            t_weights = weights[int(block_inds[i, 0]):int(block_inds[i, 0] + b_size_voxels),
                        int(block_inds[i, 1]):int(block_inds[i, 1] + b_size_voxels),
                        int(block_inds[i, 2]):int(block_inds[i, 2] + b_size_voxels)]
            t_magnitudes = magnitudes[int(block_inds[i, 0]):int(block_inds[i, 0] + b_size_voxels),
                           int(block_inds[i, 1]):int(block_inds[i, 1] + b_size_voxels),
                           int(block_inds[i, 2]):int(block_inds[i, 2] + b_size_voxels)]
            t_theta = theta[int(block_inds[i, 0]):int(block_inds[i, 0] + b_size_voxels),
                      int(block_inds[i, 1]):int(block_inds[i, 1] + b_size_voxels),
                      int(block_inds[i, 2]):int(block_inds[i, 2] + b_size_voxels)]
            t_phi = phi[int(block_inds[i, 0]):int(block_inds[i, 0] + b_size_voxels),
                    int(block_inds[i, 1]):int(block_inds[i, 1] + b_size_voxels),
                    int(block_inds[i, 2]):int(block_inds[i, 2] + b_size_voxels)]

            for l in range(b_size_voxels):
                for m in range(b_size_voxels):
                    for n in range(b_size_voxels):
                        cell_pos_x = math.ceil(l / c) - 1
                        cell_pos_y = math.ceil(m / c) - 1
                        cell_pos_z = math.ceil(n / c) - 1

                        hist_pos_theta = math.ceil(t_theta[l, m, n] / t_hist_bins) - 1
                        hist_pos_phi = math.ceil(t_phi[l, m, n] / p_hist_bins) - 1

                        if phi_histogram_bins >= hist_pos_phi > 0 and theta_histogram_bins >= hist_pos_theta > 0:
                            feature[cell_pos_x, cell_pos_y, cell_pos_z, hist_pos_theta, hist_pos_phi] += (
                                    t_magnitudes[l, m, n] * t_weights[l, m, n])
                        else:
                            error_count += 1

            feature = np.reshape(feature, ((b * b * b), theta_histogram_bins, phi_histogram_bins))
            l2 = np.linalg.norm(feature)
            if l2 != 0:
                norm_feature = feature / l2
            else:
                norm_feature = feature
            norm_feature = np.reshape(norm_feature, ((b * b * b), (theta_histogram_bins * phi_histogram_bins)))

            features.append(norm_feature)

    features = np.array(features)

    return features
