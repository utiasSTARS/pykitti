"""Example of pykitti.odometry usage."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pykitti

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = '/Users/leeclement/Desktop/odometry/dataset'

# Specify the dataset to load
sequence = '01'

# Optionally, specify the frame range to load
frame_range = range(0, 20, 5)

# Load the data
# dataset = pykitti.odometry(basedir, sequence)
dataset = pykitti.odometry(basedir, sequence, frame_range)

# Load some data
dataset.load_calib()        # Calibration data are accessible as named tuples
dataset.load_timestamps()   # Timestamps are parsed into timedelta objects
dataset.load_poses()        # Ground truth poses are loaded as 4x4 arrays
dataset.load_gray()         # Left/right images are accessible as named tuples
dataset.load_rgb()          # Left/right images are accessible as named tuples
dataset.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]

# Display some of the data
np.set_printoptions(precision=4, suppress=True)
print('\nSequence: ' + str(dataset.sequence))
print('\nFrame range: ' + str(dataset.frame_range))

print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
print('\nSecond ground truth pose:\n' + str(dataset.T_w_cam0[1]))

f, ax = plt.subplots(2, 2, figsize=(15, 5))
ax[0, 0].imshow(dataset.gray[0].left, cmap='gray')
ax[0, 0].set_title('Left Gray Image (cam0)')

ax[0, 1].imshow(dataset.gray[0].right, cmap='gray')
ax[0, 1].set_title('Right Gray Image (cam1)')

ax[1, 0].imshow(dataset.rgb[0].left)
ax[1, 0].set_title('Left RGB Image (cam2)')

ax[1, 1].imshow(dataset.rgb[0].right)
ax[1, 1].set_title('Right RGB Image (cam3)')

f2 = plt.figure()
ax2 = f2.add_subplot(111, projection='3d')
# Plot every 100th point so things don't get too bogged down
velo_range = range(0, dataset.velo[2].shape[0], 100)
ax2.scatter(dataset.velo[2][velo_range, 0],
            dataset.velo[2][velo_range, 1],
            dataset.velo[2][velo_range, 2],
            c=dataset.velo[2][velo_range, 3],
            cmap='gray')
ax2.set_title('Third Velodyne scan (subsampled)')

plt.show()
