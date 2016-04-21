"""Example of kitti.raw usage."""
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import kittitools as kitti

__author__ = "Lee Clement"
__email__ = "lee.clement@robotics.utias.utoronto.ca"

# Change this to the directory where you store KITTI data
basedir = '/Users/leeclement/Desktop/datasets'

# Specify the dataset to load
date = '2011_09_26'
drive = '0019'

# Optionally, specify the frame range to load
frame_range = range(0, 20, 5)

# Load the data
# dataset = kitti.raw(basedir, date, drive)
dataset = kitti.raw(basedir, date, drive, frame_range)

# Calibration data is loaded automatically
print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

# Load some other data
dataset.load_timestamps()   # timestamps are parsed into datetime objects
dataset.load_oxts()         # OXTS packets are loaded as dictionaries
dataset.load_gray()         # left/right images are accessible as dictionaries
dataset.load_rgb()          # left/right images are accessible as dictionaries
dataset.load_velo()         # Each scan is a Nx4 array of [x,y,z,reflectance]

# Display some of the data
np.set_printoptions(precision=4, suppress=True)
print('\nDrive: ' + str(dataset.drive))
print('\nFrame range: ' + str(dataset.frame_range))
print('\nFirst timestamp: ' + str(dataset.timestamps[0]))
print('\nSecond IMU pose:\n' + str(dataset.oxts[1].T_w_imu))

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
